
import asyncio
import importlib
import os
import random
from collections import OrderedDict

import numpy as np
from _agent._utils.metrics import Metrics
from _utils import utils
from _utils.drl_utils import robust_argmax
from _utils.drl_utils import PPO_ExperienceReplay, EarlyStopper
import asyncio

import sqlalchemy
from sqlalchemy import MetaData, Column
import dataset
import ast

from itertools import product
import tensorflow as tf
from tensorflow import keras as k
import tensorflow_probability as tfp


class Trader:
    """This trader uses the proximal policy optimization algorithm (PPO) as proposed in https://arxiv.org/abs/1707.06347.
    Any liberties and further modifications to the algorithm will be attempted to be documented here
    Impelmentation is inspired by the torch implementation from cleanRL and by
    https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/tf2/agent.py
    """
    def __init__(self, bid_price, ask_price, **kwargs):
        # Some utility parameters
        self.__participant = kwargs['trader_fns']
        self.study_name = kwargs['study_name']
        self.status = {
            'weights_loading': False
        }

        # Initialize metrics tracking
        self.track_metrics = kwargs['track_metrics']
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

        # Generate actions
        #I think we could stay with quantized actions, however I'd like to start testing on the non-quantized version ASAP so we do non quantized
        self.actions = {}
        self.flip = {}
        for action in kwargs['actions']:
            if action == 'price':
                self.actions['price'] = {'min': ask_price, 'max': bid_price}
                self.flip['price'] = np.random.choice([True, False])
            if action == 'quantity':
                self.actions['quantity'] = {'min': -1.6*17.0, 'max': 1.6*17.0}
                self.flip['quantity'] = np.random.choice([True, False])
            if action == 'storage' and 'storage' in self.__participant:
                self.actions['storage'] = {'min': -20.0, 'max': 20.0}
                self.flip['storage'] = np.random.choice([True, False])

        # initialize all the counters we need
        self.train_step = 0
        self.total_step = 0
        self.gen = 0

        #prepare TB functionality, to open TB use the terminal command: tensorboard --logdir <dir_path>
        cwd = os.getcwd()
        logs_path = os.path.join(cwd, 'PPOBuyers_vs_ExpertSellers')
        experiment_path = os.path.join(logs_path, self.study_name)
        trader_path = os.path.join(experiment_path, self.__participant['id'])

        self.summary_writer = tf.summary.create_file_writer(trader_path)

        # Initialize learning parameters
        self.learning = kwargs['learning']
        reward_function = kwargs['reward_function']
        if reward_function:
            self._rewards = importlib.import_module('_agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

        #ToDo: test if having parameter sharing helps here?

        # Hyperparameters
        self.alpha_critic = kwargs['alpha_critic']
        self.alpha_actor = kwargs['alpha_actor']
        self.batch_size = kwargs['batch_size'] #bigger is smoother, but might require a bigger replay buffer
        self.policy_clip = kwargs['policy_clip'] #ToDo: look into some type of entropy regularization to guide exploration!
        self.max_train_steps = kwargs['max_train_steps'] #ToDO: figure out reasonable default valye and make this a hyperparam
        self.kl_stop = kwargs['kl_stop'] #according to baselines tends to be bewteen 0.01 and 0.05
        self.critic_patience = kwargs['critic_patience']
        self.entropy_reg = kwargs['entropy_reg']
        self.gamma = kwargs['gamma']
        self.gae_lambda = kwargs['gae_lambda']
        self.normalize_advantages = kwargs['normalize_advantages']
        self.use_early_stop_actor = kwargs['use_early_stop_actor']
        self.use_early_stop_critic = kwargs['use_early_stop_critic']
        self.replay_buffer_length = kwargs['experience_replay_buffer_length']

        self.warmup_actor = kwargs['warmup_actor']

        self.experience_replay_buffer = PPO_ExperienceReplay(max_length=self.replay_buffer_length,
                                                            action_types=self.actions,
                                                             multivariate=True,
                                                            )

        self.ppo_actor_dist, self.ppo_actor = self.__build_actor(distribution='Beta')
        self.ppo_actor.compile(optimizer=k.optimizers.Adam(learning_rate=self.alpha_actor,),)

        self.ppo_critic = self.__build_critic()
        self.ppo_critic.compile(optimizer=k.optimizers.Adam(learning_rate=self.alpha_critic,),)
        self.ppo_critic_loss = k.losses.MeanSquaredError()


        # Buffers we need for logging stuff before putting into the PPo Memory
        self.actions_buffer = {}
        # self.pi_history = {}
        self.log_prob_buffer = {}
        self.value_buffer = {}
        self.state_buffer = {}

        #logs we need for plotting
        self.rewards_history = []
        self.value_history = []
        self.actions_history = {}
        self.pdf_history = {}
        for action in self.actions:
            self.actions_history[action] = []
            # self.pdf_history[action] = {}
            # for param in ['loc', 'scale']:
            #     self.pdf_history[action][param] = []

    def __init_metrics(self):
        import sqlalchemy
        '''
        Initializes metrics to record into database
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        self.metrics.add('rewards', sqlalchemy.Float)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)
        if 'storage' in self.__participant:
            self.metrics.add('storage_soc', sqlalchemy.Float)

    def __build_actor(self, distribution='Beta'):
        num_inputs = 2 if 'storage' not in self.actions else 3
        num_hidden = 32 if 'storage' not in self.actions else 64
        num_hidden_layers = 2 #lets see how far we get with this first
        num_actions = len(self.actions)

        initializer = k.initializers.HeNormal()

        inputs = k.layers.Input(shape=(num_inputs,), name='Actor_Input')

        internal_signal = inputs
        for hidden_layer_number in range(num_hidden_layers): #hidden layers
            internal_signal = k.layers.Dense(num_hidden,
                                           activation="elu",
                                           kernel_initializer=initializer,
                                           name='Actor_Hidden' + str(hidden_layer_number))(internal_signal)

        pi = {}
        if distribution == 'Beta' or distribution =='Dirichlet':
            self.rescale_actions = True
            self.crop_actions = False
            if num_actions > 1: #Choose Dirichlet distribution Head
                concentration = k.layers.Dense(num_actions,
                                          activation=None,
                                          kernel_initializer=initializer,
                                          name='concentration')(internal_signal)
                concentration = tf.where(tf.math.greater(concentration,1.0),  #essentially just huber function it so its bigger than 0
                                         tf.abs(concentration),
                                         tf.square(concentration))
                concentration = concentration + 1e-10 #so we prevent it from being 0
                pi['concentration'] = concentration

                ppo_actor_dist = tfp.distributions.Dirichlet

            else: #Choose Beta Head
                concentration1 = k.layers.Dense(1,
                                          activation=None,
                                          kernel_initializer=initializer,
                                          name='concentration1')(internal_signal)
                concentration1 = tf.where(tf.math.greater(concentration1,1.0),  #essentially just huber function it so its bigger than 0
                                         tf.abs(concentration1),
                                         tf.square(concentration1))
                concentration1 = concentration1 + 1e-10 #so we prevent it from being 0
                pi['concentration1'] = concentration1

                concentration0 = k.layers.Dense(1,
                                          activation=None,
                                          kernel_initializer=initializer,
                                          name='concentration0')(internal_signal)
                concentration0 = tf.where(tf.math.greater(concentration0,1.0),  #essentially just huber function it so its bigger than 0
                                         tf.abs(concentration0),
                                         tf.square(concentration0))
                concentration0 = concentration0 + 1e-10 #so we prevent it from being 0
                pi['concentration0'] = concentration0

                ppo_actor_dist = tfp.distributions.Beta
        elif distribution == 'Normal' or distribution == 'MultivariateNormal':
            self.rescale_actions = False
            self.crop_actions = True

            scale = k.layers.Dense(num_actions,
                                      activation=None,
                                      kernel_initializer=initializer,
                                      name='scale')(internal_signal)
            scale = tf.where(tf.math.greater(scale,1.0),  #essentially just huber function it so its bigger than 0
                                     tf.abs(scale),
                                     tf.square(scale))
            scale = scale + 1e-10 #so we prevent it from being 0


            loc = k.layers.Dense(num_actions,
                                      activation=None,
                                      kernel_initializer=initializer,
                                      name='loc')(internal_signal)

            if num_actions > 1:
                pi['scale_diag'] = scale
                pi['loc'] = loc
                ppo_actor_dist = tfp.distributions.MultivariateNormalDiag
            else:
                pi['scale'] = scale
                pi['loc'] = loc
                ppo_actor_dist = tfp.distributions.Normal
        else:
            print('provided faulty distribution type')

        return ppo_actor_dist, k.Model(inputs=inputs, outputs=pi)

    def __build_critic(self):
        num_inputs = 2 if 'storage' not in self.actions else 3
        num_hidden = 32 if 'storage' not in self.actions else 64
        num_hidden_layers = 2 #lets see how far we get with this first

        initializer = k.initializers.HeNormal()
        inputs = k.layers.Input(shape=(num_inputs,), name='Actor_Input')

        internal_signal = inputs
        for hidden_layer_number in range(num_hidden_layers): #hidden layers
            internal_signal = k.layers.Dense(num_hidden,
                                           activation="elu",
                                           kernel_initializer=initializer,
                                           name='Actor_Hidden' + str(hidden_layer_number))(internal_signal)

        value = k.layers.Dense(1,
                             activation=None,
                             kernel_initializer=initializer,
                             name='ValueHead')(internal_signal)

        return k.Model(inputs=inputs, outputs=value)

    def anneal(self, parameter:str, adjustment, mode:str='multiply', limit=None):
        if not hasattr(self, parameter):
            return False

        if mode not in ('subtract', 'multiply', 'set'):
            return False

        param_value = getattr(self, parameter)
        if mode == 'subtract':
            param_value = max(0, param_value - adjustment)

        elif mode == 'multiply':
            param_value *= adjustment

        elif mode == 'set':
            param_value = adjustment

        if limit is not None:
            param_value = max(param_value, limit)

        setattr(self, parameter, param_value)

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        if not self.learning:
            return
        current_round = self.__participant['timing']['current_round']
        next_settle = self.__participant['timing']['next_settle']
        round_duration = self.__participant['timing']['duration']

        reward = await self._rewards.calculate()
        if reward is None:
            await self.metrics.track('rewards', reward)
            return
        # align reward with action timing
        # in the current market setup the reward is for actions taken 3 steps ago
        # if self._rewards.type == 'net_profit':
        reward_time_offset = current_round[1] - next_settle[1] - round_duration
        reward_timestamp = current_round[1] + reward_time_offset

        await self.metrics.track('rewards', reward)
        self.rewards_history.append(reward)

        if reward_timestamp in self.state_buffer and reward_timestamp in self.actions_buffer:  # we found matching ones, buffer and pop

            self.experience_replay_buffer.add_entry(states=self.state_buffer[reward_timestamp],
                                                    actions_taken=self.actions_buffer[reward_timestamp],
                                                    log_probs=self.log_prob_buffer[reward_timestamp],
                                                    values=self.value_buffer[reward_timestamp],
                                                    rewards=reward,
                                                    episode=self.gen)

            self.actions_buffer.pop(reward_timestamp) #ToDo: check if we can pop into the above function, would look nicer
            self.log_prob_buffer.pop(reward_timestamp)
            self.value_buffer.pop(reward_timestamp)
            self.state_buffer.pop(reward_timestamp)

            if self.experience_replay_buffer.should_we_learn():
                advantage_calulated = await self.experience_replay_buffer.calculate_advantage(gamma=self.gamma,
                                                                                              gae_lambda=self.gae_lambda,
                                                                                              normalize=self.normalize_advantages,
                                                                                              entropy_reg=self.entropy_reg
                                                                                              )  # ToDo: check once more on a different immplementation if this is right
                buffer_indexed = await self.experience_replay_buffer.generate_availale_indices()  # so we can caluclate the batches faster
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, func=self.train_RL_agent)

    def train_RL_agent(self):

        early_stop_critic = False #ToDo: implement these
        early_stop_actor = False
        max_train_steps = self.train_step + self.max_train_steps

        critic_stopper = EarlyStopper(patience=self.critic_patience)
        if self.warmup_actor:
            actor_stopper = EarlyStopper(patience=self.critic_patience)

        while self.train_step <= max_train_steps and not (early_stop_actor and early_stop_critic):

            batch = self.experience_replay_buffer.fetch_batch(batchsize=self.batch_size) #to see the batch data structure check this method
            states = tf.convert_to_tensor(batch['states'], dtype=tf.float32)

            if not early_stop_critic:
                with tf.GradientTape() as tape_critic:
                #calculate critic loss and backpropagate

                    critic_Vs = self.ppo_critic(states)
                    critic_Vs = tf.squeeze(critic_Vs, axis=-1)
                    G = tf.convert_to_tensor(batch['Return'], dtype=tf.float32)
                    losses_critic = self.ppo_critic_loss(critic_Vs, G)

                    # calculate the stopping crtierions
                    if self.use_early_stop_critic:
                        early_stop_critic, self.ppo_critic = critic_stopper.check_iteration(losses_critic.numpy(),
                                                                                        self.ppo_critic)
                    # log
                    with self.summary_writer.as_default():
                        tf.summary.scalar('critic_loss', losses_critic, step=self.train_step)

                    #early stop or learn
                    if early_stop_critic:
                        # print('stopping training the critic early, after ', self.train_step + self.max_train_steps - max_train_steps)
                        pass
                    else:
                        critic_vars = self.ppo_critic.trainable_variables
                        critic_grads = tape_critic.gradient(losses_critic, critic_vars)
                        self.ppo_critic.optimizer.apply_gradients(zip(critic_grads, critic_vars))

            if not early_stop_actor:

                advantages = tf.convert_to_tensor(batch['advantages'], dtype=tf.float32)
                log_probs_old = tf.convert_to_tensor(batch['log_probs'], dtype=tf.float32)
                a_taken = tf.convert_to_tensor(batch['actions_taken'], dtype=tf.float32)
                if not self.warmup_actor:
                    with tf.GradientTape() as tape_actor:
                        pi_batch = self.ppo_actor(states)
                        dist = self.ppo_actor_dist(**pi_batch)

                        log_probs_new = dist.log_prob(a_taken)
                        # check = tf.reduce_sum(log_probs_new).numpy()
                        # if np.isnan(check) or np.isinf(check):
                        #     probs = dist.prob(a_taken)
                        #     print('shit')
                        #This is how baselines does it
                        log_probs_new = tf.squeeze(log_probs_new)
                        log_probs_old = tf.squeeze(log_probs_old)

                        ratio = tf.exp(log_probs_new - log_probs_old)  # pi(a|s) / pi_old(a|s)
                        # entropy = -dist.entropy()
                        entropy = -log_probs_new
                        soft_advantages = (1.0 - self.entropy_reg) * advantages + self.entropy_reg * entropy

                        clipped_ratio = tf.clip_by_value(ratio, 1-self.policy_clip, 1+self.policy_clip)
                        weighted_ratio = clipped_ratio * soft_advantages
                        loss_actor = -tf.math.minimum(ratio*soft_advantages, weighted_ratio)
                        # loss_actor = tf.clip_by_value(loss_actor, clip_value_min=-100, clip_value_max=100)
                        loss_actor = tf.math.reduce_mean(loss_actor)

                        # PPO early stopping as implemented in baselines
                        approx_kl = tf.math.reduce_mean(log_probs_old - log_probs_new)

                        # collect entropy because why not. If this keeps growing we might have a too small memory and too smal batchsize
                        entropy = tf.reduce_mean(-log_probs_new)

                        # log
                        with self.summary_writer.as_default():
                            tf.summary.scalar('actor_loss', loss_actor, step=self.train_step)
                            tf.summary.scalar('approx_KL', approx_kl, step=self.train_step)
                            tf.summary.scalar('entropy', entropy, step=self.train_step)

                        # early stopping condition or keep training, consider having this a running avg of 5 or sth?
                        if self.use_early_stop_actor:
                            if tf.math.reduce_mean(approx_kl).numpy() > 1.5 * self.kl_stop:
                                early_stop_actor = True
                                # print('stopping actor training due to exceeding KL-divergence tolerance with approx KL of', approx_kl,' after ' , self.train_step + self.max_train_steps - max_train_steps)

                        if not early_stop_actor:
                            # Backpropagation
                            actor_vars = self.ppo_actor.trainable_variables
                            actor_grads = tape_actor.gradient(loss_actor, actor_vars)
                            self.ppo_actor.optimizer.apply_gradients(zip(actor_grads, actor_vars))

                else:

                    with tf.GradientTape() as tape_warmup:
                        pi_new = self.ppo_actor(states)
                        losses_warmup = []

                        for key in pi_new:
                            targets = tf.ones(shape=pi_new[key].shape)
                            loss = self.ppo_critic_loss(targets, pi_new[key])
                            losses_warmup.append(loss)
                        losses_warmup = tf.reduce_sum(losses_warmup)

                    # calculate the stopping crtierions
                    if self.use_early_stop_critic:
                        early_stop_actor, self.ppo_actor = actor_stopper.check_iteration(losses_warmup.numpy(),
                                                                                        self.ppo_actor)
                    with self.summary_writer.as_default():
                        tf.summary.scalar('actor_wamrup_loss', losses_warmup, step=self.train_step)

                    # early stop or learn
                    if early_stop_actor:
                        # print('stopping actor warmup early, after ', self.train_step + self.max_train_steps - max_train_steps)
                        pass
                    else:
                        actor_vars = self.ppo_actor.trainable_variables
                        actor_grads = tape_warmup.gradient(losses_warmup, actor_vars)
                        self.ppo_actor.optimizer.apply_gradients(zip(actor_grads, actor_vars))

            self.train_step = self.train_step + 1

        #clear the buffer after we learned it
        self.experience_replay_buffer.clear_buffer()
        self.train_step = max_train_steps #to make sure we log all the algorithms that might be running in parallel at the same scales

    async def __sample_pi(self, pi_dict): #ToDo: check how this behaves for batches of actions and for single actions, we want this to be consisten!

        dist = self.ppo_actor_dist(**pi_dict)

        a_dist = dist.sample(1)
        carinality = len(a_dist.get_shape())
        to_be_reduced = np.arange(carinality - 1, dtype=int).tolist()
        a_dist = tf.squeeze(a_dist, axis=to_be_reduced)
        a_dist = a_dist.numpy().tolist()

        if self.rescale_actions: #actions between 0 1nd 1
            min = 1e-10
            a_dist = tf.clip_by_value(a_dist, clip_value_min=min, clip_value_max=0.9999999)
            a_dist = a_dist.numpy().tolist()
        if self.crop_actions:
            keys = list(self.actions.keys())
            for action_index in range(len(keys)):
                min = self.actions[keys[action_index]]['min']
                max = self.actions[keys[action_index]]['max']
                a = tf.clip_by_value(a_dist[action_index], clip_value_min=min, clip_value_max=max)
                a_dist[action_index] = a.numpy().tolist()

        log_prob = dist.log_prob(a_dist)
        carinality = len(log_prob.get_shape())
        to_be_reduced = np.arange(carinality-1, dtype=int).tolist()
        log_prob = tf.squeeze(log_prob, axis=to_be_reduced)
        log_prob = log_prob.numpy().tolist()


        a_scaled = {}
        keys = list(self.actions.keys())
        for action_index in range(len(keys)):
            a = a_dist[action_index]
            if self.rescale_actions:
                if not self.flip[keys[action_index]]: #ToDo: find a better hack than this ...  this is horrible
                    min = self.actions[keys[action_index]]['min']
                    max = self.actions[keys[action_index]]['max']
                else:
                    max = self.actions[keys[action_index]]['min']
                    min = self.actions[keys[action_index]]['max']
                a = min + (a * (max - min))

            a_scaled[keys[action_index]] = a

        return a_scaled, log_prob, a_dist

    async def act(self, **kwargs):
        # Generate state (inputs to model):
        # - time(s)
        # - next generation
        # - next load
        # - battery stats (if available)

        current_round = self.__participant['timing']['current_round']
        next_settle = self.__participant['timing']['next_settle']
        next_generation, next_load = await self.__participant['read_profile'](next_settle)
        self.net_load = next_load - next_generation

        # if 'quantity' in self.actions: #pseudosmart quantities because so far we havent figure out how to manage the full action space...
        #     if self.net_load > 0:
        #         self.actions['quantity']['min'] = 0.0
        #         self.actions['quantity']['max'] = self.net_load
        #     else:
        #         self.actions['quantity']['min'] = self.net_load
        #         self.actions['quantity']['max'] = 0.0

        timezone = self.__participant['timing']['timezone']
        # current_round_end = utils.timestamp_to_local(current_round[1], timezone)
        # next_settle_end = utils.timestamp_to_local(next_settle[1], timezone)

        state = [
                 # np.sin(2 * np.pi * current_round_end.hour / 24),
                 # np.cos(2 * np.pi * current_round_end.hour / 24),
                 # np.sin(2 * np.pi * current_round_end.minute / 60),
                 # np.cos(2 * np.pi * current_round_end.minute / 60),
                 float(next_generation/17),
                 float(next_load/17)]

        if 'storage' in self.__participant:
            storage_schedule = await self.__participant['storage']['check_schedule'](next_settle)
            soc = storage_schedule[next_settle]['projected_soc_end']
            state.append(soc)

        state = np.array(state)
        pi_dict = self.ppo_actor(tf.expand_dims(state, axis=0))

        taken_action, log_prob, dist_action = await self.__sample_pi(pi_dict)

        value = self.ppo_critic(tf.expand_dims(state, axis=0))
        value = tf.squeeze(value, axis=0).numpy().tolist()[0]

        # lets log the stuff needed for the replay buffer
        self.state_buffer[current_round[1]] = state
        self.actions_buffer[current_round[1]] = dist_action
        self.log_prob_buffer[current_round[1]] = log_prob
        self.value_buffer[current_round[1]] = value
        self.value_history.append(value)
        # for action in self.actions:
        #     for param in pi_actions:
        #         self.pdf_history[action][param].append(pi_actions[param])

        actions = await self.decode_actions(taken_action, next_settle)

        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', next_load),
                self.metrics.track('next_settle_generation', next_generation))
            if 'storage' in self.actions:
                await self.metrics.track('storage_soc', self.__participant['storage']['info']()['state_of_charge'])
        return actions

    async def decode_actions(self, taken_action, next_settle):
        actions = dict()

        if 'price' in taken_action:
            price = taken_action['price']
            price = round(price, 4)
        else:
            price = 0.111

        if 'quantity' in taken_action:
            quantity = int(taken_action['quantity'])
        else:
            quantity = self.net_load

        if quantity > 0:
            actions['bids'] = {
                str(next_settle): {
                    'quantity': quantity,
                    'price': price
                }
            }
        elif quantity < 0:
            actions['asks'] = {
                'solar': {
                    str(next_settle): {
                        'quantity': -quantity,
                        'price': price
                    }
                }
            }

        if 'storage' in self.actions:
            actions['bess'] = {
                str(next_settle): taken_action['storage']
                }
        # print(actions)

        #log actions for later histogram plot
        for action in self.actions:
            self.actions_history[action].append(taken_action[action])
        return actions

    async def step(self):
        next_actions = await self.act()
        await self.learn()
        if self.track_metrics:
            await self.metrics.save(10000)
        # print(next_actions)
        self.total_step += 1
        return next_actions

    async def end_of_generation_tasks(self):
        # self.episode_reward_history.append(self.episode_reward)
        episode_G = sum(self.rewards_history)
        print(self.__participant['id'], 'episode reward:', episode_G)

        with self.summary_writer.as_default():
            tf.summary.scalar('Return' , episode_G, step= self.gen)
            tf.summary.histogram('Rewards during Episode', self.rewards_history, step=self.gen)
            tf.summary.histogram('Values',
                                 self.value_history,
                                 step=self.gen)

            for action in self.actions:
                tf.summary.histogram(action, self.actions_history[action], step=self.gen)


            # for layer in self.ppo_critic.layers:
            #     if layer.name not in ['Input']:
            #         for weights in layer.weights:
            #             tf.summary.histogram(weights.name, weights.numpy(), step=self.gen)
            #
            # for layer in self.ppo_actor.layers:
            #     if layer.name not in ['Input']:
            #         for weights in layer.weights:
            #             tf.summary.histogram(weights.name, weights.numpy(), step=self.gen)


        self.gen = self.gen + 1

    async def reset(self, **kwargs):
        self.state_buffer.clear()
        self.value_buffer.clear()
        self.actions_buffer.clear()
        self.log_prob_buffer.clear()

        self.rewards_history.clear()
        self.value_history.clear()
        for action in self.actions:
            self.actions_history[action].clear()
        #     for param in ['loc', 'scale']:
        #         self.pdf_history[action][param].clear()

        return True