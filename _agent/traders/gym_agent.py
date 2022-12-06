# from _clients.participants.participants import Residential

import tenacity
from datetime import datetime

from TREX_Core._agent._utils.metrics import Metrics
from TREX_Core._agent._utils.heuristics import PriceHeuristics, QuantityHeuristics
import asyncio
from TREX_Core._utils import jkson as json
# import serialize
from multiprocessing import shared_memory
import importlib


class Trader:
    """
    Class: Trader
    This class implements the gym compatible trader that is used in tandem with EPYMARL TREXEnv.

    """
    def __init__(self, **kwargs):
        """
        Initializes the trader using the parameters in the TREX_Core config that was selected.
        In particular, this sets up the connections to the shared lists that are established in EPYMARL TREXEnv or any
        other process that seeks to interact with TREX. These lists need to be already initialized before the gym trader
        attempts to connect with them.

        params: kwargs -> dictionary created from the config json file in TREX_Core._configs
        """
        # Some util stuffies
        print('GOT TO THE GYM_AGENT INIT')
        self.__participant = kwargs['trader_fns']
        self.status = {
            'weights_loading': False,
            'weights_loaded': False,
            'weights_saving': False,
            'weights_saved': True
        }

        ##### Setup the shared memory names based on config #####
        self.name = self.__participant['id']
        action_list_name = self.name + "_actions"
        observation_list_name = self.name + "_obs"
        reward_list_name = self.name + "_reward"
        '''
        Shared lists get initialized on TREXENV side, so all that the agents have to do is connect to their respective 
        observation and action lists. Agents dont have to worry about making the actions pretty, they just have to send
        them into the buffer. 
        '''
        self.shared_list_action = shared_memory.ShareableList(name=action_list_name)
        self.shared_list_observation = shared_memory.ShareableList(name=observation_list_name)
        self.shared_list_reward = shared_memory.ShareableList(name=reward_list_name)
        print('shared reward', self.shared_list_reward)
        print('shared action', self.shared_list_action)
        print('shared observation', self.shared_list_observation)
        #find the right default behaviors from kwargs['default_behaviors']
        self.observation_variables = kwargs['observations']

        #ToDo - Daniel - Think about a nicer way of doing this
        #decode actions, load heuristics if necessary
        self.action_type = kwargs['actions']
        for action in kwargs['actions']:
            if kwargs['actions'][action] != 'learned':
                heuristic_type = kwargs['actions'][action]
                if 'price' == action:
                    self.price_heuristic = PriceHeuristics(type=heuristic_type)
                elif 'quantity' == action:
                    self.quantity_heuristic = QuantityHeuristics(type=heuristic_type)
                elif 'storage' == action:
                    raise NotImplementedError
                else:
                    raise NotImplementedError


        # TODO: Find out where the action space will be defined: I suspect its not here
        # Initialize the agent learning parameters for the agent (your choice)
        # self.bid_price = kwargs['bid_price'] if 'bid_price' in kwargs else None
        # self.ask_price = kwargs['ask_price'] if 'ask_price' in kwargs else None

        ####### Metrics tracking initialization ########
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

        ###### Reward function intialization from config #######
        reward_function = kwargs['reward_function']
        if reward_function:
            self._rewards = importlib.import_module('TREX_Core._agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

       #  print('init done')
    def __init_metrics(self):
        import sqlalchemy
        '''
        Pretty self explanitory, this method resets the metric lists in 'agent_metrics' as well as zeroing the metrics dictionary. 
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)

        # if self.battery:
        #     self.metrics.add('battery_action', sqlalchemy.Integer)
        #     self.metrics.add('state_of_charge', sqlalchemy.Float)

    # Core Functions, learn and act, called from outside
    async def pre_process_obs(self, ts_obs):
        print('entered preprocessing')
        # ToDo: add histograms for observations
        self.obs_order = []
        data_for_tb = []

        obs_generation, obs_load = await self.__participant['read_profile'](ts_obs)

        observations_t = []
        if not hasattr(self, 'profile_stats'):
            self.profile_stats = await self.__participant['get_profile_stats']()

        if 'generation' in self.observation_variables:
            self.obs_order.append('generation')

            if self.profile_stats:
                avg_generation = self.profile_stats['avg_generation']
                stddev_generation = self.profile_stats['stddev_generation']
                z_next_generation = (obs_generation - avg_generation) / (stddev_generation + 1e-8)
                observations_t.append(z_next_generation)
            else:
                observations_t.append(obs_generation)

        if 'load' in self.observation_variables:
            self.obs_order.append('load')

            if self.profile_stats:
                avg_load = self.profile_stats['avg_consumption']
                stddev_load = self.profile_stats['stddev_consumption']
                z_next_load = (obs_load - avg_load) / (stddev_load + 1e-8)
                observations_t.append(z_next_load)
            else:
                observations_t.append(obs_load)

        # ToDo - Daniel - there should be an inbuilt conversion for these formats
        timestamp = ts_obs[0]
        dt = datetime.fromtimestamp(ts_obs[0])
        ts_to_minutes = 1/60
        ts_to_hour = ts_to_minutes*(1/60)
        ts_to_hour_in_day = ts_to_hour*(1/24)
        ts_to_daytype = ts_to_hour_in_day * (1 / 7)
        ts_to_day_in_year = ts_to_hour_in_day * (1 / 365)

        # ToDo - Daniel - get rid of ugly if loop
        if 'time_sin_hour' in self.observation_variables:
            self.obs_order.append('time_sin_hour')
            hour_in_day = timestamp *ts_to_hour_in_day
            time_sin_hour=np.sin(2 * np.pi *hour_in_day )
            observations_t.append(time_sin_hour)

        if 'time_cos_hour' in self.observation_variables:
            self.obs_order.append('time_cos_hour')
            hour_in_day = timestamp * ts_to_hour_in_day
            time_cos_hour =np.cos(2 * np.pi * hour_in_day)
            observations_t.append(time_cos_hour)

        if 'time_sin_day' in self.observation_variables:
            self.obs_order.append('time_sin_day')
            daytype = timestamp *ts_to_daytype
            time_sin_day=np.sin(2 * np.pi * daytype)
            observations_t.append(time_sin_day)

        if 'time_cos_day' in self.observation_variables:
            self.obs_order.append('time_cos_day')
            daytype = timestamp * ts_to_daytype
            time_cos_day=np.cos(2 * np.pi * daytype)
            observations_t.append(time_cos_day)

        if 'time_sin_dayinyear' in self.observation_variables:
            self.obs_order.append('time_sin_dayinyear')
            day_in_year = timestamp *ts_to_day_in_year
            time_sin_dayinyear = np.sin(2 * np.pi * day_in_year)
            observations_t.append(time_sin_dayinyear)

        if 'time_cos_dayinyear' in self.observation_variables:
            self.obs_order.append('time_cos_dayinyear')
            day_in_year = timestamp * ts_to_day_in_year
            time_cos_dayinyear=np.cos(2 * np.pi * day_in_year)
            observations_t.append(time_cos_dayinyear)

        if 'soc' in self.observation_variables:
            self.obs_order.append('soc')
            storage_schedule = await self.__participant['storage']['check_schedule'](ts_obs)
            soc = storage_schedule[ts_obs]['projected_soc_end']
            observations_t.append(soc)

        #ToDo - Daniel & Steven - get these from special market
        if 'avg_bid_price_ls' in self.observation_variables:
            raise NotImplementedError
            self.obs_order.append('avg_bid_price_ls')
        if 'avg_ask_price_ls' in self.observation_variables:
            raise NotImplementedError
            self.obs_order.append('avg_ask_price_ls')
        if 'avg_bid_quantity_ls' in self.observation_variables:
            raise NotImplementedError
            self.obs_order.append('avg_bid_quantity_ls')
        if 'avg_ask_quantity_ls' in self.observation_variables:
            raise NotImplementedError
            self.obs_order.append('avg_bid_quantity_ls')

        return observations_t

    async def act(self, **kwargs):
        """


        """

        '''
        actions are none so far
        ACTIONS ARE FOR THE NEXT settle!!!!!

        actions = {
            'bess': {
                time_interval: scheduled_qty
            },
            'bids': {
                time_interval: {
                    'quantity': qty,
                    'price': dollar_per_kWh
                }
            },
            'asks': {
                source:{
                     time_interval: {
                        'quantity': qty,
                        'price': dollar_per_kWh?
                     }
                 }
            }
        sources inclued: 'solar', 'bess'
        Actions in the shared list 
        [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
        }
        '''
        print("in agent.act")
        ##### Initialize the actions
        print('entered act')
        actions = {}
        # TODO: these are going to have to go into the obs_creation method, waiting on daniel for these
        bid_price = 0.0
        bid_quantity = 0.0
        solar_ask_price = 0.0
        solar_ask_quantity = 0.0
        bess_ask_price = 0.0
        bees_ask_quantity = 0.0

        # Timing information
        timing = self.__participant['timing']
        self.current_round = self.__participant['timing']['current_round']
        self.round_duration = self.__participant['timing']['duration']
        self.next_round = (self.current_round[0]+self.round_duration, self.current_round[1]+self.round_duration)
        self.last_settle = self.__participant['timing']['last_settle']
        self.next_settle = self.__participant['timing']['next_settle']
        self.last_round = timing['last_round']

        #the timestep we observe vs the timestep we act on
        ts_obs = self.next_settle
        ts_act = self.next_settle

        obs_t = await self.pre_process_obs(ts_obs)

        print('Agent Observations', obs_t)
        await self.obs_to_shared_memory(obs_t)

        #### Send rewards into reward buffer:
        reward = await self._rewards.calculate()
        # send the reward value into the reward buffer:
        await self.r_to_shared_memory(reward)

        '''
        #########################################################################
        it is here that we wait for the action values to be written from epymarl
        #########################################################################
        '''
        # wait for the actions to come from EPYMARL
        await self.read_action_values()
        # actions come in with a set order, they will need to be split up

        self.decode_actions({}, ts_act)

        # TODO: these need to be set and coded
        # Bid related asks
        bid_price = self.actions[0]
        bid_quantity = self.actions[1]
        # Solar related asks
        solar_ask_price = self.actions[2]
        solar_ask_quantity = self.actions[3]
        #Bess related asks
        bess_ask_price = self.actions[4]
        bees_ask_quantity = self.actions[5]



        # if generation:
        #
        #     actions ={
        #         "asks":{next_settle:{
        #             'quantity':quantity,
        #             'price': user_actions
        #         }
        #         }
        #     }

        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', load),
                self.metrics.track('next_settle_generation', generation))

            await self.metrics.save(10000)
        return actions

    async def step(self):
        # actions must come in the following format:
        # actions = {
        #     'bess': {
        #         time_interval: scheduled_qty
        #     },
        #     'bids': {
        #         time_interval: {
        #             'quantity': qty,
        #             'price': dollar_per_kWh
        #         }
        #     },
        #     'asks' {  
        #         source: {
        #             time_interval: {
        #                 'quantity': qty,
        #                 'price': dollar_per_kWh?
        #             }
        #         }
        #     }
        #
        next_actions = await self.act()
        return next_actions

    async def reset(self, **kwargs):
        return True

    async def decode_actions(self, action_indices: dict, ts_act):
        """
        TODO: November 30, 2022: this method will be used to decode the actions that are received from epymarl.
        #one price, one quantity for now
        if quantity > 0:
            we ask --> we only need
            bid beomes quantiity = 0, price = 0
        else
            we bid
            ask becomes quantity = 0, price = 0
        """

        ts_act_generation, ts_act_load = await self.__participant['read_profile'](ts_act)
        if "price" in self.action_type:
            if self.action_type['price'] == 'learned':
                price = self.actions[1] #ToDo: find right index
            else:
                price = self.price_heuristic(generation=ts_act_generation,
                                             load=ts_act_load,
                                             )
            price = round(price, 4) #ToDo: do we want to do this here or somewhere else? is this the right number?

        if "quantity" in self.action_type:
            if self.action_type['quantity'] == 'learned':
                quantity = self.actions[1] #ToDo: find right index
            else:
                quantity = self.quantity_heuristic(generation=ts_act_generation,
                                             load=ts_act_load,
                                             )
            quantity = int(quantity) #ToDo: do we want to do this here or somewhere else?

        if "storage" in self.action_type:
            if self.action_type['storage'] == 'learned':
                storage = self.actions[1] #ToDo: find right index
            else:
                raise NotImplementedError
            storage = int(storage) #ToDo: do we want to do this here or somewhere else?

        actions = dict()
        # print(action_indices)
        # price = self.actions['price'][action_indices['price']]
        # quantity = self.actions['quantity'][action_indices['quantity']]

        if quantity > 0:
            actions['bids'] = {
                str(ts_act): {
                    'quantity': quantity,
                    'price': price
                }
            }
        elif quantity < 0:
            actions['asks'] = {
                'solar': {
                    str(ts_act): {
                        'quantity': -quantity,
                        'price': price
                    }
                }
            }

        if 'storage' in self.actions:
            target = self.actions['storage'][action_indices['storage']]
            if target:
                actions['bess'] = {
                    str(ts_act): target
                }
        # print(actions)

        #log actions for later histogram plot
        for action in self.actions:
            self.episode_actions[action].append(self.actions[action][action_indices[action]])
        return actions

    async def check_read_flag(self, shared_list):
        """
        This method checks the read flag in a shared list.
        Parameters:
            Shared_list -> shared list object to check, assumes that element 0 is the flag and that flag can be
                            intepreted as boolean
            returns ->  Boolean
        """
        if shared_list[0]:
            return True
        else:
            return False

    async def read_action_values(self):
        """
        This method checks the action buffer flag and if the read flag is set, it reads the value in the buffer and stores
        them in self.actions

        """
        self.actions = []
        # check the action flag
        while True:
            flag = await self.check_read_flag(self.shared_list_action)
            print("Flag", flag)
            if flag:
                #read the buffer
                for e, item in enumerate(self.shared_list_action):
                    print(e, item)
                    self.actions.append(item)
                # self.actions = self.shared_list_action[1:]
                print('actions', self.actions)
                #reset the flag
                await self.write_flag(self.shared_list_action, False)
                break

    async def write_flag(self, shared_list, flag):
        """
        This method sets the flag
        Parameters:
            shared_list ->  shared list object to be modified
            flag -> boolean that indicates write 0 or 1. True sets 1
        """
        print(shared_list)

        if flag:
            shared_list[0] = 1
            print("Flag was set ")
        else:
            shared_list[0] = 0
            print("Flag was not set")

    async def obs_to_shared_memory(self, obs):
        """
        This method writes the values in the observations array to the observation buffer and then sets the flag for
        EPYMARL to read the values.

        """

        # obs will be an array
        # pack the values of the obs array into the shares list
        for e, item in enumerate(obs):
            print(e, item)
            self.shared_list_observation[e+1] = item

        #set the observation flat to written
        await self.write_flag(self.shared_list_observation,True)

    async def r_to_shared_memory(self, reward):
        """
        This method writes the reward value into the rewards array and then sets the flag for EPYMARL to read the
        values.
        """
        self.shared_list_reward[-1] = reward
        await self.write_flag(self.shared_list_reward, True)




