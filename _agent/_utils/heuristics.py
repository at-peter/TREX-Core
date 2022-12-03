"includes some useful default behaviors for TREX Agents to load."

# heuristics args:
# load: float
# gen: float

def constant_value(**kwargs):
    #ToDo: make sure we have a reasonablle default etc
    return 0.11

def as_netload(**kwargs):
    load = kwargs['load']
    generation = kwargs['generation']
    net_load = load - generation
    return net_load

def relative_to_max(**kwargs):
    raise NotImplementedError

class PriceHeuristics:
    def __init__(self, **kwargs):
        type = kwargs['type']

        if type == 'constant':
            self.get_value = constant_value
        elif type == 'net_load':
            self.get_value = as_netload
        else:
            print('could not find associated heuristic', type)
            raise NotImplementedError

class QuantityHeuristics:
    def __init__(self, **kwargs):
        type = kwargs['type']

        if type == 'net-load':
            self.get_quantity = as_netload
        else:
            print('could not find associated heuristic', type)
            raise NotImplementedError


