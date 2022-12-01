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

def price_as_relative_to_max(**kwargs):
    raise NotImplementedError

class PriceHeuristics:
    def __init__(self, **kwargs):
        type = kwargs['type']

        if type == 'constant':
            self.get_value = constant_value
        elif type == 'net_load':
            self.get_value = as_netload
        else:
            print('oulld not find associated heuristicc', type)
            raise NotImplementedError


class QuantityHeuristics:
    def __init__(self, **kwargs):
        type = kwargs['type']

        if type == 'net_load':
            self.get_quantity = quantity_as_netload
        else:
            print('oulld not find assoiated prie behavior for ', type)
            raise NotImplementedError


