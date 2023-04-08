import numpy as np


class Competitor(object):

    def __init__(self, type):
        self.type = type

    def get_price(self, price_agent, price_min, price_max, price_mean, price_std):
        # Type 1: Competitor's price is always constant (price_mean)
        # Type 2: Competitor's price is randomly chosen from N(price_mean, price_std)
        # Type 3: Competitor's price is equal to the last agent's price (mimic agent strategy)
        if self.type == 'type_1':
            return price_mean
        elif self.type == 'type_2':
            return np.clip(int(np.random.normal(loc=price_mean, scale=price_std)), a_min=price_min, a_max=price_max)
        else:
            return price_agent