from amalearn.reward import RewardBase
import numpy as np

class Q3Reward2(RewardBase):
    def __init__(self, metro_mean, metro_std, taxi_mean, taxi_std, corona_cost):
        super(Q3Reward2, self).__init__()
        self.metro_mean = metro_mean
        self.metro_std = metro_std
        self.taxi_mean = taxi_mean
        self.taxi_std = taxi_std
        self.corona_cost = corona_cost

    def calculate_metro_cost(self):
        metro_population = np.random.normal(loc=self.metro_mean, scale=self.metro_std)
        corona_prob = metro_population/10 * 2 * np.pow(10, -5)
        return corona_prob * self.corona_cost

    def calculate_taxi_cost(self):
        return np.random.normal(loc=self.taxi_mean, scale=self.taxi_std)

    def get_reward(self):
        total_metro_cost = self.calculate_metro_cost()
        taxi_cost = self.calculate_taxi_cost()
        return total_metro_cost + taxi_cost
