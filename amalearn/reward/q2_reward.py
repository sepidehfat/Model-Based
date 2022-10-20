from amalearn.reward import RewardBase
import numpy as np

class Q2Reward(RewardBase):
    def __init__(self, mean, std, volume, price):
        super(Q2Reward, self).__init__()
        self.mean = mean
        self.std = std
        self.volume = volume
        self.price = price
        self.arg_selected_choice = None

    def calculate_arg_selected_choice(self):
        x = np.random.beta(2, 5, 1)[0]
        ratio = [x*p/v for p, v in zip(self.price, self.volume)]
        self.arg_selected_choice = np.argmin(ratio)

    def calculate_remained_price(self):
        i = self.arg_selected_choice
        used_volume = np.random.normal(loc=self.mean, scale=self.std)
        remained_volume = self.volume[i] - used_volume
        remained_price = remained_volume * (self.price[i]/self.volume[i])
        return remained_price

    def calculate_sigma(self, reward1, reward2, reward):
        i = self.arg_selected_choice
        if reward == reward1:
            if i == 0:
                sigma = self.price[i] - 3000
            else:
                sigma = self.price[i] - 1600
        if reward == reward2:
            sigma = np.inf
        return sigma

    def get_reward(self):
        self.calculate_arg_selected_choice()
        i = self.arg_selected_choice
        remained_price = self.calculate_remained_price()
        reward1 = remained_price + self.price[i]
        reward2 = -10
        reward = np.random.choice([reward1, reward2], p=[0.7, 0.3])
        sigma = self.calculate_sigma(reward1, reward2, reward)
        return reward, sigma
