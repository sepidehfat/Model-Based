from amalearn.reward import RewardBase
import numpy as np

class BernoulliReward(RewardBase):
    def __init__(self, bandit_prob, reward, punishment):
        super(RewardBase, self).__init__()
        self.bandit_prob = bandit_prob
        self.reward = reward
        self.punishment = punishment

    def get_reward(self):
        r = np.random.binomial(1, self.bandit_prob)
        return self.reward if r == 1 else self.punishment