import numpy as np
from amalearn.agent import AgentBase
class UCBAgent(AgentBase):
    def __init__(self, id, environment, c, alpha, beta, lambdaa):
        super(UCBAgent, self).__init__(id, environment)
        self.c = c
        self.q_values = np.zeros(self.environment.available_actions())
        self.rewards = np.zeros(self.environment.available_actions())
        self.counts = np.zeros(self.environment.available_actions())
        self.available_actions = self.environment.available_actions()
        self.alpha = alpha
        self.beta = beta
        self.lambdaa = lambdaa

    def calculate_ucb(self, action):
        if self.counts[action] == 0:
            return np.inf
        else:
            return self.q_value[action] + self.c * np.sqrt((np.log(sum(self.counts))) / self.counts[action])

    def select_action(self):
        available_actions = np.arange(1, self.available_actions)
        ucb = [self.calculate_ucb(action) for action in available_actions]
        return np.argmax(ucb)

    def update(self, action, r):
        # self.current_reward = r
        self.rewards[action] += r
        self.counts[action] += 1
        self.q_values[action] = self.rewards[action] / self.counts[action]

    def take_action(self) -> (object, float, bool, object):
        action = self.select_action()
        obs, r, d, i = self.environment.step(action)
        # print(obs, r, d, i)
        self.update(action, r)
        self.environment.render()
        return obs, r, d, i