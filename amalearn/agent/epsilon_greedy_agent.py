import numpy as np
from amalearn.agent import AgentBase


class EpsilonGreedyAgent(AgentBase):
    def __init__(self, id, environment, epsilon, status, alpha, beta, lambdaa):
        super(EpsilonGreedyAgent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.status = status
        self.q_values = np.zeros(self.environment.available_actions())
        self.rewards = np.zeros(self.environment.available_actions())
        self.counts = np.zeros(self.environment.available_actions())
        self.available_actions = self.environment.available_actions()
        self.alpha = alpha
        self.beta = beta
        self.lambdaa = lambdaa

    def select_action(self):
        available_actions = self.available_actions
        eps = self.epsilon

        best_action = np.argmax(self.q_values)
        random_action = np.random.choice(available_actions)

        prob_best_action = 1 - eps + (eps / available_actions)
        prob_random_action = eps / available_actions
        sum = prob_best_action + prob_random_action #normalize
        selected_action = np.random.choice([best_action, random_action], p=[prob_best_action/sum, prob_random_action/sum])
        return selected_action

    def update(self, action, r):
        r = self.utility(r)
        self.rewards[action] += r
        self.counts[action] += 1
        self.q_values[action] = self.rewards[action] / self.counts[action]
        if self.status == "adaptive":
            self.epsilon = self.epsilon/2

    def utility(self, r):
        if r>=0:
            return r**self.alpha
        else:
            return -self.lambdaa * abs(r)**self.beta

    def take_action(self) -> (object, float, bool, object):
        action = self.select_action()
        obs, r, d, i = self.environment.step(action)
        # print(obs, r, d, i)
        self.update(action, r)
        self.environment.render()
        return obs, r, d, i