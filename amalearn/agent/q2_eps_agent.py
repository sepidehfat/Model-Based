import numpy as np
from amalearn.agent import AgentBase

class Q2EpsAgent(AgentBase):
    def __init__(self, id, environment, epsilon, learning_rate, status, alpha, beta, lambdaa):
        super(Q2EpsAgent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.status = status
        self.q_values = np.zeros(self.environment.available_actions())
        self.rewards = np.zeros(self.environment.available_actions())
        self.counts = np.zeros(self.environment.available_actions())
        self.available_actions = self.environment.available_actions()
        self.alpha = alpha
        self.beta = beta
        self.lambdaa = lambdaa
        self.sigma = 0

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

    def update(self, action, r, inf):
        self.sigma = inf
        r = self.utility(r)
        self.rewards[action] += r
        self.counts[action] += 1
        self.q_values[action] = self.rewards[action] / self.counts[action]
        if self.status == "adaptive":
            self.epsilon = self.epsilon/2

    def utility(self, r):
        if self.sigma == np.inf:
            utility = r
        else:
            utility = r * self.sigma * 0.0005
        return utility

    def take_action(self) -> (object, float, bool, object):
        action = self.select_action()
        obs, r, d, inf = self.environment.step(action)
        print(obs, r, d, inf)
        self.update(action, r, inf)
        self.environment.render()
        return obs, r, d, inf