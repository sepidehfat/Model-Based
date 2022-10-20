import gym
from amalearn.environment import EnvironmentBase


class Q2MutliArmedBanditEnvironment(EnvironmentBase):
    def __init__(self, rewards, episode_max_length, id, container=None):
        state_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(len(rewards))

        super(Q2MutliArmedBanditEnvironment, self).__init__(action_space, state_space, id, container)
        self.arms_rewards = rewards
        self.episode_max_length = episode_max_length
        self.state = {
            'length': 0,
            'last_action': None
        }
        self.sigma = 0
        # self.volume_base = None
        # self.current_reward = None

    def get_info(self, action):
        return self.sigma

    def calculate_reward(self, action):
        reward , sigma = self.arms_rewards[action].get_reward()
        self.sigma = sigma
        return reward

    def terminated(self):
        return self.state['length'] >= self.episode_max_length

    def observe(self):
        return {}

    def available_actions(self):
        return self.action_space.n

    def next_state(self, action):
        self.state['length'] += 1
        self.state['last_action'] = action

    def reset(self):
        self.state['length'] = 0
        self.state['last_action'] = None

    def render(self, mode='human'):
        pass
        # print('{}:\taction={}'.format(self.state['length'], self.state['last_action']))

    def close(self):
        return
