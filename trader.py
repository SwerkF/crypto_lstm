import numpy as np
import gymnasium as gym
from gymnasium import spaces

"""
Environnement pour DQN simulation de trader crypto.
"""
class TraderEnv(gym.Env):
    metadata  = {'render.modes': ['human', 'rgb_array'], "render_fps": 4}

    def __init__(self, render_mode=None, max_steps=100, step_size=0.05, target=0.7):
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.step_size = float(step_size)
        self.target = float(target)

        # Observation space: [current_price, owned_assets, cash_balance]
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_price = self.np_random.uniform(0.4, 0.6)
        self.owned_assets = 0.0
        self.cash_balance = 1.0
        self.current_step = 0

        observation = np.array([self.current_price, self.owned_assets, self.cash_balance], dtype=np.float32)

        info = {}
        return observation, info

    def step(self, action):
        assert self.action_space.contains((action), "Action invalide")

        prev_value = self.owned_assets * self.current_price + self.cash_balance

        if action == 1:  # Buy
            self.owned_assets += self.step_size
            self.cash_balance -= self.step_size * self.current_price
        elif action == 2:  # Sell
            self.owned_assets -= self.step_size
            self.cash_balance += self.step_size * self.current_price

        self.current_price = self.np_random.uniform(0.4, 0.6)
        self.current_step += 1

        observation = np.array([self.current_price, self.owned_assets, self.cash_balance], dtype=np.float32)

        if self.render_mode == 'human':
            self._render_frame()

        done = self.current_step >= self.max_steps
        reward = self.owned_assets * self.current_price + self.cash_balance - prev_value

        info = {}
        return observation, reward, done, info