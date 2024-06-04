import gymnasium as gym
import numpy as np
from gymnasium import spaces

import settings
from AISettings.AISettingsInterface import AISettingsInterface
from AISettings.MegamanAISettings import GameState
from AISettings.Megaman_2_AISettings import GameState as GameState2
from settings import GAME_VERSION

possible_actions = ['', 'a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']

matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)


class CustomPyBoyGym(gym.Env):
    stop = False

    def __init__(self, pyboy, debug=False, ui=False):
        super().__init__()
        self.ui = ui
        self.pyboy = pyboy
        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug
        self.previousGameState = None

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(possible_actions))
        self.observation_space = game_area_observation_space

        self.pyboy.game_wrapper.start_game()

    def step(self, actions):
        # Move the agent
        self.previousGameState = self.aiSettings.GetGameState(self.pyboy)

        if actions == 0:
            pass
        else:
            for action in actions:
                self.pyboy.button(possible_actions[action])

        if self.ui:
            self.pyboy.tick(1)
        else:
            self.pyboy.tick(1, False)

        if GAME_VERSION == 1:
            done = GameState.done
            if done:
                GameState.done = False
        elif GAME_VERSION == 2:
            done = GameState2.done
            if done:
                GameState2.done = False

        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness

        observation = self.pyboy.game_area()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness
        self._fitness = self._fitness + self.aiSettings.GetReward(self.previousGameState, self.pyboy)

    def setAISettings(self, aisettings: AISettingsInterface):
        self.aiSettings = aisettings

    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0

        observation = self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()
