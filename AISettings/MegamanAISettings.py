import itertools

from pyboy.utils import WindowEvent

from AISettings.AISettingsInterface import AISettingsInterface
from AISettings.AISettingsInterface import Config


class GameState:
    max_x_reached = 0
    done = False

    live_lost = -10000
    lost_health = -1750
    gained_health = 2000
    damaged_boss = 2000
    killed_boss = 1000000
    moved_left = -5
    moved_right = 3

    def __init__(self, pyboy):
        # game_wrapper = pyboy.game_wrapper()
        self.boss_health = pyboy.memory[0xCAE1]
        self.screen_x_position = pyboy.memory[0xCF8D]
        self.megaman_x_position = pyboy.memory[0xCC5B] + (pyboy.memory[0xCC5C] * 255)
        self.megaman_y_position = pyboy.memory[0xCC5E]
        self.game_state = pyboy.memory[0xD02C]
        self.health = pyboy.memory[0xCADC]
        self.lives_left = pyboy.memory[0xCADD]
        self.max_x_position = 0


class MegamanAI(AISettingsInterface):
    def GetReward(self, previous_megaman: GameState, pyboy):
        current_megaman = GameState(pyboy)
        reward = 0

        # Killed boss
        if current_megaman.boss_health == 0 and previous_megaman.boss_health > 0:
            return GameState.killed_boss

        # Damaged boss
        if current_megaman.boss_health < previous_megaman.boss_health:
            reward += GameState.damaged_boss

        # Lost health
        if current_megaman.health < previous_megaman.health:
            reward += GameState.lost_health

        # Gained health
        if current_megaman.health > previous_megaman.health != 0:
            reward += GameState.gained_health

        # Lost a live
        if current_megaman.health == 0 and previous_megaman.health != 0:
            if current_megaman.lives_left > 0:
                return GameState.live_lost
            GameState.done = True
            GameState.max_x_reached = 0
            return GameState.live_lost * 2

        if not self.IsBossActive(pyboy):
            # Moving backwards
            if current_megaman.megaman_x_position < previous_megaman.megaman_x_position:
                reward += GameState.moved_left

            # Moving foward
            elif current_megaman.megaman_x_position > previous_megaman.megaman_x_position:
                reward += GameState.moved_right
                GameState.done = 0

                # Moving to new zones
                if current_megaman.megaman_x_position > GameState.max_x_reached:
                    GameState.max_x_reached = current_megaman.megaman_x_position
                    reward += (GameState.moved_right * 25)

        return reward

    def GetActions(self):
        baseActions = [WindowEvent.PRESS_BUTTON_A,
                       WindowEvent.PRESS_BUTTON_B,
                       WindowEvent.PRESS_ARROW_UP,
                       WindowEvent.PRESS_ARROW_DOWN,
                       WindowEvent.PRESS_ARROW_LEFT,
                       WindowEvent.PRESS_ARROW_RIGHT
                       ]

        totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
        withoutRepeats = []

        for combination in totalActionsWithRepeats:
            reversedCombination = combination[::-1]
            if reversedCombination not in withoutRepeats:
                withoutRepeats.append(combination)

        filteredActions = [[action] for action in baseActions] + withoutRepeats

        return filteredActions

    def PrintGameState(self, pyboy):
        pass

    def GetGameState(self, pyboy) -> GameState:
        return GameState(pyboy)

    def GetLength(self, pyboy):
        return self.GetGameState(pyboy).boss_health

    def IsBossActive(self, pyboy):
        if self.GetGameState(pyboy).boss_health > 0:
            return True
        return False

    def GetHyperParameters(self) -> Config:
        config = Config()
        config.exploration_rate_decay = 0.9999975
        config.exploration_rate_min = 0.01
        config.deque_size = 500000
        config.batch_size = 64
        config.save_every = 2e5
        config.learning_rate_decay = 0.9999985
        config.gamma = 0.8
        config.learning_rate = 0.0002
        config.burnin = 1000
        config.sync_every = 100
        return config

    def GetBossHyperParameters(self) -> Config:
        config = self.GetHyperParameters()
        config.exploration_rate_decay = 0.99999975
        return config
