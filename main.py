import datetime
import settings

from pathlib import Path
from pyboy.pyboy import *
from gym.wrappers import FrameStack, NormalizeObservation
from AISettings.AISettingsInterface import AISettingsInterface
from AISettings.MegamanAISettings import MegamanAI
from AISettings.Megaman_2_AISettings import MegamanAI as MegamanAI_2
from MetricLogger import MetricLogger
from agent import AIPlayer
from functions import alphanum_key
from wrappers import SkipFrame, ResizeObservation
import sys
from CustomPyBoyGym import CustomPyBoyGym

"""
  Variables
"""
episodes = 40000
observation_types = ["raw", "tiles", "compressed", "minimal"]
observation_type = observation_types[settings.OBSERVATION_TYPE]
action_types = ["press", "toggle", "all"]
action_type = action_types[settings.ACTION_TYPE]
gameDimensions = (20, 16)
frameStack = 4
train = False
playtest = False

"""
  Choose game
"""
gamesFolder = Path("games")
games = [os.path.join(gamesFolder, f) for f in os.listdir(gamesFolder) if
         (os.path.isfile(os.path.join(gamesFolder, f)) and (f.endswith(".gbc") or f.endswith(".gb")))]
gameNames = [f.replace(".gbc", "").replace(".gb", "") for f in os.listdir(gamesFolder) if
             (os.path.isfile(os.path.join(gamesFolder, f)) and (f.endswith(".gbc") or f.endswith(".gb")))]

print("Avaliable games: ", games)
for cnt, gameName in enumerate(games, 1):
    sys.stdout.write("[%d] %s\n\r" % (cnt, gameName))

choice = int(input("Select game[1-%s]: " % cnt)) - 1
game = games[choice]
gameName = gameNames[choice]

"""
  Choose AI mode
"""
modes = ["Evaluate (HEADLESS)", "Evaluate (UI)",
         "Train (HEADLESS)", "Train (UI)", "Playtest (UI)"]
for cnt, modeName in enumerate(modes, 1):
    sys.stdout.write("[%d] %s\n\r" % (cnt, modeName))

mode = int(input("Select mode[1-%s]: " % cnt)) - 1


def update_ui_settings(option, value):
    with open('settings.py', 'r') as file:
        lines = file.readlines()

    with open('settings.py', 'w') as file:
        for line in lines:
            if line.startswith(f'{option} ='):
                file.write(f'{option} = {value}\n')
            else:
                file.write(line)
        file.close()
    file.close()


ui = False
if mode == 0:
    train = False
elif mode == 1:
    ui = True
    train = False
elif mode == 2:
    train = True
elif mode == 3:
    ui = True
    train = True
elif mode == 4:
    ui = True
    playtest = True

"""
  Logger
"""
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir = Path("checkpoints") / gameName / now
save_dir_eval = Path("checkpoints") / gameName / (now + "-eval")
save_dir_boss = Path("checkpoints") / gameName / (now + "-boss")
checkpoint_dir = Path("checkpoints") / gameName

"""
  Load emulator
"""
pyboy = PyBoy(game)

"""
  Load enviroment
"""

aiSettings = AISettingsInterface()
if "Xtreme" in gameName:
    if '_2' in gameName:
        aiSettings = MegamanAI_2()
        update_ui_settings('GAME_VERSION', 2)
    else:
        aiSettings = MegamanAI()
        update_ui_settings('GAME_VERSION', 1)

env = CustomPyBoyGym(pyboy, ui=ui)
env.setAISettings(aiSettings)  # use this settings
filteredActions = aiSettings.GetActions()  # get possible actions
print("Possible actions: ", [[WindowEvent(i).__str__() for i in x] for x in filteredActions])

"""
  Load wrappers
"""
env = SkipFrame(env, skip=4)
env = ResizeObservation(env, gameDimensions)  # transform MultiDiscreate to Box for framestack
env = NormalizeObservation(env)  # normalize the values
env = FrameStack(env, num_stack=frameStack)

"""
  Load AI
"""
aiPlayer = AIPlayer((frameStack,) + gameDimensions, len(filteredActions), save_dir, now,
                    aiSettings.GetHyperParameters())
bossAiPlayer = AIPlayer((frameStack,) + gameDimensions, len(filteredActions), save_dir_boss, now,
                        aiSettings.GetBossHyperParameters())

if mode < 2:  # Evaluation mode
    # Load model
    folderList = [name for name in os.listdir(checkpoint_dir) if
                  os.path.isdir(checkpoint_dir / name) and len(os.listdir(checkpoint_dir / name)) != 0]

    if len(folderList) == 0:
        print("No models to load in path: ", save_dir)
        quit()

    for cnt, fileName in enumerate(folderList, 1):
        sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))

    choice = int(input("Select folder with platformer model[1-%s]: " % cnt)) - 1
    folder = folderList[choice]
    print(folder)

    fileList = [f for f in os.listdir(checkpoint_dir / folder) if f.endswith(".chkpt")]
    fileList.sort(key=alphanum_key)
    if len(fileList) == 0:
        print("No models to load in path: ", folder)
        quit()

    modelPath = checkpoint_dir / folder / fileList[-1]
    aiPlayer.loadModel(modelPath)

    choice = int(
        input("Select folder with boss model[1-%s] (if not using boss model select same as previous): " % cnt)) - 1
    folder = folderList[choice]
    print(folder)

    fileList = [f for f in os.listdir(checkpoint_dir / folder) if f.endswith(".chkpt")]
    fileList.sort(key=alphanum_key)
    if len(fileList) == 0:
        print("No models to load in path: ", folder)
        quit()

    bossModelPath = checkpoint_dir / folder / fileList[-1]
    bossAiPlayer.loadModel(bossModelPath)

"""
  Main loop
"""

if train:  # Training modes
    pyboy.set_emulation_speed(0)
    save_dir.mkdir(parents=True)
    save_dir_boss.mkdir(parents=True)
    logger = MetricLogger(save_dir_boss)
    aiPlayer.saveHyperParameters()
    bossAiPlayer.saveHyperParameters()

    print("Training mode")
    print("Total Episodes: ", episodes)
    aiPlayer.net.train()
    bossAiPlayer.net.train()

    player = aiPlayer
    for e in range(episodes):
        observation = env.reset()
        start = time.time()

        if "Xtreme" in gameName:
            if '_2' in gameName:
                with open("./games/MegaManXtreme_2.gbc.state", "rb") as f:
                    pyboy.load_state(f)
            else:
                with open("./games/MegaManXtreme.gbc.state", "rb") as f:
                    pyboy.load_state(f)

        while True:
            if aiSettings.IsBossActive(pyboy):
                player = bossAiPlayer
            else:
                player = aiPlayer
            # Act depending on current state
            actionId = player.act(observation)
            actions = filteredActions[actionId]
            # Perform action, and update a frame
            next_observation, reward, done, truncated, info = env.step(actions)

            # Add to memory
            player.cache(observation, next_observation, actionId, reward, done)
            # Learn
            q, loss = player.learn()
            # Log
            logger.log_step(reward, loss, q, player.scheduler.get_last_lr())
            # Update game state
            observation = next_observation

            if done or time.time() - start > 500:
                break

        logger.log_episode()
        logger.record(episode=e, epsilon=player.exploration_rate, stepsThisEpisode=player.curr_step,
                      maxLength=aiSettings.GetLength(pyboy))

    aiPlayer.save()
    bossAiPlayer.save()
    env.close()
elif not train and not playtest:
    print("Evaluation mode")
    pyboy.set_emulation_speed(1)

    save_dir_eval.mkdir(parents=True)
    logger = MetricLogger(save_dir_eval)

    aiPlayer.exploration_rate = 0
    aiPlayer.net.eval()

    bossAiPlayer.exploration_rate = 0
    bossAiPlayer.net.eval()

    player = aiPlayer
    for e in range(episodes):
        observation = env.reset()
        while True:
            if aiSettings.IsBossActive(pyboy):
                player = bossAiPlayer
            else:
                player = aiPlayer
            actionId = player.act(observation)
            action = filteredActions[actionId]
            next_observation, reward, done, truncated, info = env.step(action)

            logger.log_step(reward, 1, 1, 1)

            print("Reward: ", reward)
            print("Action: ", [WindowEvent(i).__str__() for i in action])
            aiSettings.PrintGameState(pyboy)

            observation = next_observation

            # print(reward)
            if done:
                break

        logger.log_episode()
        logger.record(episode=e, epsilon=player.exploration_rate, stepsThisEpisode=player.curr_step,
                      maxLength=aiSettings.GetLength(pyboy))
    env.close()

elif playtest:
    pyboy.set_emulation_speed(1)
    env.reset()
    print("Playtest mode")
    while True:
        previousGameState = aiSettings.GetGameState(pyboy)
        env.pyboy.tick()
        if settings.DEBUG:
            print("Reward: ", aiSettings.GetReward(previousGameState, pyboy))
            print("Real max length: ", aiSettings.GetLength(pyboy))
        aiSettings.PrintGameState(pyboy)
