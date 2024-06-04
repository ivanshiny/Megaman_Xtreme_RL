from collections import deque
import random
from settings import DEBUG
import numpy as np
import torch
from AISettings.AISettingsInterface import Config
from model import DDQN

class AIPlayer:
    def __init__(self, state_dim, action_space_dim, save_dir, date, config: Config):
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.save_dir = save_dir
        self.date = date
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.net = DDQN(self.state_dim, self.action_space_dim).to(device=self.device)

        self.config = config

        self.exploration_rate = self.config.exploration_rate
        self.exploration_rate_decay = self.config.exploration_rate_decay
        self.exploration_rate_min = self.config.exploration_rate_min
        self.curr_step = 0

        """
            Memory
        """
        self.memory = deque(maxlen=self.config.deque_size)
        self.batch_size = self.config.batch_size
        self.save_every = self.config.save_every  # no. of experiences between saving Mario Net

        """
            Q learning
        """
        self.gamma = self.config.gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.learning_rate_decay)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = self.config.burnin  # min. experiences before training
        self.learn_every = self.config.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = self.config.sync_every  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
            Given a state, choose an epsilon-greedy action and update value of step.

            Inputs:
            state(LazyFrame): A single observation of the current state, dimension is (state_dim)
            Outputs:
            action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if random.random() < self.exploration_rate:
            action_idx = random.randint(0, self.action_space_dim - 1)
        # EXPLOIT
        else:
            try:
                state = np.array(state)
            except ValueError:
                state = np.array(state[0])
            state = torch.tensor(state).float().to(device=self.device)
            state = state.unsqueeze(0)

            neural_net_output = self.net(state, model="online")
            action_idx = torch.argmax(neural_net_output).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs: (
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = np.array(state[0])
        next_state = np.array(next_state)

        state = torch.tensor(state).float().to(device=self.device)
        next_state = torch.tensor(next_state).float().to(device=self.device)
        action = torch.tensor([action]).to(device=self.device)
        reward = torch.tensor([reward]).to(device=self.device)
        done = torch.tensor([done]).to(device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        processed_batch = []

        for entry in batch:
            state, next_state, action, reward, done = entry

            # Asegurarse de que state tenga la forma correcta
            if state.dim() == 2:
                state = state.unsqueeze(0).unsqueeze(0)  # Agregar dimensiones de batch y canal si faltan
            elif state.dim() == 3:
                state = state.unsqueeze(0)  # Agregar una dimensión de batch si falta

            if state.shape[1] == 1:
                state = state.repeat(1, 4, 1, 1)  # Asegurar que haya 4 canales
            elif state.shape[1] == 16:
                state = state[:, :4, :, :]  # Reducir canales a los primeros 4
            elif state.shape[1] != 4:
                print(next_state)
                raise ValueError(f"Expected state to have 4 channels, but got {state.shape[1]} channels.")

            # Asegurarse de que next_state tenga la forma correcta
            if next_state.dim() == 2:
                next_state = next_state.unsqueeze(0).unsqueeze(0)  # Agregar dimensiones de batch y canal si faltan
            elif next_state.dim() == 3:
                next_state = next_state.unsqueeze(0)  # Agregar una dimensión de batch si falta

            if next_state.shape[1] == 1:
                next_state = next_state.repeat(1, 4, 1, 1)  # Asegurar que haya 4 canales
            elif next_state.shape[1] == 16:
                next_state = next_state[:, :4, :, :]  # Reducir canales a los primeros 4
            elif next_state.shape[1] != 4:
                print(next_state)
                raise ValueError(f"Expected next_state to have 4 channels, but got {next_state.shape[1]} channels.")

            processed_batch.append((state, next_state, action, reward, done))

        state, next_state, action, reward, done = map(torch.stack, zip(*processed_batch))

        return state, next_state, action, reward, done

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory get self.batch_size number of memories
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate, make predictions for the each memory
        td_est = self.td_estimate(state, action)

        # Get TD Target make predictions for next state of each memory
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()  #

        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def td_estimate(self, state, action):
        model_output = self.net(state, model="online")
        if DEBUG:
            print(f'model_output shape: {model_output.shape}')
            print(f'batch_size: {self.batch_size}')
            print(f'action: {action}')

        # Asegurándonos de que action sea un tensor
        if isinstance(action, int):
            action = torch.tensor([action])
        elif isinstance(action, list):
            action = torch.tensor(action)

        # Ajustar batch_size según la salida real
        batch_size = model_output.shape[0]
        if batch_size != self.batch_size:
            if DEBUG:
                print(f'Ajustando batch_size de {self.batch_size} a {batch_size}')
            self.batch_size = batch_size

        current_q = model_output[
            np.arange(0, self.batch_size), action.clamp(0, model_output.shape[1] - 1)]  # Q_online(s,a)
        return current_q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        with torch.no_grad():
            next_q = self.net(next_state, model="target")
            if DEBUG:
                print(f'next_q shape: {next_q.shape}')

            # Verificación de índices válidos
            next_action = next_q.argmax(1)
            if DEBUG:
                print(f'next_action: {next_action}')

            # Asegurándonos de que next_action esté dentro del rango válido
            next_action = next_action.clamp(0, next_q.shape[1] - 1)
            if DEBUG:
                print(f'next_action after clamp: {next_action}')

            max_next_q = next_q[np.arange(0, next_q.shape[0]), next_action]
            return reward + (~done).float() * self.gamma * max_next_q

    def td_estimate(self, state, action):
        # Asegúrate de que el estado tenga la forma correcta
        if state.dim() == 3:
            state = state.unsqueeze(1)  # Agregar una dimensión de canales si falta
            state = state.repeat(1, 4, 1, 1)  # Asegurar que haya 4 canales

        model_output = self.net(state, model="online")
        if DEBUG:
            print(f'model_output shape: {model_output.shape}')
            print(f'batch_size: {self.batch_size}')
            print(f'action: {action}')

        # Asegurándonos de que action sea un tensor
        if isinstance(action, int):
            action = torch.tensor([action])
        elif isinstance(action, list):
            action = torch.tensor(action)

        # Ajustar batch_size según la salida real
        batch_size = model_output.shape[0]
        if batch_size != self.batch_size:
            if DEBUG:
                print(f'Ajustando batch_size de {self.batch_size} a {batch_size}')
            self.batch_size = batch_size

        current_q = model_output[
            np.arange(0, self.batch_size), action.clamp(0, model_output.shape[1] - 1)]  # Q_online(s,a)
        return current_q

    def loadModel(self, path):
        dt = torch.load(path, map_location=torch.device(self.device))
        self.net.load_state_dict(dt["model"])
        self.exploration_rate = dt["exploration_rate"]
        print(f"Loading model at {path} with exploration rate {self.exploration_rate}")

    def saveHyperParameters(self):
        save_hyper_parameters = self.save_dir / "hyperparameters"
        with open(save_hyper_parameters, "w") as f:
            f.write(f"exploration_rate = {self.config.exploration_rate}\n")
            f.write(f"exploration_rate_decay = {self.config.exploration_rate_decay}\n")
            f.write(f"exploration_rate_min = {self.config.exploration_rate_min}\n")
            f.write(f"deque_size = {self.config.deque_size}\n")
            f.write(f"batch_size = {self.config.batch_size}\n")
            f.write(f"gamma (discount parameter) = {self.config.gamma}\n")
            f.write(f"learning_rate = {self.config.learning_rate}\n")
            f.write(f"learning_rate_decay = {self.config.learning_rate_decay}\n")
            f.write(f"burnin = {self.config.burnin}\n")
            f.write(f"learn_every = {self.config.learn_every}\n")
            f.write(f"sync_every = {self.config.sync_every}")

    def save(self):
        """
            Save the state to directory
        """
        save_path = (self.save_dir / f"mmx_net_0{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"XtremeNet saved to {save_path} at step {self.curr_step}")
