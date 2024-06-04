import torch
import torch.nn as nn
import copy

from settings import DEBUG


class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.previous_online = None
        c, h, w = input_dim

        if DEBUG:
            print('Instanciando la red')
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        # Calcular el tamaño de la salida después de las convoluciones
        self.dummy_input = torch.zeros(1, 4, h, w)
        self.dummy_output = self._forward_conv(self.dummy_input)
        dummy_flatten_dim = self.dummy_output.numel()
        if DEBUG:
            print(f'Tamaño de la salida después de convoluciones y aplanado: {dummy_flatten_dim}')

        self.linear1 = nn.Linear(dummy_flatten_dim, 512)
        self.linear2 = nn.Linear(512, output_dim)

        self.online = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            self.conv1x1,
            self.relu,
            self.flatten,
            self.linear1,
            self.relu,
            self.linear2
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.flatten(x)
        return x

    def forward(self, input, model):
        if DEBUG:
            print(f'Input shape inicial: {input.shape}')

        # Asegurar que el tensor tenga 4 dimensiones
        if input.dim() == 5:
            # Suponiendo que la segunda dimensión es la longitud de la secuencia
            input = input.view(-1, *input.shape[2:])
            if DEBUG:
                print(f'Input reshaped from 5D to 4D: {input.shape}')
        if input.dim() == 3:
            input = input.unsqueeze(1)  # Agregar una dimensión de canales si falta
            if DEBUG:
                print(f'Input reshaped to 4D: {input.shape}')

        # Asegurar que el tensor tenga 4 canales
        if input.shape[1] == 1:
            input = input.repeat(1, 4, 1, 1)
            if DEBUG:
                print(f'Input repeated to 4 channels: {input.shape}')
        elif input.shape[1] == 16:
            input = self.conv1x1(input)
            if DEBUG:
                print(f'Input reshaped from 16 to 4 channels: {input.shape}')
        elif input.shape[1] != 4:
            raise ValueError(f"Expected input to have 4 channels, but got {input.shape[1]} channels.")

        try:
            if model == "online":
                self.previous_online = self.online(input)
                if DEBUG:
                    print(f'Output shape after online network: {self.previous_online.shape}')
                return self.previous_online
            elif model == "target":
                self.previous_online = self.target(input)
                if DEBUG:
                    print(f'Output shape after target network: {self.previous_online.shape}')
                return self.previous_online
        except Exception:
            return self.previous_online
