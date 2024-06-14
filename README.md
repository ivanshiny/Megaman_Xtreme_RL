# Introducción

Este es un proyecto de final de grado (TFG), cuyo objetivo es entrenar un modelo para ser capaz de aprender a jugar a Megaman Xtreme 1 y 2.

# Requisitos

  - Python 3.10 o superior
  - Pip 24.0 o superior:
    - Instalar desde Windows:
      - __`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`__
      - __`python get-pip.py`__
    - Actualizar desde Windows:  __`python -m pip install --upgrade pip`__
    - Actualizar desde Ubuntu:  __`pip install --upgrade pip`__
  - Pyboy 2.0 o superior  __`pip install pyboy`__
  - Roms con Megaman Xtreme 1 o 2. Estas deben de contener "Xtreme" en su nombre, y Megaman Xtreme 2 debe de contener "_2".
  -	Si se quiere recargar la partida desde un punto, se puede crear un estado desde Pyboy abriendo directamente el juego con Pyboy o desde la opción playtesting en el programa, y pulsando la tecla Z en el punto desde el que se quiera cargar.

# Instalación

- Desde CMD:
  - Desde la carpeta contenedora del proyecto, instalar los requerimientos con __`pip3 install -r requirements.txt`__
  - Ejecutar __`python3 main.py`__

- Desde IDE:
  - Crear máquina virtual __`.venv`__
  - Instalar requerimientos
  - Correr desde main.py

- Desde Docker
  - Desde la carpeta contenedora del proyecto, usar el comando __`docker build --tag pyboy-rl .`__
  - Una vez en la imagen, ejecutar __`python3 main.py`__

# Referencias
Este proyecto está basado en [Pyboy-RL](https://github.com/lixado/PyBoy-RL)


# Introduction

This is a final degree project (TFG), aimed at training a model to be able to learn to play Megaman Xtreme 1 and 2.

# Requirements

  - Python 3.10 or higher
  - Pip 24.0 or higher:
    - Install on Windows:
      - curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
      - python get-pip.py
    - Update on Windows: python -m pip install --upgrade pip
    - Update on Ubuntu: pip install --upgrade pip
  - Pyboy 2.0 or higher pip install pyboy
  - Roms with Megaman Xtreme 1 or 2. These must contain "Xtreme" in their name, and Megaman Xtreme 2 must contain "_2".
  -   If you want to reload the game from a certain point, you can create a state from Pyboy by opening the game directly with Pyboy or from the playtesting option in the program, and pressing the Z key at the point from which you want to load.
    
# Installation

  - From CMD:
    - From the project's root folder, install the requirements with pip3 install -r requirements.txt
    - Run python3 main.py
    
  - From IDE:
    - Create a virtual machine .venv
    - Install requirements
    - Run from main.py

  - From Docker:
    - From the project's root folder, use the command docker build --tag pyboy-rl .
    - Once in the image, run python3 main.py

# References

This project is based on Pyboy-RL
