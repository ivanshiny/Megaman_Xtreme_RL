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
