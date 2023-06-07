# Adaptive Issac Gym

## Intro

This example trains the Crazyflie drone model to hover near a fixed position. It is achieved by applying thrust forces to the four rotors.

<img src="https://user-images.githubusercontent.com/6352136/185715165-b430a0c7-948b-4dce-b3bb-7832be714c37.gif" width="300" height="150"/>

## Setup

Install [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html). For convenience, we recommend exporting the following environment variables to your ~/.bashrc or ~/.zshrc files:
```
# Isaac Sim root directory
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2022.2.0"
```

Isaac Sim provides a built-in Python 3.7 environment containing a PyTorch (1.13.0) installation. So there are different options to set up a development environment.
 
### Built-in Python environment

The built-in python interpreter launch script is at `${ISAACSIM_PATH}/python.sh`. So the straightforward way to use the built-in environment is
```
# to launch a script
${ISAACSIM_PATH}/python.sh path/to/script.py
# to install some package
${ISAACSIM_PATH}/python.sh -m pip install name_of_package_here
```
`python.sh` works by setting up the necessary environment variables including `PYTHONPATH` so that you can import the functionalities of Isaac SIm as modules in Python scripts. Check `${ISAACSIM_PATH}/setup_python_env.sh` for more detail.


### Module installation

Install `omniisaacgymenvs` as a python module for `PYTHON_PATH`:

```bash
PYTHON_PATH -m pip install -e .
```

The following error may appear during the initial installation. This error is harmless and can be ignored.

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
```

## Usage
```python
# play
PYTHON_PATH play_isaac.py task=Crazyflie num_envs=16

# train
PYTHON_PATH train_isaac.py task=Crazyflie headless=True wandb_activate=True

# evaluate 
PYTHON_PATH train_isaac.py task=Crazyflie test=True num_envs=16 checkpoint=runs/Ant/nn/Ant.pth
```


## TODO 

 - [x] move isaac crazyflie task related code to this repository
 - [ ] hanging an object to the bottom of the quadrotor (use rigid link to connect the object first, then add primismatic joint to simulate the loose rope. )
 - [ ] domain randomization support (the length of the rope and the hagging point's position relative to the quadrotor. )
 - [ ] multiple drone enviroment. 
