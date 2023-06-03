# Adaptive Issac Gym

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

 - [ ] move isaac crazyflie task related code to this repository
 - [ ] hanging an object to the bottom of the quadrotor (use rigid link to connect the object first, then add primismatic joint to simulate the loose rope. )
 - [ ] domain randomization support (the length of the rope and the hagging point's position relative to the quadrotor. )
 - [ ] multiple drone enviroment. 