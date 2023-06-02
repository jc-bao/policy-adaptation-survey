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