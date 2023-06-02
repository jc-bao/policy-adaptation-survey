# Adaptive Issac Gym

## Usage
```python
# train
PYTHON_PATH train_isaac.py task=Crazyflie headless=True

# evaluate 
PYTHON_PATH train_isaac.py task=Crazyflie checkpoint=runs/Ant/nn/Ant.pth test=True num_envs=64
```