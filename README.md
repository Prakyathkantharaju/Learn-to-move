# Learn-to-move
Reinforcement learning envs for locomotion systems. like walking, hopping and humanoid walking

Written in python using stable-baselines3 and gym.

## Install and setup
Installing the depdencies
```bash
pip install -r requirements.txt
```

# Hopper with disturbance
Custom hopper enviroment using mujoco.

## Training
```bash
python Models/hopper/hopper_train.py
```


## Test
```bash
python Models/hopper/hopper_test.py
```


# Video of the simulation

## training

https://user-images.githubusercontent.com/34353557/169610407-8e179f45-1124-4119-9d1a-8219b94cbd8e.mp4


## Test in adverse env

https://user-images.githubusercontent.com/34353557/169610432-877d6f03-9ee4-4ffe-b829-7cff5f1f28be.mp4


## Important rendering
Important information regarding rendering the model. 
 
 
 
If you are trying to render the image, please use the following command.

```bash
set -Ux LD_PRELOAD /usr/lib/x86_64-linux-gnu/libGLEW.so
```


Do not forget to delete the LD_PRELOAD variable.
```bash
unset LD_PRELOAD
```
