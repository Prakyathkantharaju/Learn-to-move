# Learn-to-move
Reinforcement learning envs for locomotion systems. like walking, hopping and humanoid walking

Written in python using stable-baselines3 and gym.

## Install and setup
Installing the depdencies
```bash
pip install -r requirements.txt
```

# Blind Hopper with disturbance

In this env the hopper has not information about the surroundings. It's only objective is to **move forward with high velocity**.





## Training

I have made training in multiple env together using [PPO policy](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

```bash
python Models/hopper/hopper_train_large_models.py
```


## Test
```bash
python Models/hopper/hopper_test.py
```

## Performing video.

### Hard_evn
https://user-images.githubusercontent.com/34353557/169736742-fd08db6a-a5ab-447f-abd6-2347b9571295.mp4

### wall jump

https://user-images.githubusercontent.com/34353557/169736884-68638296-e370-4149-abb9-e588e0f878ba.mp4

### small-no dist

https://user-images.githubusercontent.com/34353557/169737065-396608b3-f7c3-4611-8e5d-c24e25790e56.mp4






## Important rendering
Important information regarding rendering the model. 




## Notes on the dm control parkour.

The PPO is not working on the dm control parkour env.
As shown int he video above.

Possible reasons:
- The Env is not able to stand up.
- The Env is too hard for the current hopper. Need more actions.
So I am moving on to the hopper openai env. 
 
 
If you are trying to render the image, please use the following command.

```bash
set -Ux LD_PRELOAD /usr/lib/x86_64-linux-gnu/libGLEW.so
```


Do not forget to delete the LD_PRELOAD variable.
```bash
unset LD_PRELOAD
```
