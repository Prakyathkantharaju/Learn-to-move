from hopper_full import HopperMine

# GlfwContext(offscreen=True)  # Create a window to init GLFW.

env = HopperMine('/home/prakyathkantharaju/gitfolder/personal/Learn-to-move/gym_envs/hopper_full_view/mujoco_models/hopper.xml',1)


for i in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    print(observation.shape)





