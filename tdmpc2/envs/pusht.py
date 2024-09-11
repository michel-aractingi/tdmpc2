import numpy as np
import gymnasium as gym
import gym_pusht
import cv2


class PushTWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg

    def reset(self, **kwargs):
        ret = super().reset(**kwargs)[0]
        if self.env.obs_type == "environment_state_agent_pos":
            obs = np.concatenate((ret['environment_state'], ret['agent_pos'])).astype(np.float32)
        elif self.env.obs_type == "pixels_agent_pos":
            pixels = cv2.resize(ret['pixels'], dsize=(64,64), interpolation=cv2.INTER_AREA)#.astype(np.float32)
            obs = {"rgb": pixels.reshape((3, 64, 64)), "state": ret['agent_pos']}["rgb"]
        self.env.step(np.zeros(self.env.action_space.shape))
        return obs

    def step(self, action):
        observation, reward, terminated, truncated, info =  self.env.step(action)
        if self.env.obs_type == "environment_state_agent_pos":
            observation = np.concatenate((observation['environment_state'], observation['agent_pos'])).astype(np.float32)
        elif self.env.obs_type == "pixels_agent_pos":
            pixels = cv2.resize(observation['pixels'], dsize=(64,64), interpolation=cv2.INTER_AREA)#.astype(np.float32)
            observation = {"rgb": pixels.reshape((3, 64, 64)), "state": observation['agent_pos']}["rgb"]
        done = False
        if terminated or truncated: done = True
        return observation, reward, done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render().copy()


def make_env(cfg):
    """
    Make Push-T environment.
    """
    if "pusht" not in cfg.task: raise ValueError("Unkown Task.")
    obs_type = cfg.obs
    if cfg.obs=='state': obs_type = "environment_state_agent_pos"
    if cfg.obs=='rgb': obs_type = "pixels_agent_pos"
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type=obs_type)
    env = PushTWrapper(env, cfg)
    env.max_episode_steps = env.env._max_episode_steps#env.env.env.env._max_episode_steps
    return env
