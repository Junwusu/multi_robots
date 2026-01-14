import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class RslRlAmpVecEnvWrapper(RslRlVecEnvWrapper):
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        obs_dict["amp"] = self.unwrapped.extras["amp_obs"]
        return obs_dict["policy"], {"observations": obs_dict}

class RslRlSymmetricVecEnvWrapper(RslRlVecEnvWrapper):
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        obs_dict["amp"] = self.unwrapped.extras["amp_obs"]
        return obs_dict["policy"], {"observations": obs_dict}


from tensordict import TensorDict

class RslRlExplicitEstimationVecEnvWrapper(RslRlVecEnvWrapper):
    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        print(obs_dict)
        return TensorDict(obs_dict, batch_size=[self.num_envs])
    