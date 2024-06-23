import pickle
from typing import Any, Dict, Optional

# import numpy as np
# import pyntcloud as pcl
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseEnv


@ENVIRONMENTS.register_module()
class BaseSurveillanceEnv(BaseEnv):
    def __init__(
        self,
        surveil_cfg_path: str,
    ):
        """ """

        self.init_visibility(surveil_cfg_path)

        # should clarify the init observation distribution in convenience of `reset()`
        raise NotImplementedError()

    def step(self, observation, action) -> Dict[str, Any]:
        """ """
        assert action in self.action_space, "Unknown action."
        assert observation in self.observation_space, "Unknown input observation."

        transition = dict(
            next_observation=None,
            reward=None,
            terminated=None,
            truncated=None,
        )

        assert (
            transition["next_observation"] is None
            or transition["next_observation"] in self.observation_space
        ), "Unknown output observation."

        return transition

    def reset(self, seed: Optional[int] = None):
        """ """

        # do env init works
        super().reset(seed=seed)

        # return init observation according to observation distribution
        init_observation = None

        return init_observation

    def init_visibility(self, cfg_path: str):
        with open(cfg_path, "rb") as f:
            surv_cfg = pickle.load(f)

        occupancy = surv_cfg["occupancy"]
        install_permitted = surv_cfg["install_permitted"]
        must_monitor = surv_cfg["must_monitor"]

        if occupancy.ndim != 3:
            raise ValueError("`occupancy` must be a 3D space.")
        if occupancy.shape != install_permitted.shape:
            raise ValueError(
                f"`install_permitted` shape {install_permitted.shape} must match "
                f"`occupancy` shape {occupancy.shape}."
            )
        elif occupancy.shape != must_monitor.shape:
            raise ValueError(
                f"`must_monitor` shape {must_monitor.shape} must match  `occupancy`"
                f"shape {occupancy.shape}."
            )

        self.occupancy = occupancy
        self.install_permit_mask = install_permitted
        self.must_monitor_mask = must_monitor

    def save_pointcloud(points, filename):
        pass
