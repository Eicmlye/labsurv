from copy import deepcopy

import gym
from labsurv.utils.string import WARN
from mmcv import ConfigDict
from mmcv.utils import Registry, build_from_cfg


def build_env(env_cfg: ConfigDict, registry: Registry):
    """
    ## Description:

        Environment builder wrapper.

    ## Arguments:

        env_cfg (ConfigDict): if `id` is an existed key, `gym` envs will be built.
        Otherwise use customized envs registered by `mmcv`.

        registry (Registry): `mmcv` registry class.

    ## Returns:

        An environment object.
    """
    cfg = deepcopy(env_cfg)
    if "id" in cfg.keys():
        id = cfg.pop("id")
        if "type" in cfg.keys():
            cfg.pop("type")
            env_cfg.pop("type")
            print(WARN(f"Both `id` and `type` found in config, use gym env `{id}`."))
        return gym.make(id, **cfg)
    elif "type" in cfg.keys():
        return build_from_cfg(cfg, registry)

    raise KeyError("Neither `id` nor `type` is found in the config file.")
