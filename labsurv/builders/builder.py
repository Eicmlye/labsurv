from mmcv.utils import Registry

from .build_env import build_env

AGENTS = Registry("agents")
ENVIRONMENTS = Registry("environments", build_func=build_env)
HOOKS = Registry("hooks")
LOSSES = Registry("losses")
STRATEGIES = Registry("strategies")
REPLAY_BUFFERS = Registry("replay_buffers")
RUNNERS = Registry("runners")
EXPLORERS = Registry("explorers")
