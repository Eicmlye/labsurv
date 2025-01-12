from .ddpg import agent_cfg as ddpg_agent
from .ddpg_add_only import agent_cfg as ddpg_add_only_agent
from .ddpg_add_only_clean import agent_cfg as ddpg_add_only_clean_agent
from .reinforce import agent_cfg as reinforce_agent

__all__ = [
    "reinforce_agent",
    "ddpg_agent",
    "ddpg_add_only_agent",
    "ddpg_add_only_clean_agent",
]
