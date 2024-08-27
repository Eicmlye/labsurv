from .dqn import agent_cfg as dqn_agent
from .reinforce import agent_cfg as reinforce_agent

__all__ = [
    "dqn_agent",
    "reinforce_agent",
]