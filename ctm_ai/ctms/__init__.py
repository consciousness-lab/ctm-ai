from .ctm import ConsciousTuringMachine as CTM
from .ctm_base import BaseConsciousTuringMachine as BaseCTM
from .ctm_webagent import WebConsciousTuringMachine as WebCTM

ToolCTM = CTM

__all__ = ['CTM', 'BaseCTM', 'ToolCTM', 'WebCTM']
