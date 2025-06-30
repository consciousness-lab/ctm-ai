from .ctm import ConsciousnessTuringMachine as CTM
from .ctm_base import BaseConsciousnessTuringMachine as BaseCTM

# Backward compatibility: ToolConsciousnessTuringMachine is now just an alias
ToolConsciousnessTuringMachine = CTM

__all__ = ['CTM', 'BaseCTM', 'ToolConsciousnessTuringMachine']
