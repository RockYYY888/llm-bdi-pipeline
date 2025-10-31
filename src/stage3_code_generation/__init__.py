"""
Stage 3: AgentSpeak Code Generation from DFA Decomposition

This module provides LLM-based AgentSpeak code generation guided by
DFA transition information from recursive DFA decomposition.
"""

from .agentspeak_generator import AgentSpeakGenerator

__all__ = [
    'AgentSpeakGenerator'
]
