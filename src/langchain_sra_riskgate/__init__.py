"""Stop AI agents from making unsafe stablecoin payments: SRA-RiskGate for LangChain."""
from .gate import AGENTKIT_EXTRACTORS, GateDecision, PaymentGate, PaymentIntent
from .model import OpenAICompatibleModelCheck
from .tools import guard_tools

__version__ = "0.3.0"

__all__ = [
    "AGENTKIT_EXTRACTORS",
    "GateDecision",
    "OpenAICompatibleModelCheck",
    "PaymentGate",
    "PaymentIntent",
    "SRARiskGateMiddleware",
    "guard_tools",
]


def __getattr__(name: str):
    # Imported lazily so the package also works on LangChain 0.3, which has no middleware.
    if name == "SRARiskGateMiddleware":
        from .middleware import SRARiskGateMiddleware

        return SRARiskGateMiddleware
    raise AttributeError(name)
