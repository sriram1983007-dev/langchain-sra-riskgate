"""LangChain v1 agent middleware (``create_agent(..., middleware=[...])``)."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "SRARiskGateMiddleware needs LangChain 1.x: pip install 'langchain-sra-riskgate[middleware]'. "
        "On LangChain 0.3 (e.g. coinbase-agentkit-langchain), use guard_tools() instead."
    ) from e

from .gate import GateDecision, PaymentGate


class SRARiskGateMiddleware(AgentMiddleware):
    """Checks every money-moving tool call before it runs.

    Approved calls run normally. Held or rejected calls never reach the tool; the agent
    gets an error ``ToolMessage`` explaining why. Pass a :class:`PaymentGate` or its
    keyword arguments.
    """

    def __init__(self, gate: Optional[PaymentGate] = None, **gate_kwargs: Any):
        super().__init__()
        if gate is not None and gate_kwargs:
            raise TypeError("Pass either gate= or PaymentGate keyword arguments, not both.")
        self.gate = gate or PaymentGate(**gate_kwargs)

    @staticmethod
    def _blocked(request: Any, decision: GateDecision) -> ToolMessage:
        tc = request.tool_call
        return ToolMessage(
            content=decision.to_tool_message_text(),
            tool_call_id=tc["id"],
            name=tc["name"],
            status="error",
            artifact={"sra_riskgate": {"decision": decision.decision, "flags": list(decision.flags)}},
        )

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        tc = request.tool_call
        decision = self.gate.evaluate(tc["name"], tc.get("args"))
        if decision is None:
            return handler(request)
        if not decision.approved:
            return self._blocked(request, decision)
        try:
            return handler(request)
        except BaseException:
            self.gate.release(decision)
            raise

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        tc = request.tool_call
        # The optional model check does blocking I/O, so keep it off the event loop.
        decision = await asyncio.to_thread(self.gate.evaluate, tc["name"], tc.get("args"))
        if decision is None:
            return await handler(request)
        if not decision.approved:
            return self._blocked(request, decision)
        try:
            return await handler(request)
        except BaseException:
            self.gate.release(decision)
            raise
