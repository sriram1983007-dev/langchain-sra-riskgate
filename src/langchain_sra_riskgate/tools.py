"""Wrap LangChain tools directly. Works on langchain-core 0.3 and 1.x.

Use this with ``coinbase-agentkit-langchain`` (which pins LangChain 0.3) or with any agent
framework that takes LangChain tools.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

from langchain_core.tools import BaseTool, StructuredTool

from .gate import PaymentGate


def _wrap(tool: BaseTool, gate: PaymentGate) -> BaseTool:
    def run(**kwargs: Any) -> Any:
        decision = gate.evaluate(tool.name, kwargs)
        if decision is not None and not decision.approved:
            return decision.to_tool_message_text()
        try:
            return tool.invoke(kwargs)
        except BaseException:
            gate.release(decision)
            raise

    async def arun(**kwargs: Any) -> Any:
        decision = await asyncio.to_thread(gate.evaluate, tool.name, kwargs)
        if decision is not None and not decision.approved:
            return decision.to_tool_message_text()
        try:
            return await tool.ainvoke(kwargs)
        except BaseException:
            gate.release(decision)
            raise

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=run,
        coroutine=arun,
        return_direct=tool.return_direct,
    )


def guard_tools(
    tools: Sequence[BaseTool], gate: Optional[PaymentGate] = None, **gate_kwargs: Any
) -> list[BaseTool]:
    """Return ``tools`` with every money-moving tool wrapped by the gate.

    Other tools are returned unchanged. Pass a :class:`PaymentGate` or its keyword arguments.
    """
    if gate is not None and gate_kwargs:
        raise TypeError("Pass either gate= or PaymentGate keyword arguments, not both.")
    gate = gate or PaymentGate(**gate_kwargs)
    return [_wrap(t, gate) if gate.is_gated(t.name) else t for t in tools]
