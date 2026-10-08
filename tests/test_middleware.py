"""End-to-end through LangChain 1.x create_agent with a scripted chat model."""
import asyncio

import pytest

pytest.importorskip("langchain.agents.middleware")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from langchain_sra_riskgate import PaymentGate, SRARiskGateMiddleware  # noqa: E402
from sra_riskgate.chains import USDC_ADDRESSES  # noqa: E402

YOU = "0x" + "b" * 40
USDC = USDC_ADDRESSES[8453]
SENT = []


@tool("ERC20ActionProvider_transfer")
def erc20_transfer(amount: str, contract_address: str, destination_address: str) -> str:
    """Transfer ERC20 tokens."""
    SENT.append(amount)
    return f"Transferred {amount}"


@tool("WalletActionProvider_get_balance")
def get_balance() -> str:
    """Get balance."""
    return "42"


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(amount, call_id="c1"):
    return AIMessage(content="", tool_calls=[{
        "name": "ERC20ActionProvider_transfer", "id": call_id,
        "args": {"amount": str(amount), "contract_address": USDC, "destination_address": YOU}}])


def agent(script, **gate_kwargs):
    SENT.clear()
    model = ScriptedModel(messages=iter(script))
    return create_agent(model=model, tools=[erc20_transfer, get_balance],
                        middleware=[SRARiskGateMiddleware(**gate_kwargs)])


def tool_messages(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def test_approved_payment_executes():
    a = agent([call(5), AIMessage(content="done")], max_amount_usdc=10)
    msgs = tool_messages(a.invoke({"messages": [("user", "pay")]}))
    assert SENT == ["5"] and msgs[0].content == "Transferred 5"


def test_held_payment_never_reaches_tool():
    a = agent([call(500), AIMessage(content="ok")], max_amount_usdc=10)
    msgs = tool_messages(a.invoke({"messages": [("user", "pay")]}))
    assert SENT == []
    assert msgs[0].status == "error" and "HELD" in msgs[0].content
    assert msgs[0].artifact["sra_riskgate"]["flags"] == ["over_payment_limit"]


def test_non_payment_tool_untouched():
    script = [AIMessage(content="", tool_calls=[{"name": "WalletActionProvider_get_balance",
                                                 "id": "b", "args": {}}]),
              AIMessage(content="done")]
    msgs = tool_messages(agent(script).invoke({"messages": [("user", "bal")]}))
    assert msgs[0].content == "42" and msgs[0].status == "success"


def test_split_payments_hit_session_budget():
    script = [call(9, "a"), call(9, "b"), call(9, "c"), AIMessage(content="done")]
    a = agent(script, max_amount_usdc=10, session_budget_usdc=20)
    msgs = tool_messages(a.invoke({"messages": [("user", "pay")]}))
    assert SENT == ["9", "9"]
    assert "session_budget_exceeded" in msgs[2].content


def test_async_agent():
    a = agent([call(500), AIMessage(content="ok")], max_amount_usdc=10)
    msgs = tool_messages(asyncio.run(a.ainvoke({"messages": [("user", "pay")]})))
    assert SENT == [] and msgs[0].status == "error"


def test_middleware_accepts_shared_gate():
    gate = PaymentGate(max_amount_usdc=1)
    assert SRARiskGateMiddleware(gate).gate is gate
    with pytest.raises(TypeError):
        SRARiskGateMiddleware(gate, max_amount_usdc=2)
