import asyncio

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from langchain_sra_riskgate import PaymentGate, guard_tools
from sra_riskgate.chains import USDC_ADDRESSES

YOU = "0x" + "b" * 40
USDC = USDC_ADDRESSES[8453]


class TransferSchema(BaseModel):
    amount: str
    contract_address: str
    destination_address: str


def make_tools(calls):
    def transfer(**kw):
        calls.append(kw)
        return "sent"

    async def atransfer(**kw):
        calls.append(kw)
        return "sent-async"

    def balance():
        return "100"

    return [
        StructuredTool(name="ERC20ActionProvider_transfer", description="send tokens",
                       args_schema=TransferSchema, func=transfer, coroutine=atransfer),
        StructuredTool.from_function(balance, name="WalletActionProvider_get_balance",
                                     description="balance"),
    ]


def args(amount):
    return {"amount": str(amount), "contract_address": USDC, "destination_address": YOU}


def test_only_payment_tools_wrapped_and_metadata_kept():
    calls = []
    raw = make_tools(calls)
    guarded = guard_tools(raw, max_amount_usdc=10)
    assert guarded[1] is raw[1]
    assert guarded[0] is not raw[0]
    assert guarded[0].name == raw[0].name and guarded[0].description == raw[0].description
    assert guarded[0].args_schema is TransferSchema


def test_approved_runs_blocked_does_not():
    calls = []
    t = guard_tools(make_tools(calls), max_amount_usdc=10)[0]
    assert t.invoke(args(5)) == "sent"
    out = t.invoke(args(50))
    assert "NOT sent" in out and len(calls) == 1


def test_async_path():
    calls = []
    t = guard_tools(make_tools(calls), max_amount_usdc=10)[0]
    assert asyncio.run(t.ainvoke(args(5))) == "sent-async"
    assert "NOT sent" in asyncio.run(t.ainvoke(args(50)))
    assert len(calls) == 1


def test_budget_released_when_tool_raises():
    def boom(**kw):
        raise RuntimeError("rpc down")
    tool = StructuredTool(name="ERC20ActionProvider_transfer", description="d",
                          args_schema=TransferSchema, func=boom)
    gate = PaymentGate(session_budget_usdc=10)
    t = guard_tools([tool], gate=gate)[0]
    with pytest.raises(RuntimeError):
        t.invoke(args(8))
    assert gate.spent_usdc == 0


def test_gate_and_kwargs_are_exclusive():
    with pytest.raises(TypeError):
        guard_tools([], gate=PaymentGate(), max_amount_usdc=1)
