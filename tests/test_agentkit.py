"""Against the real coinbase-agentkit + coinbase-agentkit-langchain tools (LangChain 0.3)."""
from decimal import Decimal

import pytest

pytest.importorskip("coinbase_agentkit_langchain")
from coinbase_agentkit import (  # noqa: E402
    AgentKit, AgentKitConfig, EvmWalletProvider, erc20_action_provider, wallet_action_provider,
    x402_action_provider)
from coinbase_agentkit.network import Network  # noqa: E402
from coinbase_agentkit_langchain import get_langchain_tools  # noqa: E402

from langchain_sra_riskgate import AGENTKIT_EXTRACTORS, PaymentGate, guard_tools  # noqa: E402
from sra_riskgate.chains import USDC_ADDRESSES  # noqa: E402

ME, YOU = "0x" + "a" * 40, "0x" + "b" * 40
USDC = USDC_ADDRESSES[84532]


class SpyWallet(EvmWalletProvider):
    """Records every call; any call after the gate means the action actually ran."""

    def __init__(self):
        self.calls = []

    def _hit(self, name):
        self.calls.append(name)

    def get_address(self): return ME
    def get_name(self): return "spy"
    def get_network(self): return Network(protocol_family="evm", network_id="base-sepolia", chain_id="84532")
    def get_balance(self): self._hit("get_balance"); return Decimal(0)
    def native_transfer(self, to, value): self._hit("native_transfer"); return "0xnative"
    def send_transaction(self, tx): self._hit("send_transaction"); return "0xhash"
    def wait_for_transaction_receipt(self, *a, **k): return {"status": 1}
    def read_contract(self, *a, **k): self._hit("read_contract"); raise RuntimeError("no rpc in tests")
    def sign_message(self, *a): raise NotImplementedError
    def sign_transaction(self, *a): raise NotImplementedError
    def sign_typed_data(self, *a): raise NotImplementedError

    def __getattr__(self, name):  # anything else, e.g. multicall helpers
        if name.startswith("_"):
            raise AttributeError(name)
        def fn(*a, **k):
            self._hit(name)
            raise RuntimeError("no rpc in tests")
        return fn


@pytest.fixture
def kit_tools(monkeypatch):
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    wallet = SpyWallet()
    kit = AgentKit(AgentKitConfig(wallet_provider=wallet, action_providers=[
        wallet_action_provider(), erc20_action_provider(), x402_action_provider()]))
    return wallet, get_langchain_tools(kit)


def by_name(tools):
    return {t.name: t for t in tools}


def test_real_agentkit_tool_names_are_covered(kit_tools):
    _, tools = kit_tools
    names = {t.name for t in tools}
    assert set(AGENTKIT_EXTRACTORS) <= names, set(AGENTKIT_EXTRACTORS) - names


def test_only_money_tools_wrapped(kit_tools):
    _, tools = kit_tools
    gate = PaymentGate(chain_id=84532)
    guarded = guard_tools(tools, gate=gate)
    wrapped = {g.name for g, t in zip(guarded, tools) if g is not t}
    assert set(AGENTKIT_EXTRACTORS) <= wrapped
    assert "WalletActionProvider_get_balance" not in wrapped
    assert "ERC20ActionProvider_get_balance" not in wrapped
    assert "x402ActionProvider_make_http_request" not in wrapped


def test_erc20_over_limit_never_touches_wallet(kit_tools):
    wallet, tools = kit_tools
    t = by_name(guard_tools(tools, chain_id=84532, max_amount_usdc=10))["ERC20ActionProvider_transfer"]
    out = t.invoke({"amount": "500", "contract_address": USDC, "destination_address": YOU})
    assert "NOT sent" in out and wallet.calls == []


def test_erc20_within_limit_reaches_agentkit(kit_tools):
    wallet, tools = kit_tools
    t = by_name(guard_tools(tools, chain_id=84532, max_amount_usdc=10))["ERC20ActionProvider_transfer"]
    t.invoke({"amount": "5", "contract_address": USDC, "destination_address": YOU})
    assert wallet.calls  # AgentKit's own action ran (and failed on our fake RPC)


def test_native_transfer_held(kit_tools):
    wallet, tools = kit_tools
    t = by_name(guard_tools(tools))["WalletActionProvider_native_transfer"]
    out = t.invoke({"to": YOU, "value": "1"})
    assert "unsupported_asset" in out and wallet.calls == []


def test_x402_retry_over_limit_held(kit_tools):
    wallet, tools = kit_tools
    t = by_name(guard_tools(tools, max_amount_usdc=1))["x402ActionProvider_retry_with_x402"]
    out = t.invoke({
        "url": "https://api.example.com/x", "scheme": "exact", "network": "base-sepolia",
        "max_amount_required": "5000000", "resource": "https://api.example.com/x",
        "pay_to": YOU, "max_timeout_seconds": 60, "asset": USDC,
    })
    assert "over_payment_limit" in out and wallet.calls == []


def test_x402_direct_held(kit_tools):
    wallet, tools = kit_tools
    t = by_name(guard_tools(tools))["x402ActionProvider_make_http_request_with_x402"]
    assert "x402_amount_unknown" in t.invoke({"url": "https://api.example.com/x"})
    assert wallet.calls == []
