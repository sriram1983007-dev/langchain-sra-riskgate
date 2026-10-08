from decimal import Decimal

import pytest

from langchain_sra_riskgate import PaymentGate
from sra_riskgate.chains import USDC_ADDRESSES

ME, YOU, BAD = "0x" + "a" * 40, "0x" + "b" * 40, "0x" + "c" * 40
USDC_BASE = USDC_ADDRESSES[8453]
ERC20 = "ERC20ActionProvider_transfer"
X402_RETRY = "x402ActionProvider_retry_with_x402"
X402_DIRECT = "x402ActionProvider_make_http_request_with_x402"


def usdc(amount, to=YOU, contract=USDC_BASE):
    return {"amount": str(amount), "contract_address": contract, "destination_address": to}


def x402_opt(amount_atomic="10000", pay_to=YOU, network="eip155:8453", asset=USDC_BASE, **extra):
    opt = {"scheme": "exact", "network": network, "asset": asset, "amount": amount_atomic, "pay_to": pay_to}
    opt.update(extra)
    return {"url": "https://api.example.com/data", "selected_payment_option": opt}


def test_non_payment_tools_pass_through():
    g = PaymentGate()
    assert g.evaluate("WalletActionProvider_get_balance", {}) is None
    assert g.evaluate("search_web", {"q": "x"}) is None


def test_small_usdc_transfer_approved_and_reserved():
    g = PaymentGate(max_amount_usdc=100)
    d = g.evaluate(ERC20, usdc("25.5"))
    assert d.approved and d.intent.is_usdc and d.intent.chain_id == 8453
    assert g.spent_usdc == Decimal("25.5")


def test_over_limit_held():
    d = PaymentGate(max_amount_usdc=100).evaluate(ERC20, usdc("100.01"))
    assert d.decision == "hold" and d.flags == ("over_payment_limit",)


def test_exact_limit_approved():
    assert PaymentGate(max_amount_usdc=100).evaluate(ERC20, usdc("100")).approved


def test_blocked_and_invalid_recipients_rejected():
    g = PaymentGate(blocked_recipients=[BAD.upper().replace("0X", "0x")])
    assert g.evaluate(ERC20, usdc(1, to=BAD)).flags == ("blocked_recipient",)
    for to in ("vitalik.eth", "0x123", "0x" + "0" * 40, ""):
        assert g.evaluate(ERC20, usdc(1, to=to)).decision == "reject"


def test_self_transfer_rejected():
    assert PaymentGate(wallet_address=ME).evaluate(ERC20, usdc(1, to=ME)).flags == ("self_transfer",)


def test_allowlist_holds_unknown_recipient():
    g = PaymentGate(allowed_recipients=[YOU])
    assert g.evaluate(ERC20, usdc(1)).approved
    assert g.evaluate(ERC20, usdc(1, to=BAD)).flags == ("recipient_not_allowlisted",)


@pytest.mark.parametrize("amount", ["0", "-5", "abc", "nan", "inf", None])
def test_bad_amounts_rejected(amount):
    d = PaymentGate().evaluate(ERC20, usdc(amount))
    assert d.decision == "reject" and d.flags == ("invalid_payment_arguments",)


def test_non_usdc_token_held_by_default():
    other = "0x" + "d" * 40
    d = PaymentGate().evaluate(ERC20, usdc(1, contract=other))
    assert d.decision == "hold" and d.flags == ("unsupported_asset",)
    d = PaymentGate(allow_non_usdc=True).evaluate(ERC20, usdc(10**9, contract=other))
    assert d.approved and d.flags == ("non_usdc_amount_unchecked",)


def test_usdc_on_other_chain_not_recognised_when_chain_pinned():
    # Base USDC contract while the wallet is on Ethereum mainnet: not USDC there.
    d = PaymentGate(chain_id=1).evaluate(ERC20, usdc(1))
    assert d.flags == ("unsupported_asset",)


def test_native_transfer_held_by_default():
    d = PaymentGate().evaluate("WalletActionProvider_native_transfer", {"to": YOU, "value": "0.5"})
    assert d.decision == "hold" and d.intent.asset == "native"


def test_approve_is_gated_like_a_transfer():
    args = {"amount": "1000000", "contract_address": USDC_BASE, "spender_address": YOU}
    d = PaymentGate(max_amount_usdc=100).evaluate("ERC20ActionProvider_approve", args)
    assert d.decision == "hold" and d.intent.kind == "erc20_approve"


def test_session_budget_stops_splitting():
    g = PaymentGate(max_amount_usdc=100, session_budget_usdc=150)
    assert g.evaluate(ERC20, usdc(90)).approved
    d = g.evaluate(ERC20, usdc(90))
    assert d.flags == ("session_budget_exceeded",)
    assert g.spent_usdc == 90


def test_release_returns_budget():
    g = PaymentGate(session_budget_usdc=100)
    d = g.evaluate(ERC20, usdc(80))
    g.release(d)
    assert g.spent_usdc == 0 and g.evaluate(ERC20, usdc(80)).approved


def test_depeg_thresholds_and_feed_errors():
    def gate(v):
        return PaymentGate(peg_feed=lambda c: v)
    assert gate(-0.2).evaluate(ERC20, usdc(1)).approved
    assert gate(-1.5).evaluate(ERC20, usdc(1)).flags == ("moderate_depeg",)
    assert gate(6).evaluate(ERC20, usdc(1)).flags == ("critical_depeg",)
    assert gate(float("nan")).evaluate(ERC20, usdc(1)).flags == ("peg_feed_error",)

    def boom(c):
        raise TimeoutError
    assert PaymentGate(peg_feed=boom).evaluate(ERC20, usdc(1)).flags == ("peg_feed_error",)


def test_x402_retry_atomic_amount_converted():
    d = PaymentGate().evaluate(X402_RETRY, x402_opt("10000"))
    assert d.approved and d.intent.amount == Decimal("0.01") and d.intent.chain_id == 8453


def test_x402_retry_over_limit_and_legacy_fields():
    d = PaymentGate(max_amount_usdc=1).evaluate(
        X402_RETRY, x402_opt(None, network="base", max_amount_required="5000000"))
    assert d.flags == ("over_payment_limit",)


def test_x402_retry_flat_v1_fields_agentkit_074():
    args = {"url": "https://x", "scheme": "exact", "network": "base", "max_amount_required": "2500000",
            "resource": "https://x", "pay_to": YOU, "max_timeout_seconds": 60, "asset": USDC_BASE}
    d = PaymentGate(max_amount_usdc=1).evaluate(X402_RETRY, args)
    assert d.flags == ("over_payment_limit",) and d.intent.amount == Decimal("2.5")


def test_x402_retry_without_amount_rejected():
    d = PaymentGate().evaluate(X402_RETRY, {"url": "https://x", "pay_to": YOU})
    assert d.decision == "reject"


def test_x402_retry_accepts_pydantic_like_objects():
    class Opt:
        scheme, network, asset, amount, pay_to = "exact", "eip155:8453", USDC_BASE, "20000", YOU
        max_amount_required = price = None
    d = PaymentGate().evaluate(X402_RETRY, {"url": "https://x", "selected_payment_option": Opt()})
    assert d.approved and d.intent.amount == Decimal("0.02")


def test_x402_non_usdc_asset_held():
    d = PaymentGate().evaluate(X402_RETRY, x402_opt(asset="0x" + "e" * 40))
    assert d.flags == ("unsupported_asset",)


def test_x402_direct_held_unless_host_allowed():
    args = {"url": "https://api.example.com/data"}
    assert PaymentGate().evaluate(X402_DIRECT, args).flags == ("x402_amount_unknown",)
    assert PaymentGate(allowed_x402_hosts=["api.example.com"]).evaluate(X402_DIRECT, args).approved


def test_strict_mode_holds_unknown_money_tools():
    g = PaymentGate()
    assert g.evaluate("pay_invoice", {"id": 1}).flags == ("ungated_money_tool",)
    assert PaymentGate(passthrough_tools=["pay_invoice"]).evaluate("pay_invoice", {}) is None
    assert PaymentGate(strict=False).evaluate("pay_invoice", {}) is None


@pytest.mark.parametrize("name,gated", [
    ("WowActionProvider_buy_token", True), ("WethActionProvider_wrap_eth", True),
    ("sendPayment", True), ("AaveActionProvider_supply", True),
    ("x402ActionProvider_make_http_request", False), ("x402ActionProvider_discover_x402_services", False),
    ("get_payment_status", False), ("display_chart", False), ("CdpApiActionProvider_request_faucet_funds", False),
])
def test_strict_mode_word_matching(name, gated):
    assert PaymentGate().is_gated(name) is gated


def test_custom_extractor():
    from langchain_sra_riskgate import PaymentIntent

    def pay_vendor(tool, args, chain):
        return PaymentIntent(tool, "custom", args["to"], Decimal(str(args["usd"])), "USDC", chain)
    g = PaymentGate(extractors={"pay_vendor": pay_vendor}, max_amount_usdc=10)
    assert g.evaluate("pay_vendor", {"to": YOU, "usd": 5}).approved
    assert g.evaluate("pay_vendor", {"to": YOU, "usd": 50}).decision == "hold"


def test_extractor_bug_fails_closed():
    def broken(tool, args, chain):
        raise RuntimeError("bug")
    d = PaymentGate(extractors={"pay": broken}).evaluate("pay", {})
    assert not d.approved


def test_internal_error_fails_closed(monkeypatch):
    g = PaymentGate()
    monkeypatch.setattr(g, "_evaluate", lambda *a: 1 / 0)
    d = g.evaluate(ERC20, usdc(1))
    assert d.decision == "hold" and d.source == "fail_closed"


def test_model_check_can_only_tighten():
    seen = []

    def model(intent, args):
        seen.append((intent, args))
        return {"decision": "reject", "flags": ["prompt_injection"], "explanation": "injected"}
    g = PaymentGate(model_check=model)
    d = g.evaluate(ERC20, usdc(1))
    assert d.decision == "reject" and d.flags == ("prompt_injection",) and d.source == "model"
    assert seen[0][1]["destination_address"] == YOU
    assert g.spent_usdc == 0
    # The model is never asked about payments the rules already stopped.
    seen.clear()
    PaymentGate(model_check=model, max_amount_usdc=1).evaluate(ERC20, usdc(5))
    assert seen == []


@pytest.mark.parametrize("out", [{}, {"decision": "yes"}, None])
def test_model_bad_output_holds(out):
    d = PaymentGate(model_check=lambda i, a: out or {}).evaluate(ERC20, usdc(1))
    assert d.decision == "hold"


def test_model_exception_holds():
    def model(i, a):
        raise ConnectionError
    d = PaymentGate(model_check=model).evaluate(ERC20, usdc(1))
    assert d.flags == ("model_unavailable",)


def test_on_decision_called_and_errors_swallowed():
    log = []
    PaymentGate(on_decision=log.append).evaluate(ERC20, usdc(1))
    assert log[0].approved

    def bad(d):
        raise ValueError
    with pytest.warns(RuntimeWarning):
        assert PaymentGate(on_decision=bad).evaluate(ERC20, usdc(1)).approved


def test_block_message_tells_agent_not_to_retry():
    text = PaymentGate(max_amount_usdc=1).evaluate(ERC20, usdc(5)).to_tool_message_text()
    assert "NOT sent" in text and "Do not retry" in text and "over_payment_limit" in text


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        PaymentGate(max_amount_usdc=0)
    with pytest.raises(ValueError):
        PaymentGate(depeg_hold_pct=5, depeg_reject_pct=1)
