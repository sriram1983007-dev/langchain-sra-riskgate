"""The payment gate: decides approve / hold / reject for a money-moving tool call.

Everything here fails closed. A bug, a bad argument, an unreachable price feed or an
unreachable model produces ``hold`` or ``reject``, never ``approve``.
"""
from __future__ import annotations

import math
import re
import threading
import warnings
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib.parse import urlparse

from sra_riskgate.chains import USDC_ADDRESSES, USDC_DECIMALS, chain_id_from_network, is_usdc

APPROVE, HOLD, REJECT = "approve", "hold", "reject"

_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_ZERO_ADDRESS = "0x" + "0" * 40

# Verbs that suggest a tool moves money. In strict mode (the default) a tool whose name
# contains one of these words, but that has no extractor, is held rather than allowed through.
_MONEY_VERBS = frozenset({
    "transfer", "send", "pay", "approve", "withdraw", "swap", "trade", "buy", "sell", "supply",
    "deposit", "borrow", "repay", "mint", "bridge", "wrap", "unwrap", "stake", "unstake",
    "purchase", "checkout", "tip", "bet",
})


def _name_words(tool_name: str) -> set[str]:
    """Words in a tool's own name, ignoring an AgentKit ``XxxActionProvider_`` prefix."""
    method = tool_name.split("ActionProvider_", 1)[-1]
    method = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", method)
    return {w for w in re.split(r"[^A-Za-z0-9]+", method.lower()) if w}


@dataclass(frozen=True)
class PaymentIntent:
    """What a tool call would do with money, in whole token units."""

    tool: str
    kind: str
    recipient: Optional[str]
    amount: Optional[Decimal]
    asset: str
    chain_id: Optional[int] = None
    url: Optional[str] = None

    @property
    def is_usdc(self) -> bool:
        return self.asset == "USDC"


@dataclass(frozen=True)
class GateDecision:
    decision: str
    flags: tuple[str, ...]
    explanation: str
    intent: Optional[PaymentIntent] = None
    source: str = "rules"
    model_verdict: Optional[dict] = field(default=None, compare=False)

    @property
    def approved(self) -> bool:
        return self.decision == APPROVE

    def to_tool_message_text(self) -> str:
        """Text returned to the agent in place of the tool result when a call is stopped."""
        word = "HELD for human review" if self.decision == HOLD else "REJECTED"
        return (
            f"SRA-RiskGate {word}: this payment was NOT sent. {self.explanation} "
            f"Flags: {', '.join(self.flags) or 'none'}. Do not retry this payment with changed "
            "details or split it into smaller payments; tell the user it needs their review."
        )


# ---------------------------------------------------------------- AgentKit extractors


def _get(obj: Any, *names: str) -> Any:
    """Read a field from a dict or an object (AgentKit passes pydantic models)."""
    for n in names:
        if isinstance(obj, Mapping):
            if obj.get(n) not in (None, ""):
                return obj[n]
        elif getattr(obj, n, None) not in (None, ""):
            return getattr(obj, n)
    return None


def _decimal(value: Any, what: str) -> Decimal:
    try:
        d = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"{what} is not a number: {value!r}") from e
    if not d.is_finite() or d <= 0:
        raise ValueError(f"{what} must be a positive finite number, got {value!r}")
    return d


def _usdc_chain(contract: str, chain_id: Optional[int]) -> Optional[int]:
    """Chain id if ``contract`` is the known USDC contract (on ``chain_id`` if given)."""
    if chain_id is not None:
        return chain_id if is_usdc(chain_id, contract) else None
    for cid, addr in USDC_ADDRESSES.items():
        if addr.lower() == str(contract).lower():
            return cid
    return None


def _native_transfer(tool: str, args: Mapping[str, Any], chain_id: Optional[int]) -> PaymentIntent:
    return PaymentIntent(tool, "native_transfer", _get(args, "to"),
                         _decimal(_get(args, "value"), "value"), "native", chain_id)


def _erc20(kind: str, recipient_field: str):
    def extract(tool: str, args: Mapping[str, Any], chain_id: Optional[int]) -> PaymentIntent:
        contract = str(_get(args, "contract_address") or "")
        usdc_chain = _usdc_chain(contract, chain_id)
        return PaymentIntent(
            tool, kind, _get(args, recipient_field), _decimal(_get(args, "amount"), "amount"),
            "USDC" if usdc_chain is not None else contract, usdc_chain or chain_id,
        )
    return extract


def _x402_retry(tool: str, args: Mapping[str, Any], chain_id: Optional[int]) -> PaymentIntent:
    # AgentKit 0.7.x (x402 v1) passes the payment fields flat; newer AgentKit nests
    # them under selected_payment_option.
    opt = _get(args, "selected_payment_option") or args
    network_chain = chain_id_from_network(_get(opt, "network"))
    asset = str(_get(opt, "asset") or "")
    usdc = is_usdc(network_chain, asset)
    atomic = _get(opt, "max_amount_required", "amount")
    if atomic is not None:
        amount = _decimal(atomic, "payment amount")
        if usdc:
            amount = amount / (Decimal(10) ** USDC_DECIMALS)
    else:
        price = str(_get(opt, "price") or "")
        if not price.startswith("$"):
            raise ValueError("payment option has no amount")
        amount = _decimal(price[1:], "price")
    return PaymentIntent(tool, "x402_payment", _get(opt, "pay_to", "payTo"), amount,
                         "USDC" if usdc else asset, network_chain, _get(args, "url"))


def _x402_direct(tool: str, args: Mapping[str, Any], chain_id: Optional[int]) -> PaymentIntent:
    # The price is only known after the server answers 402, so there is no amount yet.
    return PaymentIntent(tool, "x402_unpriced", None, None, "USDC", chain_id, _get(args, "url"))


Extractor = Callable[[str, Mapping[str, Any], Optional[int]], PaymentIntent]

#: Coinbase AgentKit (Python) tool names, as produced by ``get_langchain_tools``.
AGENTKIT_EXTRACTORS: dict[str, Extractor] = {
    "WalletActionProvider_native_transfer": _native_transfer,
    "ERC20ActionProvider_transfer": _erc20("erc20_transfer", "destination_address"),
    "ERC20ActionProvider_approve": _erc20("erc20_approve", "spender_address"),
    "x402ActionProvider_retry_with_x402": _x402_retry,
    "x402ActionProvider_make_http_request_with_x402": _x402_direct,
}


# ---------------------------------------------------------------- the gate


def _addr(a: Optional[str]) -> str:
    return str(a or "").strip().lower()


class PaymentGate:
    """Deterministic checks on money-moving tool calls, plus an optional model check.

    Args:
        max_amount_usdc: Largest single USDC payment approved without review.
        session_budget_usdc: Total USDC this gate will approve over its lifetime
            (one gate per agent session). Stops an agent splitting a large payment into
            many small ones. ``None`` disables it.
        allowed_recipients: If set, any other recipient is held for review.
        blocked_recipients: Always rejected.
        allow_non_usdc: Let native-token and non-USDC token payments through. They are
            not covered by the USDC limits, so this is off by default (they are held).
        chain_id: The wallet's chain. Used to recognise the USDC contract.
        allowed_x402_hosts: Hosts the agent may call with AgentKit's one-step
            ``make_http_request_with_x402``. That action pays before the price is known,
            so calls to other hosts are held.
        wallet_address: The agent's own address, used to reject self-transfers.
        peg_feed: ``peg_feed(chain_id) -> float``, the USDC peg deviation in percent.
        depeg_hold_pct / depeg_reject_pct: Peg-deviation thresholds.
        model_check: ``model_check(intent, args) -> dict`` with at least ``decision``.
            Runs only after the rules approve. See :class:`OpenAICompatibleModelCheck`.
        extractors: Extra or replacement extractors, keyed by tool name.
        strict: Hold money-like tools (transfer, swap, pay, ...) that have no extractor.
        passthrough_tools: Tool names never gated, even in strict mode.
        on_decision: Called with every :class:`GateDecision`, e.g. for an audit log.
    """

    def __init__(
        self,
        *,
        max_amount_usdc: float = 100.0,
        session_budget_usdc: Optional[float] = None,
        allowed_recipients: Optional[Iterable[str]] = None,
        blocked_recipients: Iterable[str] = (),
        allow_non_usdc: bool = False,
        chain_id: Optional[int] = None,
        allowed_x402_hosts: Iterable[str] = (),
        wallet_address: Optional[str] = None,
        peg_feed: Optional[Callable[[Optional[int]], float]] = None,
        depeg_hold_pct: float = 1.0,
        depeg_reject_pct: float = 5.0,
        model_check: Optional[Callable[[PaymentIntent, Mapping[str, Any]], Mapping[str, Any]]] = None,
        extractors: Optional[Mapping[str, Extractor]] = None,
        strict: bool = True,
        passthrough_tools: Iterable[str] = (),
        on_decision: Optional[Callable[[GateDecision], None]] = None,
    ):
        self.max_amount = _decimal(max_amount_usdc, "max_amount_usdc")
        self.session_budget = (
            None if session_budget_usdc is None else _decimal(session_budget_usdc, "session_budget_usdc")
        )
        if not 0 < depeg_hold_pct < depeg_reject_pct:
            raise ValueError("Thresholds must satisfy 0 < depeg_hold_pct < depeg_reject_pct.")
        self.allowed = None if allowed_recipients is None else {_addr(a) for a in allowed_recipients}
        self.blocked = {_addr(a) for a in blocked_recipients}
        self.allow_non_usdc = allow_non_usdc
        self.chain_id = chain_id
        self.x402_hosts = {h.lower() for h in allowed_x402_hosts}
        self.wallet = _addr(wallet_address) or None
        self.peg_feed = peg_feed
        self.depeg_hold = depeg_hold_pct
        self.depeg_reject = depeg_reject_pct
        self.model_check = model_check
        self.extractors = {k.lower(): v for k, v in {**AGENTKIT_EXTRACTORS, **(extractors or {})}.items()}
        self.strict = strict
        self.passthrough = {t.lower() for t in passthrough_tools}
        self.on_decision = on_decision
        self._lock = threading.Lock()
        self._spent = Decimal(0)

    # ---- which tools are gated

    def is_gated(self, tool_name: str) -> bool:
        name = tool_name.lower()
        if name in self.passthrough:
            return False
        return name in self.extractors or (self.strict and bool(_name_words(tool_name) & _MONEY_VERBS))

    @property
    def spent_usdc(self) -> Decimal:
        return self._spent

    # ---- the decision

    def evaluate(self, tool_name: str, args: Optional[Mapping[str, Any]]) -> Optional[GateDecision]:
        """Return ``None`` for tools that are not gated, else a decision.

        An ``approve`` reserves the amount against the session budget; call
        :meth:`release` if the tool then fails.
        """
        if not self.is_gated(tool_name):
            return None
        try:
            decision = self._evaluate(tool_name, dict(args or {}))
        except Exception as e:  # fail closed on any bug
            decision = GateDecision(HOLD, ("sra_riskgate_internal_error",),
                                    f"The risk gate hit an internal error ({type(e).__name__}).",
                                    source="fail_closed")
        self._notify(decision)
        return decision

    def release(self, decision: Optional[GateDecision]) -> None:
        """Give back budget reserved by an approval whose tool call did not complete."""
        if decision and decision.approved and decision.intent and decision.intent.is_usdc \
                and decision.intent.amount is not None:
            with self._lock:
                self._spent = max(Decimal(0), self._spent - decision.intent.amount)

    def _notify(self, decision: GateDecision) -> None:
        if self.on_decision is None:
            return
        try:
            self.on_decision(decision)
        except Exception as e:  # an audit hook must not crash the agent
            warnings.warn(f"on_decision callback failed: {e!r}", RuntimeWarning, stacklevel=3)

    def _evaluate(self, tool_name: str, args: dict) -> GateDecision:
        extractor = self.extractors.get(tool_name.lower())
        if extractor is None:
            return GateDecision(HOLD, ("ungated_money_tool",),
                                f"'{tool_name}' looks like it moves funds but has no SRA-RiskGate "
                                "extractor. Add one via extractors=, or list it in passthrough_tools.")
        try:
            intent = extractor(tool_name, args, self.chain_id)
        except Exception as e:
            return GateDecision(REJECT, ("invalid_payment_arguments",),
                                f"The payment arguments could not be read ({e}).")

        def stop(decision: str, flag: str, why: str) -> GateDecision:
            return GateDecision(decision, (flag,), why, intent)

        # x402 one-step requests: no recipient or amount yet.
        if intent.kind == "x402_unpriced":
            host = (urlparse(str(intent.url or "")).hostname or "").lower()
            if host and host in self.x402_hosts:
                return self._finish(intent, args, Decimal(0))
            return stop(HOLD, "x402_amount_unknown",
                        f"Paying {host or 'this URL'} before its price is known is not allowed; "
                        "use the two-step x402 flow or add the host to allowed_x402_hosts.")

        recipient = _addr(intent.recipient)
        if recipient in self.blocked:
            return stop(REJECT, "blocked_recipient", f"Recipient {recipient} is blocked.")
        if not _ADDRESS_RE.match(recipient) or recipient == _ZERO_ADDRESS:
            return stop(REJECT, "invalid_recipient",
                        f"Recipient {intent.recipient!r} is not a valid 0x address.")
        if self.wallet and recipient == self.wallet:
            return stop(REJECT, "self_transfer", "Recipient is the agent's own wallet.")
        if self.allowed is not None and recipient not in self.allowed:
            return stop(HOLD, "recipient_not_allowlisted", f"Recipient {recipient} is not on the allowlist.")

        if not intent.is_usdc:
            if not self.allow_non_usdc:
                return stop(HOLD, "unsupported_asset",
                            f"Asset {intent.asset} is not USDC; only USDC payments are checked "
                            "automatically (set allow_non_usdc=True to let these through).")
            return self._finish(intent, args, Decimal(0), ("non_usdc_amount_unchecked",))

        assert intent.amount is not None
        if intent.amount > self.max_amount:
            return stop(HOLD, "over_payment_limit",
                        f"{intent.amount} USDC is over the {self.max_amount} USDC per-payment limit.")

        if self.peg_feed is not None:
            try:
                dev = abs(float(self.peg_feed(intent.chain_id)))
            except Exception as e:
                return stop(HOLD, "peg_feed_error", f"Could not read the USDC peg ({type(e).__name__}).")
            if not math.isfinite(dev):
                return stop(HOLD, "peg_feed_error", "The USDC peg feed returned an invalid value.")
            if dev >= self.depeg_reject:
                return stop(REJECT, "critical_depeg", f"USDC is {dev}% off its peg.")
            if dev >= self.depeg_hold:
                return stop(HOLD, "moderate_depeg", f"USDC is {dev}% off its peg.")

        return self._finish(intent, args, intent.amount)

    def _finish(self, intent: PaymentIntent, args: dict, amount: Decimal, flags: tuple[str, ...] = ()) -> GateDecision:
        if self.model_check is not None:
            try:
                verdict = dict(self.model_check(intent, args))
                model_decision = verdict.get("decision")
            except Exception as e:
                return GateDecision(HOLD, ("model_unavailable",),
                                    f"The SRA-RiskGate model check failed ({type(e).__name__}).",
                                    intent, "fail_closed")
            if model_decision not in (APPROVE, HOLD, REJECT):
                return GateDecision(HOLD, ("malformed_model_output",),
                                    "The SRA-RiskGate model returned no usable decision.",
                                    intent, "fail_closed", verdict)
            if model_decision != APPROVE:
                model_flags = tuple(str(f) for f in verdict.get("flags") or ()) or ("model_" + model_decision,)
                return GateDecision(model_decision, model_flags,
                                    str(verdict.get("explanation") or "The model did not approve."),
                                    intent, "model", verdict)

        with self._lock:
            if self.session_budget is not None and self._spent + amount > self.session_budget:
                return GateDecision(HOLD, ("session_budget_exceeded",),
                                    f"This would take total approved spend to {self._spent + amount} "
                                    f"USDC, over the {self.session_budget} USDC session budget.", intent)
            self._spent += amount
        return GateDecision(APPROVE, flags, "Passed SRA-RiskGate checks.", intent,
                            "model" if self.model_check else "rules")
