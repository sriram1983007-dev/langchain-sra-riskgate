# langchain-sra-riskgate

[![Tests](https://github.com/sriram1983007-dev/langchain-sra-riskgate/actions/workflows/tests.yml/badge.svg)](https://github.com/sriram1983007-dev/langchain-sra-riskgate/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-sra-riskgate)](https://pypi.org/project/langchain-sra-riskgate/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

LangChain middleware that checks every payment an AI agent tries to make **before the tool runs**.
Each payment gets a verdict of `approve`, `hold` or `reject`. A held or rejected payment never reaches the wallet.

It is built for [Coinbase AgentKit](https://github.com/coinbase/agentkit) agents. AgentKit's own
[Managing Risk](https://github.com/coinbase/agentkit#%EF%B8%8F-managing-risk) section warns that injected text can trigger an on-chain
transfer, and that AgentKit itself does not enforce spend caps or allowlist destinations. This package
adds those checks outside the model, so a hijacked agent cannot talk its way past them.

```bash
pip install "langchain-sra-riskgate[middleware]"   # LangChain 1.x
pip install langchain-sra-riskgate                 # LangChain 0.3, e.g. with coinbase-agentkit-langchain
```

## What it checks

It recognises these AgentKit tools automatically:

| AgentKit tool | Treated as |
|---|---|
| `ERC20ActionProvider_transfer` | Token payment to `destination_address` |
| `ERC20ActionProvider_approve` | Payment, because an allowance lets the spender take the funds |
| `WalletActionProvider_native_transfer` | Native-token payment |
| `x402ActionProvider_retry_with_x402` | x402 payment to `pay_to` |
| `x402ActionProvider_make_http_request_with_x402` | x402 payment whose price is not known yet |

Each payment is then checked against these rules:

| Check | Result |
|---|---|
| Recipient, or your own `wallet_address`, on the OFAC SDN list (exact match against the dated snapshot bundled with `sra-riskgate`; turn off with `check_sanctions=False`) | reject |
| Recipient on `blocked_recipients` | reject |
| Recipient not a valid `0x` address, zero address, or the agent's own wallet | reject |
| Recipient not on `allowed_recipients` (when you set one) | hold |
| Asset is not USDC (native ETH and other tokens) | hold, unless `allow_non_usdc=True` |
| USDC amount below 0.000001 (rounds to zero on-chain) | reject |
| Amount over `max_amount_usdc` (default 100) | hold |
| Total approved spend over `session_budget_usdc` (stops split payments) | hold |
| USDC off its peg (with a `peg_feed`) by ≥1% / ≥5% | hold / reject |
| One-step x402 request to a host not on `allowed_x402_hosts` | hold |
| Arguments claim their own verification ("risk level: low (verified offline)", "sanctions screening passed", "ignore previous instructions") | hold |
| Unreadable arguments | reject |
| A tool whose name contains a word such as transfer, swap, pay or withdraw, with no extractor (strict mode) | hold |
| Any internal error, feed error or model error | hold |

Tools that don't move money (balances, searches, plain HTTP requests) pass through untouched.

The gate fails closed. No error path can produce `approve`.

## LangChain 1.x: middleware

```python
from langchain.agents import create_agent
from langchain_sra_riskgate import SRARiskGateMiddleware

agent = create_agent(
    model=llm,
    tools=tools,  # your AgentKit tools as LangChain tools
    middleware=[
        SRARiskGateMiddleware(
            chain_id=8453,                    # Base mainnet
            max_amount_usdc=50,
            session_budget_usdc=200,
            allowed_recipients=["0x1234..."],
            wallet_address=wallet.get_address(),
        ),
    ],
)
```

When a call is stopped, the agent receives an error `ToolMessage` like this:

```
SRA-RiskGate HELD for human review: this payment was NOT sent. 500 USDC is over the 50 USDC
per-payment limit. Flags: over_payment_limit. Do not retry this payment with changed details or
split it into smaller payments; tell the user it needs their review.
```

The verdict and flags are also attached as `artifact={"sra_riskgate": {...}}`.

You can also add LangChain's `HumanInTheLoopMiddleware` so a person must approve payment tools. The gate still
runs after the person approves.

## LangChain 0.3 and coinbase-agentkit-langchain: `guard_tools`

`coinbase-agentkit-langchain` currently requires LangChain 0.3, which has no middleware.
For that setup, wrap the tools themselves:

```python
from coinbase_agentkit import AgentKit, AgentKitConfig, erc20_action_provider, wallet_action_provider
from coinbase_agentkit_langchain import get_langchain_tools
from langchain_sra_riskgate import guard_tools

agentkit = AgentKit(AgentKitConfig(wallet_provider=wallet,
                                   action_providers=[wallet_action_provider(), erc20_action_provider()]))
tools = guard_tools(get_langchain_tools(agentkit), chain_id=8453, max_amount_usdc=50, session_budget_usdc=200)
# use `tools` with create_react_agent or any agent that takes LangChain tools
```

Only the money-moving tools are wrapped. Names, descriptions and argument schemas are unchanged.

## Your own payment tools

Tell the gate how to read a custom tool's arguments:

```python
from decimal import Decimal
from langchain_sra_riskgate import PaymentGate, PaymentIntent

def pay_vendor(tool, args, chain_id):
    return PaymentIntent(tool, "custom", args["wallet"], Decimal(str(args["usdc"])), "USDC", chain_id)

gate = PaymentGate(extractors={"pay_vendor": pay_vendor}, max_amount_usdc=25)
middleware = SRARiskGateMiddleware(gate)
```

If a tool's name looks like it moves money but the gate has no extractor for it, strict mode holds it by default.
Add the tool to `passthrough_tools` if it is safe.

Use `on_decision=` to log every verdict to an audit trail.

## Optional: a second check by SRA-RiskGate-4B

After the rules approve a payment, the gate can also ask
[SRA-RiskGate-4B](https://huggingface.co/sriram1983007/SRA-RiskGate-4B), served through any OpenAI-compatible
endpoint (Ollama by default).

```bash
ollama pull sriram1983007/sra-riskgate
```

```python
from langchain_sra_riskgate import OpenAICompatibleModelCheck, PaymentGate

def build_input(intent, args):
    policy = {"max_amount_usdc": 50, "allowed_assets": ["USDC"]}
    payload = {"to": intent.recipient, "amount_usdc": str(intent.amount), "chain_id": intent.chain_id}
    tool_results = {"sanctions_screening": my_screening(intent.recipient)}  # your real checks
    return policy, payload, tool_results

gate = PaymentGate(max_amount_usdc=50, model_check=OpenAICompatibleModelCheck(build_input))
```

The model was trained to reason over a payment **together with the results of real verification tools**:
decoded payment, signature check, risk attestations and sanctions screening. It does not run those checks itself,
and this package does not invent their results. You supply them in `build_input`.

The model can only make a decision stricter. It is asked only about payments the rules already approved.
If it answers anything other than `approve`, or is unreachable or off-schema, the payment is not sent.

On its held-out benchmark, SRA-RiskGate-4B reached 99.5% decision accuracy, with a 0.47% unsafe-approval rate
([predictions and scores](https://huggingface.co/datasets/sriram1983007/sra-bench-results)).
That benchmark uses synthetic data. Measure it on your own traffic before relying on it.

## Limitations

- **Coverage.** Only tools the gate recognises are checked. If you register other fund-moving tools, add
  extractors for them or keep strict mode on.
- **Recipient addresses.** Only `0x` addresses are accepted. ENS names are rejected rather than resolved.
- **USDC limits only.** Amount limits and the budget apply to USDC. Native and other tokens are held
  unless you opt in with `allow_non_usdc=True`, and then their amounts are not limited.
- **One-step x402 requests.** Their price is unknown when the gate runs, so they are allowed only for
  hosts you list. AgentKit's own `max_payment_usdc` still applies to them.
- **Budget accounting.** Spend is counted when a payment is approved. It is released only if the tool raises.
- **Scope.** This is a guardrail, not compliance. The OFAC match is an exact comparison with a dated snapshot
  (pass `sanctions_list=` for a fresher one); it does not replace sanctions screening or KYC.

## Related

- [`sra-riskgate`](https://github.com/sriram1983007-dev/sra-riskgate): core SDK, an AgentKit action provider and x402 hooks
- [`sra-riskgate-mcp`](https://github.com/sriram1983007-dev/sra-riskgate-mcp): the same checks as an MCP server
- [SRA-RiskGate-4B](https://huggingface.co/sriram1983007/SRA-RiskGate-4B) and its
  [benchmark](https://huggingface.co/datasets/sriram1983007/sra-stablecoin-risk-bench)

Apache-2.0.
