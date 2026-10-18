# Changelog

## 0.3.0
- **OFAC SDN address match:** payments to a recipient, or from your `wallet_address`, on the OFAC SDN list
  are **rejected** (flags `ofac_sdn_match`, `sanctioned_recipient` / `sanctioned_sender`). It is an exact,
  offline match against the dated snapshot bundled with sra-riskgate 0.7.1+, not a screening service.
  Turn it off with `check_sanctions=False`, or pass `sanctions_list=` (a `sra_riskgate.SanctionsList`) for a
  fresher list. If the list cannot be loaded, the payment is held.
- USDC amounts below one atomic unit (0.000001) are **rejected** (`amount_below_smallest_unit`); they round
  to zero on-chain and can never execute. Before, such a payment was approved and failed later.
- Security: `OpenAICompatibleModelCheck` refuses a `base_url` / `SRA_BASE_URL` that is not `http://` or
  `https://` (raises `ValueError` at construction).
- Requires sra-riskgate 0.7.1 or newer.

## 0.2.1
- For x402 calls, only the seller's payment option (description, extra, resource...) is scanned for
  self-asserted verification claims, not the agent's own request URL, headers, query or body. A paid API
  call that asks "has sanctions screening passed for wallet X?" is no longer held by mistake.
- Picks up the hardened claims scanner from sra-riskgate 0.5.1.

## 0.2.0
- Payment tool calls whose arguments claim their own verification ("risk level: low (verified offline)",
  "sanctions screening passed", "ignore previous instructions" ...) are now held with the flag
  `self_asserted_verification_in_payload`. Disable with `check_payload_claims=False`.
  Uses `scan_payload_claims` from sra-riskgate 0.5.0; see its changelog for the measurement and caveats.
- Requires sra-riskgate 0.5.0 or newer.

## 0.1.0

- First release.
- `SRARiskGateMiddleware` for LangChain 1.x agents (`wrap_tool_call`, sync and async).
- `guard_tools()` for LangChain 0.3, including `coinbase-agentkit-langchain`.
- Built-in recognition of Coinbase AgentKit's ERC-20 transfer and approve, native transfer, and x402 payment tools
  (AgentKit 0.7.x flat x402 fields and the newer nested `selected_payment_option`).
- Rules: blocked recipients, recipient allowlist, address validity, self-transfers, USDC-only by default,
  per-payment limit, session budget, USDC depeg, x402 host allowlist, and strict mode for unrecognised money tools.
- Optional second check by SRA-RiskGate-4B through any OpenAI-compatible server. It fails closed.
