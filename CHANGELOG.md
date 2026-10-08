# Changelog

## 0.1.0

- First release.
- `SRARiskGateMiddleware` for LangChain 1.x agents (`wrap_tool_call`, sync and async).
- `guard_tools()` for LangChain 0.3, including `coinbase-agentkit-langchain`.
- Built-in recognition of Coinbase AgentKit's ERC-20 transfer and approve, native transfer, and x402 payment tools
  (AgentKit 0.7.x flat x402 fields and the newer nested `selected_payment_option`).
- Rules: blocked recipients, recipient allowlist, address validity, self-transfers, USDC-only by default,
  per-payment limit, session budget, USDC depeg, x402 host allowlist, and strict mode for unrecognised money tools.
- Optional second check by SRA-RiskGate-4B through any OpenAI-compatible server. It fails closed.
