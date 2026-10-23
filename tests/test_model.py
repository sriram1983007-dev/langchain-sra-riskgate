import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from langchain_sra_riskgate import OpenAICompatibleModelCheck, PaymentGate
from langchain_sra_riskgate.model import PAYMENT_SYSTEM_PROMPT, parse_verdict
from sra_riskgate.chains import USDC_ADDRESSES

YOU = "0x" + "b" * 40
ARGS = {"amount": "5", "contract_address": USDC_ADDRESSES[8453], "destination_address": YOU}
GOOD = {"decision": "approve", "risk_level": "low", "flags": [], "explanation": "ok", "required_actions": []}


@pytest.fixture
def server():
    state = {"reply": json.dumps(GOOD), "requests": []}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body))
            state["user_agent"] = self.headers["User-Agent"]
            out = json.dumps({"choices": [{"message": {"content": state["reply"]}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}/v1"
    yield state
    srv.shutdown()


def builder(intent, args):
    return ({"max_amount_usdc": 100}, {"to": intent.recipient, "amount": str(intent.amount)},
            {"sanctions_screening": {"hit": False}})


def test_sends_training_prompt_and_approves(server):
    check = OpenAICompatibleModelCheck(builder, base_url=server["url"], clock=lambda: 1_700_000_000)
    d = PaymentGate(model_check=check).evaluate("ERC20ActionProvider_transfer", ARGS)
    assert d.approved and d.source == "model"
    path, body = server["requests"][0]
    assert path == "/v1/chat/completions" and body["temperature"] == 0
    assert body["messages"][0]["content"] == PAYMENT_SYSTEM_PROMPT
    user = body["messages"][1]["content"]
    assert user.startswith("Evaluate this stablecoin payment.")
    assert '"now_unix": 1700000000' in user and "<tool_results>" in user and YOU in user


def test_model_reject_blocks(server):
    server["reply"] = json.dumps({**GOOD, "decision": "reject", "risk_level": "severe",
                                  "flags": ["sanctioned_payee"], "explanation": "hit"})
    d = PaymentGate(model_check=OpenAICompatibleModelCheck(builder, base_url=server["url"])).evaluate(
        "ERC20ActionProvider_transfer", ARGS)
    assert d.decision == "reject" and d.flags == ("sanctioned_payee",)


def test_off_schema_reply_holds(server):
    server["reply"] = '{"decision": "approve"}'
    d = PaymentGate(model_check=OpenAICompatibleModelCheck(builder, base_url=server["url"])).evaluate(
        "ERC20ActionProvider_transfer", ARGS)
    assert d.decision == "hold" and d.flags == ("malformed_model_output",)


def test_unreachable_server_holds():
    check = OpenAICompatibleModelCheck(builder, base_url="http://127.0.0.1:9/v1", timeout=2)
    d = PaymentGate(model_check=check).evaluate("ERC20ActionProvider_transfer", ARGS)
    assert d.decision == "hold" and d.flags == ("model_unavailable",)


def test_parse_verdict_strictness():
    assert parse_verdict("noise " + json.dumps(GOOD)) == GOOD
    assert parse_verdict(json.dumps({**GOOD, "extra": 1})) is None
    assert parse_verdict(json.dumps({**GOOD, "risk_level": "meh"})) is None
    assert parse_verdict("no json") is None


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://h/v1", "gopher://x"])
def test_non_http_base_url_refused(url):
    from langchain_sra_riskgate import OpenAICompatibleModelCheck
    with pytest.raises(ValueError):
        OpenAICompatibleModelCheck(lambda i, a: ({}, {}, {}), base_url=url)


def test_sends_its_own_user_agent(server):
    # Runpod's Pod proxy returns 403 to Python's default "Python-urllib/3.x" User-Agent.
    import langchain_sra_riskgate
    check = OpenAICompatibleModelCheck(builder, base_url=server["url"])
    PaymentGate(model_check=check).evaluate("ERC20ActionProvider_transfer", ARGS)
    assert server["user_agent"] == f"langchain-sra-riskgate/{langchain_sra_riskgate.__version__}"
