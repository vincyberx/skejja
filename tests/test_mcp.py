"""Test del server MCP con il client in memoria dell'SDK: elenco degli strumenti e chiamate end to end."""
import pytest

pytest.importorskip("mcp")
import anyio  # noqa: E402
from mcp import Client  # noqa: E402

from skejja.mcp_server import create_mcp  # noqa: E402
from skejja.systemone import SystemOne  # noqa: E402
from test_systemone import EXAMPLES, FakeModel  # noqa: E402


def call(server, name, args):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(name, args)
    return anyio.run(go)


def test_tools_listed():
    async def go():
        async with Client(create_mcp(SystemOne(FakeModel()))) as client:
            return await client.list_tools()
    names = {t.name for t in anyio.run(go).tools}
    assert names == {"choose", "yes_no", "systemone", "calibrate_question"}


def test_choose_and_yes_no():
    server = create_mcp(SystemOne(FakeModel()))
    r = call(server, "choose", {"text": "turn on the lights", "options": {"on": "turn on the lights",
                                                                         "music": "play music"}})
    assert not r.is_error and r.structured_content["choice"] == "on"
    r = call(server, "yes_no", {"text": "I want a refund", "question": "Does the user want a refund?"})
    assert not r.is_error
    assert r.structured_content["yes"] > 0.5 and r.structured_content["calibration"] == "global"


def test_calibration_then_yes_no_uses_it():
    server = create_mcp(SystemOne(FakeModel()))
    r = call(server, "calibrate_question", {"question": "Does the user want a refund?", "examples": EXAMPLES})
    assert not r.is_error and r.structured_content["examples"] == 12
    r = call(server, "yes_no", {"text": "refund", "question": "Does the user want a refund?"})
    assert r.structured_content["calibration"] == "question"


def test_systemone_and_errors():
    server = create_mcp(SystemOne(FakeModel()))
    r = call(server, "systemone", {"state": {"msg": "turn on the lights"}, "questions": {
        "a": {"type": "choice", "criteria": {"on": "turn on the lights", "off": "music"}},
        "b": {"type": "score", "criteria": ["low", "high"]}}})
    assert not r.is_error and set(r.structured_content["answers"]) == {"a", "b"}
    r = call(server, "systemone", {"state": "s", "questions": {"q": {"type": "rank"}}})
    assert r.is_error
