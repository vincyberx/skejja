"""Server MCP di Skejja: le stesse decisioni dell'API /v1/systemone, come strumenti per agenti (Claude e altri).

Richiede l'extra `mcp` (`pip install "skejja[mcp]"`, SDK MCP 2.x). Si avvia in stdio con `skejja mcp <bundle>`:
il client lo lancia come sottoprocesso, quindi non apre porte di rete.

Strumenti:
- `choose`: una scelta fra opzioni (Choice), la forma più semplice per un agente;
- `yes_no`: la probabilità di un sì (Noul), con la calibrazione della domanda se esiste;
- `systemone`: una richiesta completa nel formato di /v1/systemone (più domande sullo stesso stato);
- `calibrate_question`: calibrazione per domanda di una noul con esempi etichettati.
"""
from typing import Any

from mcp.server import MCPServer

from .systemone import SystemOne

INSTRUCTIONS = (
    "Skejja is a small local decision model: it does not write text, it returns calibrated probabilities. "
    "Use it for fast routing and checks on short user messages (commands, intents, yes/no about what the user "
    "wants), in English, Italian, Spanish, German or French. Write option descriptions in English. It is weak at "
    "reasoning, long documents and many options: do not use it for those."
)


def create_mcp(engine: SystemOne) -> MCPServer:
    server = MCPServer("skejja", instructions=INSTRUCTIONS)

    @server.tool()
    def choose(text: str, options: dict[str, str], instructions: str | None = None) -> dict[str, Any]:
        """Pick one option for a short text. `options` maps option names to short English descriptions,
        e.g. {"lights_on": "turn on the lights", "music": "play music"}. Returns the chosen name, the
        probability of every option and a confidence between 0 and 1."""
        q = {"type": "choice", "criteria": options}
        if instructions:
            q["instructions"] = instructions
        out = engine.run({"state": text, "questions": {"q": q}})
        return {**out["answers"]["q"], "truncated": bool(out["skejja"]["truncated"])}

    @server.tool()
    def yes_no(text: str, question: str) -> dict[str, Any]:
        """Probability that the answer to a yes/no question about a short text is yes, e.g. question
        "Does the user want to turn on the lights?". `calibration` is "question" when the question was
        calibrated with examples, otherwise "global" (less reliable on new questions)."""
        out = engine.run({"state": text, "questions": {"q": {"type": "noul", "instructions": question}}})
        return {"yes": out["answers"]["q"]["noul"], "calibration": out["skejja"]["calibration"]["q"],
                "truncated": bool(out["skejja"]["truncated"])}

    @server.tool()
    def systemone(state: str | dict[str, Any] | list[Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Several typed questions about the same state, in the /v1/systemone format: questions maps an id to
        {"type": "choice"|"noul"|"score", "instructions": ..., "criteria": ...}. Choice criteria: {name:
        description}; noul criteria (optional): {"true": ..., "false": ...}; score criteria: list of levels from
        lowest to highest."""
        return engine.run({"state": state, "questions": questions})

    @server.tool()
    def calibrate_question(question: str, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Calibrate a yes/no question with labelled examples [{"state": "...", "answer": true/false}, ...]
        (at least 8, best 32-64, both answers present). Afterwards yes_no uses this calibration."""
        return engine.calibrate({"question": {"type": "noul", "instructions": question}, "examples": examples})

    return server
