"""PE-Ledger-Solver Harness for Physical-Engineering's Last Exam (PLE).

Combines:
1. Living Engineering Ledger (structured state preventing context amnesia).
2. Offline Engineering Reference & Pint Unit Verification Engine.
3. Full MCP server discovery & tool bridging via taskbridge.MCPClient.
4. 3-Stage Execution: Planner -> Solver/Worker -> Deterministic Gatekeeper + Hostile Senior PE Verifier.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import openai

import eng_db
import ledger as ledger_mod
import taskbridge

MODEL = os.environ["MODEL_NAME"]
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)

WRAP_UP_SEC = 240
WRAP_UP = {
    "role": "user",
    "content": "Time is up for this step. Stop and reply with your answer.",
}
RESULT_CHARS = 20_000
RETRY_WAITS = (5, 15, 30, 60, 120)
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1000
MAX_NUDGES = 3
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
CACHE_MARK = "claude" in MODEL.lower()
PLAN_SHARE = 0.10
CHECK_SHARE = 0.25
MIN_STEP_SEC = 120
MAX_FIX_ROUNDS = 2

client = openai.AsyncOpenAI(max_retries=5, timeout=1800)

PLANNER_PROMPT = """You are a Principal Physical Engineer planning a complex engineering task in an isolated Linux container (workdir: {workdir}, no internet access).
Inspect the task files using `bash` and `read_file` (do NOT create or modify any deliverables yet).
Use `ledger_update` to record the engineering discipline, applicable codes/standards, and exact deliverable paths.
Available MCP servers in this container: {mcp_names}.

Output a concise, actionable engineering plan covering:
1. Every required deliverable with its exact container path and file format.
2. Governing engineering codes, formulas, and hard physical/safety constraints.
3. Whether formal optimization (SciPy MILP / Z3 / NetworkX) or simulation scripts should be written.
4. How the final deliverables will be verified deterministically."""

WORKER_PROMPT = """You are an Expert Licensed Professional Engineer (PE) executing a task in an isolated Linux container (workdir: {workdir}, no internet access).
A project plan and Living Engineering Ledger follow.

CRITICAL RULES:
- Compute ALL physics, loads, geometry, and optimizations with Python scripts (NumPy, SciPy, SymPy, Z3, NetworkX) — NEVER do multi-step engineering math in your head.
- Use `eng_verify_units` to verify dimensional consistency whenever converting or combining physical units.
- Use `eng_lookup` for standard AISC steel shapes, NEC 75C ampacities, and ASCE 7 LRFD load combinations.
- Use `ledger_update` to record verified invariants (with units) and update deliverable statuses as you build them.
- Write every deliverable to its EXACT required path and format, then re-open and test it (e.g., run `solution.py`, validate `.zip`, parse `.ifc`).
- For technical memos/reviews, cite specific standard sections (e.g., ASCE 7-16, AISC 360, IBC, NEC, ASHRAE), state assumptions, and show governing load cases and safety factors.

When finished, reply without tool calls: summarize each deliverable, its exact path, and the key verified numbers."""

VERIFIER_PROMPT = """You are a Hostile Senior PE & City Building Inspector performing a strict QA/QC audit in an isolated Linux container (workdir: {workdir}).
Your ONLY job is to find gaps, errors, unverified assumptions, unit mismatches, code violations, or formatting bugs in the engineer's deliverables. Do NOT modify files.

Audit checklist:
1. Run `audit_deliverables` on all expected deliverable paths to verify existence, non-zero size, Python AST validity, and ZIP integrity.
2. Open and inspect EVERY deliverable via `read_file` or `bash`. Independently recompute key numbers in Python.
3. Check for hard constraint violations, 3D clashes, missing safety factors, or missing standard/code citations.

If and ONLY IF every deliverable is present, valid, and 100% verified, reply with the single word PASS and nothing else.
Otherwise, reply with a numbered list of specific redlines that must be fixed."""


def tool_def(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required if required is not None else list(properties.keys()),
            },
        },
    }


BASE_TOOLS = [
    tool_def("bash", "Run a bash command in the task container (120s limit).", {"command": {"type": "string"}}),
    tool_def("read_file", "Read a text file from the task container.", {"path": {"type": "string"}}),
    tool_def(
        "write_file",
        "Create or overwrite a file in the task container; parent directories are created.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
    ),
    tool_def(
        "eng_verify_units",
        "Check dimensional analysis and convert physical units using pint (e.g. expression='120 * kip / (4 * ft**2)', target_unit='kPa').",
        {"expression": {"type": "string"}, "target_unit": {"type": "string"}},
        required=["expression"],
    ),
    tool_def(
        "eng_lookup",
        "Look up offline engineering reference tables: category in {'steel', 'nec', 'asce7'}, optional key (e.g. 'W14X90', '4/0 AWG').",
        {"category": {"type": "string"}, "key": {"type": "string"}},
        required=["category"],
    ),
    tool_def(
        "ledger_update",
        "Update the Living Engineering Ledger with discipline, codes, verified invariants, or deliverable statuses.",
        {
            "discipline": {"type": "string"},
            "add_codes": {"type": "array", "items": {"type": "string"}},
            "set_invariants": {"type": "object", "additionalProperties": {"type": "string"}},
            "set_deliverables": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        required=[],
    ),
    tool_def(
        "audit_deliverables",
        "Run deterministic syntax, ZIP integrity, and non-empty file checks on deliverable paths inside the task container.",
        {"paths": {"type": "array", "items": {"type": "string"}}},
        required=["paths"],
    ),
]


def clip(text: str) -> str:
    if len(text) <= RESULT_CHARS:
        return text
    half = RESULT_CHARS // 2
    return f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut ...]\n{text[-half:]}"


def marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not CACHE_MARK:
        return messages
    out = list(messages)
    for i in {0, len(out) - 1}:
        m, content = out[i], out[i].get("content")
        if isinstance(content, str) and content:
            out[i] = {
                **m,
                "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
            }
        else:
            out[i] = {**m, "cache_control": {"type": "ephemeral"}}
    return out


async def audit_paths_in_container(session: taskbridge.Session, paths: list[str]) -> str:
    if not paths:
        return json.dumps({"ok": False, "error": "No deliverable paths provided"})
    py_script = (
        "import ast, json, os, zipfile\n"
        f"paths = {json.dumps(paths)}\n"
        "results = {}\n"
        "for p in paths:\n"
        "  if not os.path.exists(p):\n"
        "    results[p] = {'exists': False}\n"
        "    continue\n"
        "  if os.path.isdir(p):\n"
        "    files = [os.path.join(dp, f) for dp, _, fn in os.walk(p) for f in fn]\n"
        "    results[p] = {'exists': True, 'type': 'dir', 'file_count': len(files)}\n"
        "    continue\n"
        "  size = os.path.getsize(p)\n"
        "  info = {'exists': True, 'type': 'file', 'size_bytes': size}\n"
        "  if size == 0:\n"
        "    info['error'] = 'empty file'\n"
        "  elif p.endswith('.py'):\n"
        "    try:\n"
        "      with open(p, 'r', encoding='utf-8') as f: ast.parse(f.read())\n"
        "      info['ast_valid'] = True\n"
        "    except Exception as e:\n"
        "      info['ast_valid'] = False\n"
        "      info['error'] = str(e)\n"
        "  elif p.endswith('.zip'):\n"
        "    try:\n"
        "      with zipfile.ZipFile(p, 'r') as zf:\n"
        "        bad = zf.testzip()\n"
        "        info['zip_valid'] = bad is None\n"
        "        info['zip_members'] = zf.namelist()[:50]\n"
        "    except Exception as e:\n"
        "      info['zip_valid'] = False\n"
        "      info['error'] = str(e)\n"
        "  results[p] = info\n"
        "print(json.dumps(results, indent=2))\n"
    )
    res = await session.exec(f"python3 -c {json.dumps(py_script)}")
    return clip(res["output"])


async def discover_mcp_tools(session: taskbridge.Session) -> tuple[list[dict[str, Any]], dict[str, tuple[taskbridge.MCPClient, str]]]:
    mcp_tool_defs: list[dict[str, Any]] = []
    mcp_dispatch: dict[str, tuple[taskbridge.MCPClient, str]] = {}
    for spec in session.mcp_servers:
        srv_name = spec.get("name", "mcp")
        url = spec.get("url")
        if not url:
            continue
        try:
            mcp = taskbridge.MCPClient(session, url)
            await mcp.initialize()
            tools = await mcp.list_tools()
            for t in tools:
                t_name = t["name"]
                safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp__{srv_name}__{t_name}")[:64]
                mcp_tool_defs.append({
                    "type": "function",
                    "function": {
                        "name": safe_name,
                        "description": f"[MCP:{srv_name}] {t.get('description', '')}",
                        "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                    },
                })
                mcp_dispatch[safe_name] = (mcp, t_name)
        except Exception as exc:
            await session.log(f"MCP init failed for {srv_name}: {exc}")
    return mcp_tool_defs, mcp_dispatch


async def run_tool(
    session: taskbridge.Session,
    eng_ledger: ledger_mod.EngineeringLedger,
    mcp_dispatch: dict[str, tuple[taskbridge.MCPClient, str]],
    fn: dict[str, Any],
) -> str:
    name = fn["name"]
    try:
        args = json.loads(fn["arguments"] or "{}")
        if not isinstance(args, dict):
            raise TypeError("not a JSON object")
        fn["arguments"] = json.dumps(args, allow_nan=False)
    except (TypeError, ValueError) as exc:
        fn["arguments"] = "{}"
        return f"error: invalid JSON arguments ({exc})"

    try:
        if name == "bash":
            res = await session.exec(args["command"])
            return clip(f"exit code {res['exit_code']}\n{res['output']}")
        if name == "read_file":
            data = await session.read_file(args["path"])
            return clip(data.decode("utf-8", "replace"))
        if name == "write_file":
            n = await session.write_file(args["path"], args["content"])
            return f"wrote {n} bytes to {args['path']}"
        if name == "eng_verify_units":
            return json.dumps(eng_db.verify_units(args["expression"], args.get("target_unit")), indent=2)
        if name == "eng_lookup":
            return eng_db.lookup(args["category"], args.get("key", ""))
        if name == "ledger_update":
            return eng_ledger.update(
                discipline=args.get("discipline"),
                add_codes=args.get("add_codes"),
                set_invariants=args.get("set_invariants"),
                set_deliverables=args.get("set_deliverables"),
            )
        if name == "audit_deliverables":
            return await audit_paths_in_container(session, args.get("paths", []))
        if name in mcp_dispatch:
            mcp_client, orig_name = mcp_dispatch[name]
            out = await mcp_client.call_tool(orig_name, args)
            return clip(json.dumps(out, default=str))
        return f"error: unknown tool {name}"
    except (taskbridge.TaskError, KeyError, ValueError, Exception) as exc:
        return f"error: {type(exc).__name__}: {exc}"


async def shorten(session: taskbridge.Session, messages: list[dict[str, Any]]) -> bool:
    long_idxs = [
        i for i, m in enumerate(messages)
        if m["role"] == "tool" and len(m.get("content", "")) > 2 * CUT_KEEP
    ]
    if not long_idxs:
        return False
    cut = long_idxs[: max(1, len(long_idxs) // 2)]
    for i in cut:
        c = messages[i]["content"]
        messages[i] = {**messages[i], "content": f"{c[:CUT_KEEP]}\n[... truncated for context limit ...]"}
    await session.log(f"compacted {len(cut)} older tool results")
    return True


async def recover(session: taskbridge.Session, messages: list[dict[str, Any]], exc: openai.APIError, waits: Any, until: float) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 400 and CONTEXT_FULL.search(str(exc)):
        return await shorten(session, messages)
    wait = next(waits, None)
    if wait is None or not (status is None or status in (408, 409, 429) or status >= 500):
        return False
    left = session.time_left()
    if left is not None and left > until:
        wait = min(wait, left - until)
    await session.log(f"model call error ({str(exc)[:180]}); retrying in {wait:.0f}s")
    await asyncio.sleep(wait)
    return True


async def chat_step(
    session: taskbridge.Session,
    eng_ledger: ledger_mod.EngineeringLedger,
    mcp_dispatch: dict[str, tuple[taskbridge.MCPClient, str]],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    until: float = 0.0,
) -> str:
    nudges = 0
    while True:
        left = session.time_left()
        wrapping = left is not None and left <= until
        if wrapping:
            messages.append(WRAP_UP)

        waits = iter(RETRY_WAITS)
        while True:
            try:
                kwargs: dict[str, Any] = {
                    "model": MODEL,
                    "messages": marked(messages),
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "reasoning_effort": EFFORT,
                }
                if tools and not wrapping:
                    kwargs["tools"] = tools
                resp = await client.chat.completions.create(**kwargs)
                break
            except openai.APIError as exc:
                if not await recover(session, messages, exc, waits, until):
                    raise

        if resp.usage:
            details = resp.usage.prompt_tokens_details
            await session.usage(
                MODEL,
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
                getattr(details, "cached_tokens", 0) or 0,
            )

        choice = resp.choices[0]
        msg = choice.message
        assistant_msg = msg.model_dump(exclude_none=True)
        messages.append(assistant_msg)
        text = msg.content or ""

        if wrapping or (choice.finish_reason != "length" and not msg.tool_calls):
            if text or nudges >= MAX_NUDGES:
                return text
            nudges += 1
            messages.append({"role": "user", "content": "Continue and provide your final summary."})
            continue

        nudges = 0
        if msg.tool_calls:
            for idx, tc in enumerate(msg.tool_calls):
                fn = {"name": tc.function.name, "arguments": tc.function.arguments}
                result = await run_tool(session, eng_ledger, mcp_dispatch, fn)
                if "tool_calls" in assistant_msg and idx < len(assistant_msg["tool_calls"]):
                    assistant_msg["tool_calls"][idx]["function"]["arguments"] = fn["arguments"]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        elif choice.finish_reason == "length":
            messages.append({"role": "user", "content": "Continue from where you stopped."})


async def solve(session: taskbridge.Session) -> str:
    eng_ledger = ledger_mod.EngineeringLedger()
    mcp_tool_defs, mcp_dispatch = await discover_mcp_tools(session)
    all_tools = BASE_TOOLS + mcp_tool_defs
    read_only_tools = [t for t in all_tools if t["function"]["name"] != "write_file"]

    total_sec = session.time_left() or 3600.0
    plan_until = max(0.0, total_sec * (1.0 - PLAN_SHARE))
    check_until = max(WRAP_UP_SEC, total_sec * CHECK_SHARE)
    mcp_names = ", ".join(s.get("name", "?") for s in session.mcp_servers) or "none"

    await session.log(f"starting PE-Ledger-Solver on {MODEL} ({len(mcp_tool_defs)} MCP tools)")

    # 1. Planner
    plan_msgs = [
        {"role": "system", "content": PLANNER_PROMPT.format(workdir=session.workdir, mcp_names=mcp_names)},
        {"role": "user", "content": session.instruction},
    ]
    plan = await chat_step(session, eng_ledger, mcp_dispatch, plan_msgs, read_only_tools, until=plan_until)
    await session.log("planning complete")

    # 2. Worker / Solver
    worker_msgs = [
        {"role": "system", "content": WORKER_PROMPT.format(workdir=session.workdir)},
        {
            "role": "user",
            "content": (
                f"TASK:\n{session.instruction}\n\n"
                f"ENGINEERING PLAN:\n{plan}\n\n"
                f"{eng_ledger.render_summary()}"
            ),
        },
    ]
    summary = await chat_step(session, eng_ledger, mcp_dispatch, worker_msgs, all_tools, until=check_until)
    session.final_message = summary

    # 3. Hostile Senior PE Verifier + Fix Loop
    for round_idx in range(MAX_FIX_ROUNDS):
        left = session.time_left()
        if left is not None and left < WRAP_UP_SEC + MIN_STEP_SEC:
            break
        verifier_msgs = [
            {"role": "system", "content": VERIFIER_PROMPT.format(workdir=session.workdir)},
            {
                "role": "user",
                "content": (
                    f"TASK:\n{session.instruction}\n\n"
                    f"ENGINEERING PLAN:\n{plan}\n\n"
                    f"{eng_ledger.render_summary()}\n\n"
                    f"ENGINEER SUMMARY:\n{summary}"
                ),
            },
        ]
        verdict = await chat_step(session, eng_ledger, mcp_dispatch, verifier_msgs, read_only_tools, until=WRAP_UP_SEC)
        if verdict.strip().upper().startswith("PASS"):
            await session.log(f"PE verifier PASSED on round {round_idx + 1}")
            break
        eng_ledger.redlines = [verdict.strip()]
        await session.log(f"PE verifier found gaps (round {round_idx + 1}); sending to worker")
        worker_msgs.append({
            "role": "user",
            "content": (
                f"HOSTILE SENIOR PE REVIEW REJECTED THE DELIVERABLES:\n{verdict}\n\n"
                f"{eng_ledger.render_summary()}\n\n"
                "Fix every redline, re-verify in bash, and reply with your updated summary."
            ),
        })
        summary = await chat_step(session, eng_ledger, mcp_dispatch, worker_msgs, all_tools, until=WRAP_UP_SEC)
        session.final_message = summary

    return summary


if __name__ == "__main__":
    taskbridge.run(solve)
