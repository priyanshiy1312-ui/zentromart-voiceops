"""ZentroMart support agent.

* PromptOnlyAgent  - Step 1 baseline: system prompt only, no documents, no tools.
* SupportAgent     - Steps 2-3: RAG over the knowledge base + multi-turn memory + tools
                     (lookup_order, create_ticket, ...) with guards so the agent can never
                     claim an action its code did not perform.
"""
import json
import re
from dataclasses import dataclass, field

from . import config, prompts
from .kb import KnowledgeBase
from .llm import LLM
from .tools_adapter import ToolBox, mask_args


@dataclass
class TurnResult:
    reply: str
    decision: str = "answer"
    intent: str = ""
    sources: list = field(default_factory=list)       # chunk ids the model says it used
    chunks: list = field(default_factory=list)        # chunks retrieved for this turn
    tool_calls: list = field(default_factory=list)    # [{action, args, result, ok}]


_DECISION_ALIASES = {"handover": "escalate", "hand_over": "escalate", "escalation": "escalate",
                     "unknown": "answer", "decline": "refuse", "refused": "refuse", "clarification": "clarify"}


def norm_decision(value):
    v = str(value or "answer").strip().lower()
    return _DECISION_ALIASES.get(v, v)


def parse_json(raw):
    """Parse the model's JSON action. Never returns raw JSON text as a spoken reply."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    if raw.startswith("{") or '"action"' in raw:  # truncated / malformed JSON: salvage what we can
        out = {"_parse_failed": True}
        am = re.search(r'"action"\s*:\s*"(\w+)"', raw)
        rm = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if am:
            out["action"] = am.group(1)
        if rm and out.get("action", "reply") == "reply":
            out["reply"] = rm.group(1).encode().decode("unicode_escape", "ignore") if "\\" in rm.group(1) else rm.group(1)
            out.pop("_parse_failed")
        return out
    return {"action": "reply", "reply": raw, "decision": "answer"}


class PromptOnlyAgent:
    """Step 1: no documents, no tools."""

    def __init__(self, llm=None):
        self.llm = llm or LLM()
        self.history = []

    def reset(self):
        self.history = []

    def respond_full(self, text):
        self.history.append({"role": "user", "content": text})
        reply = self.llm.chat(prompts.STEP1_SYSTEM, self.history[-12:], max_tokens=300).strip()
        self.history.append({"role": "assistant", "content": reply})
        return TurnResult(reply=reply, decision="answer")

    def respond(self, text):
        return self.respond_full(text).reply


class SupportAgent:
    def __init__(self, llm=None, kb=None, toolbox=None, top_k=None, max_steps=5):
        self.llm = llm or LLM()
        self.kb = kb or KnowledgeBase()
        self.tools = toolbox or ToolBox()
        self.top_k = top_k or config.TOP_K
        self.max_steps = max_steps
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.history = []  # alternating user / assistant text (no JSON, no chunks)
        self.state = {"order_id": None, "order": None, "ticket": None, "turns": 0, "verify_failures": 0}

    # -------------------------------------------------------------- retrieval
    def _retrieval_query(self, text):
        if not self.history:
            return text
        order = self.state.get("order") or {}
        facts = json.dumps(order, ensure_ascii=False, default=str)[:600] if order else "none"
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-6:])
        try:
            q = self.llm.chat(prompts.REWRITE_SYSTEM,
                              [{"role": "user", "content": f"Conversation:\n{convo}\n\nKnown order facts: {facts}\n\n"
                                                           f"Latest customer message: {text}"}],
                              max_tokens=80).strip()
            return q or text
        except Exception:
            return f"{self.history[-2]['content']} {text}" if len(self.history) >= 2 else text

    def _merge(self, base, extra, limit=8):
        seen = {c["id"] for c in base}
        return (base + [c for c in extra if c["id"] not in seen])[:limit]

    # ---------------------------------------------------------------- prompts
    def _compose(self, text, chunks, tool_log):
        docs = "\n\n".join(
            f"[{c['id']}] source={c['source']} status={c['status']} effective_from={c['effective_from'] or 'n/a'}\n{c['text']}"
            for c in chunks) or "(none retrieved)"
        st = {"order_id": self.state["order_id"], "verified": bool(self.state["order"]),
              "verify_failures": self.state["verify_failures"],
              "order_data": self.state["order"], "ticket": self.state["ticket"]}
        parts = [
            f"TODAY: {config.TODAY.isoformat()} ({config.TODAY.strftime('%A')})",
            f"FIRST_TURN: {'yes' if self.state['turns'] == 0 else 'no'}",
            "CONVERSATION_STATE: " + json.dumps(st, ensure_ascii=False, default=str)[:3500],
            "RETRIEVED_DOCUMENTS:\n" + docs,
        ]
        if tool_log:
            parts.append("TOOL_RESULTS_THIS_TURN:\n" + "\n".join(
                json.dumps(t, ensure_ascii=False, default=str)[:1800] for t in tool_log))
        parts.append(f"CUSTOMER_MESSAGE: {text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ turn
    def respond(self, text):
        return self.respond_full(text).reply

    def respond_full(self, text):
        text = (text or "").strip()
        chunks = self.kb.search(self._retrieval_query(text), k=self.top_k)
        tool_log, calls = [], []
        data, reply = {}, ""

        for _ in range(self.max_steps):
            msgs = self.history[-12:] + [{"role": "user", "content": self._compose(text, chunks, tool_log)}]
            data = parse_json(self.llm.chat(prompts.AGENT_SYSTEM, msgs, max_tokens=600))
            action = data.get("action", "reply")

            if data.get("_parse_failed"):
                tool_log.append({"note": "Your previous output was not one complete valid JSON object. "
                                         "Reply with exactly one complete JSON object and nothing else."})
                data = {}
                continue

            if action == "reply":
                # Guard: never confirm a handover that no tool call performed.
                if norm_decision(data.get("decision")) == "escalate" and not self.state["ticket"] \
                        and not any(c["action"] == "create_ticket" for c in calls):
                    tool_log.append({"note": "You chose decision=escalate but create_ticket has not been called. "
                                             "Call create_ticket with a short summary first, then confirm using its result."})
                    continue
                reply = (data.get("reply") or "").strip()
                break

            args = data.get("args") or {}
            result, ok = self.tools.run(action, args)
            call = {"action": action, "args": mask_args(args), "result": result, "ok": ok}
            calls.append(call)
            tool_log.append({"tool": action, "args": mask_args(args), "result": result, "ok": ok})

            if action == "lookup_order":
                oid = str(args.get("order_id", "")).strip()
                if oid:
                    self.state["order_id"] = oid
                gave_proof = any(args.get(k) for k in ("verification_value", "phone_last4", "pincode"))
                if ok:
                    self.state["order"] = result
                    status = result.get("status", "") if isinstance(result, dict) else ""
                    chunks = self._merge(chunks, self.kb.search(f"{text} {status} order", k=self.top_k))
                elif gave_proof:
                    self.state["verify_failures"] += 1
            elif action == "create_ticket" and ok:
                self.state["ticket"] = result
            elif action == "create_ticket" and not ok:
                tool_log.append({"note": "Ticket creation FAILED (created is not true). Do NOT tell the customer a ticket or "
                                         "handover was created; apologise and offer to try again."})

        if not reply:
            reply = "Sorry, I'm having trouble with that. Could you please say it again?"
            data.setdefault("decision", "clarify")

        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": reply})
        self.state["turns"] += 1
        return TurnResult(
            reply=reply,
            decision=norm_decision(data.get("decision")),
            intent=data.get("intent", ""),
            sources=data.get("used_sources") or [],
            chunks=[{k: c[k] for k in ("id", "section_id", "source", "status", "score", "text")} for c in chunks],
            tool_calls=calls,
        )
