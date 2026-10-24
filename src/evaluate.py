"""Evaluate on practice/questions.csv (Step 1 vs Step 2) and practice/conversations.csv (Step 3).

Scoring uses the dataset's own fields (no guessing):
  * must_include      : ' ; ' separates required items, ' / ' separates alternatives (any one is enough)
  * must_not_include  : none of these may appear
  * expected_decision / acceptable_decisions : the agent's `decision` must be one of them
  * gold_evidence     : document section IDs (e.g. WAR-2) that retrieval should find
  * expected_tool_calls (conversations): tool names that should be called on that turn

Commands
    python -m src.evaluate --decisions          # decision vocabulary + sample tool-call rows
    python -m src.evaluate --show-columns
    python -m src.evaluate questions --limit 10
    python -m src.evaluate questions --use-asr
    python -m src.evaluate questions --judge    # extra LLM judge vs reference_answer (more calls)
    python -m src.evaluate conversations
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict

from . import config, prompts
from .agent import PromptOnlyAgent, SupportAgent, parse_json
from .data import read_csv
from .llm import LLM

OUT = config.ROOT / "eval_results"
TOOLS = ("lookup_order", "create_ticket", "check_pincode", "product_info", "date_calc")


# ----------------------------------------------------------------- scoring helpers
def items(spec):
    return [x.strip() for x in re.split(r"\s;\s|;", spec or "") if x.strip()]


def missing_required(text, spec):
    t = (text or "").lower()
    out = []
    for item in items(spec):
        alts = [a.strip().lower() for a in re.split(r"\s/\s", item) if a.strip()]
        if not any(a in t for a in alts):
            out.append(item)
    return out


def forbidden_found(text, spec):
    t = (text or "").lower()
    bad = []
    for item in items(spec):
        for alt in re.split(r"\s/\s", item):
            alt = alt.strip().lower()
            if alt and alt in t:
                bad.append(alt)
    return bad


def gold_sections(gold):
    """Section IDs like WAR-2 (entries such as products.csv:ELE-0012 are data lookups, not sections)."""
    return [g.strip().upper() for g in re.split(r"[;,]", gold or "") if g.strip() and ":" not in g and ".csv" not in g]


def retrieval_scores(chunks, gold):
    want = gold_sections(gold)
    if not want:
        return None, None
    got = {c["section_id"].upper() for c in chunks if c["section_id"]}
    found = [w for w in want if w in got]
    return bool(found), len(found) / len(want)


def decision_ok(decision, row):
    acceptable = {d.lower() for d in items(row.get("acceptable_decisions", ""))}
    if row.get("expected_decision"):
        acceptable.add(row["expected_decision"].strip().lower())
    return (decision or "").strip().lower() in acceptable if acceptable else None


def judge(llm, question, reference, answer):
    msg = f"Customer question: {question}\nReference answer: {reference}\nAgent answer: {answer}"
    try:
        out = parse_json(llm.chat(prompts.JUDGE_SYSTEM, [{"role": "user", "content": msg}], max_tokens=300))
        return bool(out.get("correct")), out.get("reason", "")
    except Exception as exc:
        return False, f"judge error: {exc}"


def pct(a, b):
    return round(100 * a / b, 1) if b else None


# ----------------------------------------------------------------- questions (Step 1 vs 2)
def eval_questions(limit=None, use_asr=False, use_judge=False, delay=0.0):
    rows = read_csv(config.PRACTICE_DIR / "questions.csv")[: limit or None]
    llm = LLM()
    step1, agent = PromptOnlyAgent(llm), SupportAgent(llm=llm)
    results, t = [], Counter()
    for row in rows:
        question = row.get("asr_text" if use_asr else "question", "") or row.get("question", "")
        rec = {"id": row.get("id"), "language": row.get("language"), "intent": row.get("intent"), "question": question}
        for name, ag in (("step1", step1), ("step2", agent)):
            ag.reset()
            res = ag.respond_full(question)
            miss = missing_required(res.reply, row.get("must_include"))
            bad = forbidden_found(res.reply, row.get("must_not_include"))
            content_ok = not miss and not bad
            rec[f"{name}_answer"], rec[f"{name}_content_ok"] = res.reply, content_ok
            rec[f"{name}_missing"], rec[f"{name}_forbidden"] = miss, bad
            t[f"{name}_content"] += content_ok
            if name == "step2":
                d_ok = decision_ok(res.decision, row)
                hit, recall = retrieval_scores(res.chunks, row.get("gold_evidence"))
                rec.update(decision=res.decision, expected_decision=row.get("expected_decision"), decision_ok=d_ok,
                           retrieved=[c["id"] for c in res.chunks], retrieval_hit=hit, retrieval_recall=recall,
                           tools=[c["action"] for c in res.tool_calls], full_pass=bool(d_ok and content_ok))
                t["decision"] += bool(d_ok)
                t["full"] += rec["full_pass"]
                if hit is not None:
                    t["hit_n"] += 1
                    t["hit"] += hit
                    t["recall_sum"] += recall
            if use_judge and row.get("reference_answer"):
                ok, why = judge(llm, question, row["reference_answer"], res.reply)
                rec[f"{name}_judge"], rec[f"{name}_judge_why"] = ok, why
                t[f"{name}_judge"] += ok
                t["judge_n"] += name == "step1"
            if delay:
                time.sleep(delay)
        t["n"] += 1
        print(f"{rec['id']:>8}  step1={'ok ' if rec['step1_content_ok'] else 'BAD'}  step2={'ok ' if rec['step2_content_ok'] else 'BAD'}"
              f"  decision={rec['decision']}/{rec['expected_decision']}  hit={rec['retrieval_hit']}")
    n = t["n"]
    summary = {
        "questions": n, "text_used": "asr_text" if use_asr else "question",
        "step1_content_pass_pct": pct(t["step1_content"], n), "step2_content_pass_pct": pct(t["step2_content"], n),
        "step2_decision_accuracy_pct": pct(t["decision"], n), "step2_full_pass_pct": pct(t["full"], n),
        "retrieval_hit_pct": pct(t["hit"], t["hit_n"]),
        "retrieval_avg_recall_pct": pct(t["recall_sum"], t["hit_n"]),
    }
    if use_judge:
        summary["step1_judge_pct"], summary["step2_judge_pct"] = pct(t["step1_judge"], t["judge_n"]), pct(t["step2_judge"], t["judge_n"])
    OUT.mkdir(exist_ok=True)
    tag = "asr" if use_asr else "clean"
    (OUT / f"questions_results_{tag}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / f"questions_summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY (put these numbers in report.pdf):")
    print(json.dumps(summary, indent=2))


# ----------------------------------------------------------------- conversations (Step 3)
def tool_names(spec):
    return [m for m in TOOLS if m in (spec or "")]


def eval_conversations(limit=None, delay=0.0):
    rows = read_csv(config.PRACTICE_DIR / "conversations.csv")
    convs = defaultdict(list)
    for r in rows:
        convs[r["conversation_id"]].append(r)
    agent, out, t = SupportAgent(), [], Counter()
    for cid, turns in list(convs.items())[: limit or None]:
        agent.reset()
        for r in sorted(turns, key=lambda x: int(x["turn"]) if str(x.get("turn", "")).isdigit() else 0):
            res = agent.respond_full(r["customer_message"])  # NOTE: order_id / customer_id columns are NOT given to the agent
            miss = missing_required(res.reply, r.get("must_include"))
            bad = forbidden_found(res.reply, r.get("must_not_include"))
            d_ok = decision_ok(res.decision, r)
            want_tools, got_tools = tool_names(r.get("expected_tool_calls")), [c["action"] for c in res.tool_calls]
            tools_ok = all(w in got_tools for w in want_tools)
            hit, _ = retrieval_scores(res.chunks, r.get("gold_evidence"))
            rec = {"conversation_id": cid, "turn": r.get("turn"), "scenario": r.get("scenario"),
                   "customer": r["customer_message"], "agent": res.reply, "decision": res.decision,
                   "expected_decision": r.get("expected_decision"), "decision_ok": d_ok,
                   "missing": miss, "forbidden": bad, "expected_tools": r.get("expected_tool_calls"),
                   "tools": got_tools, "tools_ok": tools_ok, "retrieval_hit": hit,
                   "sources": [c["id"] for c in res.chunks]}
            out.append(rec)
            t["turns"] += 1
            t["decision"] += bool(d_ok)
            t["content"] += (not miss and not bad)
            t["tools"] += tools_ok
            if hit is not None:
                t["hit_n"] += 1
                t["hit"] += hit
            flag = "ok " if (d_ok and not miss and not bad and tools_ok) else "BAD"
            print(f"{flag} {cid} t{r.get('turn')}: {r['customer_message'][:50]!r} -> {res.decision}/{r.get('expected_decision')} tools={got_tools}")
            if delay:
                time.sleep(delay)
    n = t["turns"]
    summary = {"conversations": len(convs) if not limit else min(limit, len(convs)), "turns": n,
               "decision_accuracy_pct": pct(t["decision"], n), "content_pass_pct": pct(t["content"], n),
               "tool_call_match_pct": pct(t["tools"], n), "retrieval_hit_pct": pct(t["hit"], t["hit_n"])}
    OUT.mkdir(exist_ok=True)
    (OUT / "conversations_transcripts.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "conversations_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))


# ----------------------------------------------------------------- inspection
def show_decisions():
    q = read_csv(config.PRACTICE_DIR / "questions.csv")
    c = read_csv(config.PRACTICE_DIR / "conversations.csv")
    print("questions.csv expected_decision:", dict(Counter(r["expected_decision"] for r in q)))
    print("questions.csv acceptable_decisions:", dict(Counter(r["acceptable_decisions"] for r in q)))
    print("conversations.csv expected_decision:", dict(Counter(r["expected_decision"] for r in c)))
    print("conversations.csv expected_tool_calls (distinct):", dict(Counter(r["expected_tool_calls"] for r in c)))
    print("\nSample conversation rows that expect tool calls:")
    shown = 0
    for r in c:
        if r.get("expected_tool_calls") and shown < 6:
            print(" ", json.dumps({k: r[k] for k in ("conversation_id", "turn", "customer_message", "expected_decision",
                                                      "expected_tool_calls")}, ensure_ascii=False))
            shown += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["questions", "conversations"], default="questions")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--use-asr", action="store_true", help="use asr_text (speech-to-text style) instead of clean text")
    ap.add_argument("--judge", action="store_true", help="also grade against reference_answer with an LLM judge")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions (rate limits)")
    ap.add_argument("--show-columns", action="store_true")
    ap.add_argument("--decisions", action="store_true")
    a = ap.parse_args()
    if a.decisions:
        return show_decisions()
    if a.show_columns:
        for name in ("questions.csv", "conversations.csv"):
            rows = read_csv(config.PRACTICE_DIR / name)
            print(name, "->", list(rows[0].keys()) if rows else "MISSING")
        return
    if a.mode == "questions":
        eval_questions(a.limit, a.use_asr, a.judge, a.delay)
    else:
        eval_conversations(a.limit, a.delay)


if __name__ == "__main__":
    main()
