#!/usr/bin/env python
"""Batch voice run (the organisers' entry point).

    python run.py --audio_dir ./test_audio --out_dir ./outputs

For every audio file in --audio_dir: Gnani STT -> agent (RAG + tools) -> Gnani TTS.
Saves, per file, <name>.json (transcript, reply, decision, tool calls, retrieved chunks) and
<name>_reply.wav, plus results.jsonl / results.csv for the whole folder.

Sessions: every file is an independent conversation by default. If files are named
<conversation>_<turn>.<ext> (e.g. c07_02.wav) use --group-by-prefix to keep memory across turns.
Other modes: --text-only "question"   --interactive (type instead of speak, no Gnani needed)
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

from src import config
from src.agent import SupportAgent
from src.voice import RECOVERY_REPLY, Voice, detect_lang

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def session_key(path, group):
    if group:
        m = re.match(r"^(.*?)[_\-]?(\d+)$", path.stem)
        return m.group(1) if m else path.stem
    return path.stem


def run_batch(audio_dir, out_dir, lang, group):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((f for f in Path(audio_dir).iterdir() if f.suffix.lower() in AUDIO_EXT), key=natural_key)
    if not files:
        sys.exit(f"No audio files found in {audio_dir}")
    voice, agent, last_session, rows = Voice(), SupportAgent(), None, []
    for f in files:
        sess = session_key(f, group)
        if sess != last_session:
            agent.reset()
            last_session = sess
        reply_path = out_dir / f"{f.stem}_reply.wav"
        try:
            heard = voice.transcribe(f, lang)
            if heard.strip():
                res = agent.respond_full(heard)
                reply, decision = res.reply, res.decision
                tools = [{"action": c["action"], "args": c["args"], "ok": c["ok"]} for c in res.tool_calls]
                chunks = res.chunks
            else:  # unclear / silent audio -> polite recovery, never crash
                reply, decision, tools, chunks = RECOVERY_REPLY, "clarify", [], []
            audio_out = voice.speak(reply, reply_path, detect_lang(reply, lang))
        except Exception as exc:  # keep the batch going; still emit a spoken apology
            heard, decision, tools, chunks = "", "error", [], []
            reply = "Sorry, something went wrong on my side. Could you please try again?"
            audio_out = voice.speak(reply, reply_path, lang)
            print(f"[error] {f.name}: {exc}")
        rec = {"file": f.name, "session": sess, "transcript": heard, "reply": reply, "decision": decision,
               "tool_calls": tools, "retrieved_chunks": chunks, "reply_audio": Path(audio_out).name if audio_out else None}
        (out_dir / f"{f.stem}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(rec)
        print(f"{f.name}: {heard!r} -> {reply!r}")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "transcript", "reply", "decision", "sources"])
        for r in rows:
            w.writerow([r["file"], r["transcript"], r["reply"], r["decision"],
                        ";".join(c["id"] for c in r["retrieved_chunks"])])
    print(f"\nDone. {len(rows)} files -> {out_dir}")


def interactive():
    agent = SupportAgent()
    print("ZentroMart agent (text mode). Type 'quit' to exit, 'reset' for a new customer.")
    while True:
        try:
            text = input("customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("quit", "exit"):
            break
        if text == "reset":
            agent.reset()
            continue
        res = agent.respond_full(text)
        print(f"agent> {res.reply}\n       [{res.decision}; tools={[c['action'] for c in res.tool_calls]}; "
              f"sources={[c['id'] for c in res.chunks[:3]]}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio_dir")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--lang", default=config.DEFAULT_LANG, help="STT language code, e.g. en-IN, hi-IN, gu-IN")
    ap.add_argument("--group-by-prefix", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--text-only")
    a = ap.parse_args()
    if a.interactive:
        return interactive()
    if a.text_only:
        res = SupportAgent().respond_full(a.text_only)
        print(json.dumps({"reply": res.reply, "decision": res.decision,
                          "chunks": [c["id"] for c in res.chunks]}, indent=2, ensure_ascii=False))
        return
    if not a.audio_dir:
        ap.error("--audio_dir is required (or use --interactive / --text-only)")
    run_batch(a.audio_dir, a.out_dir, a.lang, a.group_by_prefix)


if __name__ == "__main__":
    main()
