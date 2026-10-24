# VoiceOps 2026 — ZentroMart Voice Support Agent

A customer-support agent for the fictional Indian store **ZentroMart**. It answers from the
company's policy documents (RAG), holds a multi-turn conversation with order lookup / ticket
tools, and talks over the phone using **Gnani STT + TTS** (the only speech services used).

GitHub repo: `<PASTE YOUR PUBLIC GITHUB URL HERE>`

```
audio in -> Gnani STT -> agent (LLM + RAG + tools) -> Gnani TTS -> audio out
```

## Architecture (src/)
| File | Role |
|---|---|
| `kb.py` | Chunks `kb/*.md` by `[SECTION-ID]`, keeps status / effective_from, hybrid dense + BM25 retrieval, source precedence (pol > agent > help > faq > mkt), superseded docs excluded |
| `agent.py` | `PromptOnlyAgent` (Step 1 baseline) and `SupportAgent` (Steps 2-3): grounded answers, conversation memory, tool loop, "no fake actions" guards |
| `tools_adapter.py` | Calls the dataset's own `lookup_order` / `create_ticket` (verification enforced by the tool) + pincode, product and date helpers |
| `data.py` | Business-day maths (Mon-Sat minus `holidays.csv`), reference-data lookups. Never reads orders/customers directly |
| `voice.py` | Gnani STT/TTS, recovery from empty audio, spoken-text normalisation (years in dates, grouped IDs, rupees) |
| `evaluate.py` | Step 1 vs Step 2 on `practice/questions.csv`, and `practice/conversations.csv` |

Key behaviours: answers only from retrieved chunks; says "I don't know" and offers a human when
the documents don't cover a question; asks for phone last-4 or pincode before any order details;
confirms a handover only after `create_ticket` returns `created=True`; first reply says it is an AI assistant.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then fill in keys
# extract the dataset zip next to this repo and set DATASET_DIR in .env
python Capstone_Project_Dataset_FTTT_2026/validate_dataset.py
```
Keys needed: `GNANI_API_KEY` (app.gnani.ai/voice) and an LLM key (`ANTHROPIC_API_KEY`, or
`OPENAI_API_KEY` + `OPENAI_BASE_URL` with `LLM_PROVIDER=openai`). Set `EMBEDDING_MODEL=none` to use BM25-only retrieval
(no model download).

## Run command (batch, used by the organisers)
```bash
python run.py --audio_dir ./test_audio --out_dir ./outputs
# or: ./run.sh ./test_audio ./outputs
```
For every audio file it saves `<name>.json` (transcript, reply, decision, tool calls, **retrieved chunks**),
`<name>_reply.wav` (spoken reply) and `results.csv / results.jsonl`.
Options: `--lang hi-IN` (STT language), `--group-by-prefix` (files named `conv_turn.wav` share memory).

## Other ways to run
```bash
python run.py --interactive            # type to the agent, no Gnani needed
python run.py --text-only "Can I return a phone after 15 days?"
python app.py                          # browser demo with microphone (for the demo video)
```

## Evaluation
```bash
python -m src.evaluate --show-columns        # inspect practice CSV columns first
python -m src.evaluate questions             # Step 1 (prompt-only) vs Step 2 (RAG) + retrieval hit-rate
python -m src.evaluate questions --use-asr   # same, on ASR-style text
python -m src.evaluate conversations         # multi-turn transcripts + decision / tool-call match
```
Results are written to `eval_results/`. Answers are graded by an LLM judge against the expected
answer / decision; retrieval hit-rate compares retrieved chunks with `gold_evidence`.

## Rules followed
Gnani is the only STT/TTS; only the synthetic dataset is used; no hardcoded answers; no keys in the repo.
