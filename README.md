# Interview Trainer

A Streamlit chatbot that runs realistic job-interview practice sessions with an AI
interviewer. Pick a company, role, interview stage, add background information
(CV highlights, the job ad, focus topics), and the app:

- generates an interview plan including a hidden **interviewer checklist** (a scorecard
  for that role and stage),
- **conducts the interview** — one question at a time, probing follow-ups, until it
  has what it needs on every checklist item,
- accepts answers by **voice** (microphone, transcribed) or text, and can also **voice**
  its questions,
- ends with a score (0–100), the filled-out checklist with evidence, a
  question-by-question breakdown of what you actually answered and asked, strengths,
  improvements, and a candid summary,
- names **one thing to take away** — the point an interviewer would flag in
  the debrief,
- **stores every session on disk**, so when you repeat the same company/role/stage the
  app avoids questions it has already asked, deliberately probes the weak points from
  last time, and tracks how your score moves.

The interviewer will not tell you what it is grading. The coaching happens in the
evaluation at the end. By default each repeat round varies the interviewer persona, the
questions and the pressure — or you can replay a previous round exactly, with the same
checklist, planned questions and interviewer.

## Pages in the UI

- **Interview** — set up a round (company, role, stage, length from 10 to 60 minutes,
  background information, interviewer voice, and whether to plan with or without your
  history), run it, and read the evaluation.
- **History** — every stored session for a company/role/stage: the scores side by side,
  the full evaluation of any one of them, and its transcript.
- **Research** — fill a company's context folder for the planner; see below.

## What you need

- For the interview LLM, either:
  - an active Claude subscription — the default. The app drives the bundled Claude Code
    CLI, so it authenticates with your `claude login` and needs no key.
  - or an Anthropic API key, to bill the Anthropic API directly.
- Optional: an OpenAI API key, for voice. Without it the app runs text-only.

## Static company context

To help the interviewer ask relevant questions, provide context about the company
(like its mission, values, and culture) and the role (like a job description). Drop files
into `data/context/<company>/` — say `data/context/abc_corp/research.md` and
`data/context/abc_corp/job_description.txt` — and they are picked up automatically whenever
you start an interview for that company. The planner distills them into a short factual
company brief for the interviewer.

The **Research** page fills that folder for you. Describe what you want researched and the
app searches and reads the web, then saves a briefing as `research_<timestamp>.md` in the
right company folder, ready for the next interview. It works through live sources, so a
run takes a few minutes. The report is about the company only. The same page lists
everything already in a company's folder, so you can see exactly what the interviewer is
being told. Nothing is ever overwritten; delete reports you have outgrown.

## Quick start (local)

```bash
cp .env.example .env      # OPENAI_API_KEY for voice; no ANTHROPIC_API_KEY needed
make install
make run
```

## Quick start (Docker)

```bash
cp .env.example .env
make docker-build
make docker-up            # open http://localhost:8501
```

Two things are mounted in, so the container and a local `make run` are the same app:

- `./data/` — sessions and `data/context/` research files. Both ways of running read and
  write the one folder, and it survives rebuilds. The container runs as uid 1000, so what
  it writes stays yours.
- `~/.claude/` — your Claude Code login. The image ships the CLI but not your
  credentials, so mounting it lets Docker use your `claude login` instead of an API key.
  Set `ANTHROPIC_API_KEY` in `.env` instead if you would rather bill the API, or point
  `CLAUDE_HOME` at a different config directory.

The mount is read-write on purpose — the CLI refreshes its OAuth token in place, and a
read-only mount would work until that token expired. The container therefore shares that
folder's config file with your own CLI rather than keeping a copy.

## Development

```bash
make install  # dependencies, including pytest
make test     # unit tests (no API calls — LLM is stubbed)
make format   # black
make lint     # flake8
```

`make help` lists every target, including the `docker-*` ones used above.

Project layout: `frontend/` (Streamlit UI only) · `src/` (business logic, storage,
API adapters — no streamlit imports) · `data/` (stored sessions and the `data/context/`
research files — gitignored) · `tests/` · `docs/`.