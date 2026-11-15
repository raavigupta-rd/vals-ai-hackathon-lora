# Project instructions

Vals hackathon, poker track, on the `pokerbots-starter` base. Two people work here: A (infra and RL) and B (data
and scouting). The working plan is `plan.md`; read it before starting a task.

- Both people now work on the shared `poker` branch. Fetch and pull with `--ff-only` before starting edits;
  coordinate file ownership and push completed commits promptly. `poker-a` and `poker-b` are historical branches.
- Read `README (2).md` for strategy priorities, reconciled with confirmed event rules and capabilities in `plan.md`
  and `poker/CAPABILITIES.md`. Conditional SFT/on-policy suggestions do not establish permission or API support.

- Ask which lane (A or B) the user is in if it is not clear, and stay inside that lane's steps and files (`plan.md`, "Team and setup").
- Pick agents, models and skills from the "Claude setup" tables in `plan.md`. Opus plans and reviews; Sonnet subagents do bulk work; Haiku only for bulk labeling after code filters.
- Never submit to the arena or delete a deployment unless the user asks in that message. Only lane A submits.
- Put new code in `poker/`. Keep `src/pokerbots/` empty: it is the AC2 project package.
- The arena plays the official `harness/` unmodified. Training-only changes must not break play under the event rules; check with `harness/play_local.py`.
- Tests: `uv run pytest` at the root and `cd harness && uv run pytest`.
- When the plan changes (new judge rulings, new owners, new steps), update `plan.md` so both people see it.
