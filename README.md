# Tool Robust Exploration

Styu

## Layout

- `src/tool_robust_poc/`
    - tasks: Datasets we use
    - conditions: The prompt conditions
    - atttack_opt: The automated redteam attack generation
    - runners, reporting.
- `scripts/` — table generators and final-run launch scripts.
- `data/` — task input items.
- `results/` — paper-input result archives (Git LFS).

## Setup

```
uv sync
```

Note though there is a dependency
on fllmingo (a currently internal LLM wrapper package). I need to clean that and figure out about exporting here. If you wanted to run from scratch with this existing code, agents could probably migrate it over to normal APIs fairly easily (the parts used are a thin wrapper and all the parameters passed through to the API apparent). This code is mostly intended as a reference for the writeup.
