# NFL Survivor Optimizer

Which team should I pick this week, and which teams should I save for later?

In an NFL Survivor pool, each entry picks one team to win each week and can use each team only once. This Python project turns sportsbook odds into win-probability estimates, tracks previous picks and compares legal plans. It grew from a manual spreadsheet into a decision tool with explicit objectives and readable weekly recommendations.

## Model

For **one entry**, the optimizer maximizes the expected number of successful weeks. A safe pick today can be worth passing up if that team offers more value later. The output includes a proposed path, its survival curve and current-week alternatives.

For **up to three entries**, the objective is expected prize share under a supplied number of other finalists. Entries choosing the same team share its outcome; opposite sides of a game cannot both win. Candidate mode compares promising paths. Global mode searches all legal fixed paths and reports whether it established an optimum or stopped with an open gap.

The weekly workflow updates odds, records picks/results and produces Markdown and JSON reports. Future picks are provisional and can be reconsidered next week.

## Usage

Use Python 3.14 for the tested dependency versions. From the project folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pytest -q
python examples/demo.py
python examples/global_demo.py
```

## Examples

The examples use invented games, odds, dates and entries. They need no credentials or network access after installation. Open the Markdown reports under `.demo-output/` to see the picks, model estimates and search status. The JSON reports save the inputs and settings.

The global example includes a small search that finishes and a search deliberately stopped after one node. Its lower bound is the best legal portfolio found; its upper bound limits what an unexplored portfolio could achieve. An open gap means the optimum has not been established.

To rerun a saved report without fetching new odds:

```bash
python survivor.py replay .demo-output/outputs/2030/week_01/synthetic_small_small_pool.json
```

Use `--output .demo-output/second` when rerunning a demo, since existing reports are not overwritten. `python examples/benchmark_global.py` runs a few bounded synthetic timing examples. [Model details and live commands](docs/technical-reference.md) cover the search controls and optional weekly workflow.

## Limitations

Probabilities are market estimates, not known true outcomes. Different games and weeks are assumed independent, explicit ties are omitted, and the portfolio model uses a fixed number of other finalists rather than predicting contestants' choices. The project has no established predictive advantage or profitability result.

A full-season global search may stop before certifying an optimum; reports retain its bounds and stop reason. Small tests compare the optimizer with exhaustive enumeration, but do not establish practical full-season speed. Live runs also depend on complete odds coverage, and saved future prices may be old.

Only code, tests, documentation and invented examples are bundled. Keep real inputs, downloaded odds and personal records in the ignored runtime directories; check provider terms before sharing downloaded data. This is a personal portfolio project. No source license is provided.

The Odds API is optional. Supply your own API key or your own input data. No author's credentials are included.
