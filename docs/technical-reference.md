# Model and command reference

## Single-entry decisions

For selected win probabilities p1 through pT:

```
Expected successful weeks = p1 + p1*p2 + ... + p1*p2*...*pT
```

SciPy/HiGHS solves a mixed-integer linear program with one eligible team per week and no team reuse. Previously used teams are excluded. The result applies to the supplied probabilities and constraints, within its reported numerical gap.

Saving a team can improve this objective:

| Plan | First pick | Second pick | Expected successful weeks |
| --- | ---: | ---: | ---: |
| A then C | 0.95 | 0.60 | 1.520 |
| B then A | 0.80 | 0.94 | 1.552 |

This is an illustrative calculation, not measured predictive advantage.

## Multiple-entry decisions

An exact alive/dead state calculation evaluates each portfolio of fixed paths. Same-team picks share an outcome; opposite sides cannot both win. Different games and weeks are assumed independent, and explicit ties are omitted.

The objective is expected prize share E[K/(K+F)], with zero when K=0. K is the number of owned finalists and F is a supplied fixed number of other finalists. This assumes a terminal payoff; it does not predict other contestants' choices. Fixed-F sensitivity evaluates the selected portfolio. Rerunning with another F may select different picks.

Candidate mode builds paths with conditional beam search. It evaluates every generated combination when they fit the budget; otherwise it uses a seeded sample and local improvement. Reports distinguish these cases and make no global optimum claim for candidate mode.

## Global three-entry optimization

`--search global` uses branch-and-bound over all legal fixed paths for one to three active entries. It does not truncate paths with a beam. Shared and opposing outcomes use the exact state calculation.

A legal portfolio supplies the lower bound. A maximum-product no-repeat assignment bounds each entry's remaining survival probability. Combining these optimistic probabilities with the concave envelope of K/(K+F) gives an upper bound on possible prize share. Branches are pruned only when they cannot improve beyond the requested tolerance.

Reports show both bounds, absolute/relative gaps, visited nodes and the stop reason. `optimal_within_tolerance` certifies a gap within the disclosed tolerance and numerical slack. `node_limit` or `time_limit` with an open gap does not certify an optimum. The optimizer still returns a legal portfolio when stopped.

Controls on `recommend-large` and `weekly-run --pool large`:

- `--global-max-nodes`: default 100000; zero disables the node limit.
- `--global-max-seconds`: default 30; zero disables the time limit.
- `--global-absolute-gap`: default 1e-10.

Limits are checked between nodes after initial setup. Setup or one operation can overrun a time threshold. The number of paths grows exponentially, so a real full season may be too large to certify in practical time.

```bash
python examples/global_demo.py
python examples/benchmark_global.py
```

The demo includes a certified small problem and a one-node stopped search. Tests compare tiny problems against independent enumeration of all legal portfolios and game outcomes. Benchmarks use invented inputs, including a bounded full-season-sized example.

## Saved reports

Reports save probabilities, their sources and available input times, pick history, search settings and results. The candidate search seed is saved so the same sample can be rerun.

```bash
python survivor.py replay PATH_TO_REPORT.json
```

This uses saved inputs without refreshing odds or accessing credentials. Editing valid inputs is allowed and runs a new calculation. Global reruns use the saved visited-node count without a time cutoff. A run originally stopped by time can report a node-limit stop on rerun; timings and near-tied numerical decisions can vary across computers.

Retrieval or manual-verification time is distinct from a bookmaker's update time. Unknown update times remain unknown. Running the optimizer does not make older prices fresh. Required manual fallbacks expire after 24 hours by default for live commands; saved-report reruns keep the recorded inputs.

## Optional live workflow

Use an ignored directory such as `--root runtime` for real data. The standard runtime directories are excluded by `.gitignore`; add equivalent exclusions for custom locations.

```bash
export THE_ODDS_API_KEY="your-own-key"
python survivor.py --root runtime setup-season --season YEAR
python survivor.py --root runtime download-schedule --season YEAR
python survivor.py --root runtime check-odds-coverage --season YEAR
python survivor.py --root runtime refresh-odds --season YEAR --week WEEK
python survivor.py --root runtime record-pick --season YEAR --contest small_pool --entry demo --week WEEK --team TEAM
python survivor.py --root runtime record-result --season YEAR --contest small_pool --entry demo --week WEEK --result win
python survivor.py --root runtime weekly-run --season YEAR --pool small --entry demo
```

Replace the placeholders with your own values. The Odds API is optional; each user supplies their own key or input data. Never commit a key. The public package does not read the private project's configuration files. Error messages redact credentials, and the offline demos/reruns do not require a key.

`--help` lists manual-fallback, result and multi-entry options. `recommend-small` accepts either contest's entry history. `recommend-large` excludes eliminated entries. `weekly-run --pool large` requires `--other-survivors`, accepts `--seed` (default 0), and can stop if future odds coverage is incomplete. A recommendation does not submit or record a pick.

Each bookmaker's two-sided implied probabilities are proportionally normalized, then the median fair probability is taken across books. Contest eligibility rules are implemented in code; generated TOML values are illustrative and do not drive optimization. Additional CSV headers and counters are storage support, not recommendation inputs.
