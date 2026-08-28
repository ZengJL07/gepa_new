# AIME Math

Optimize a math-solving prompt for AIME competition problems. The solver LLM (GPT-4.1-mini with chain-of-thought) is fixed — GEPA optimizes only the system prompt.

Despite the name, this directory hosts several prompt-optimization tasks sharing one harness: datasets are entries in `datasets.SPECS`, grading rules are entries in `scoring.SCORERS`, and the `AIME_*` environment variables configure all of them. Besides the math datasets there is a vision task — see [PPT whitespace detection](#ppt-whitespace-detection-pptblank) below.

## Dataset

- **Train + Val**: `AI-MO/aimo-validation-aime` (AIME 2022–2024), split 50/50
- **Test**: `MathArena/aime_2025` (AIME 2025)

## Setup

From the repo root (`gepa/`):

```bash
uv venv
uv pip install datasets dspy litellm
uv pip install -e .  # must come after dspy to avoid PyPI overwrite
```

## Run

```bash
export OPENAI_API_KEY=...
uv run python -m examples.aime_math.main
```

After optimization, the script evaluates both the baseline and best-found prompt on the AIME 2025 test set and prints the improvement.

## PPT whitespace detection (`pptblank`)

A vision task on the same harness: given a rendered presentation slide, does it have an excessive-whitespace layout defect? Both solver and reflector must be vision models (the solver sees the slide, the reflector sees the slide with annotation boxes drawn on it).

### Answer format

The model states its verdict inside `\box{}` — `\box{yes}` or `\box{no}` — which separates the final answer from any reasoning around it. The parser takes the **last** box in the response, so a prompt that restates the template before committing (`"answer \box{yes} or \box{no} ... \box{no}"`) is read as its final answer rather than its first mention. `\boxed{...}` (the real LaTeX macro, which models emit out of habit), `$...$` wrappers, surrounding whitespace and any casing are all accepted.

A bare `yes` / `no` with no box still grades on its merits — a formatting slip should not be scored as a wrong *answer* — but the feedback then tells the reflection LM the format drifted, so it can tighten the prompt. `\box{maybe}` is treated as a genuine non-answer: the parser will not scavenge a verdict out of the surrounding prose once a box is present.

### Preparing the data

The source is the annotated `dataset_v4` from the ai-ppt-dataset tooling. One command converts it into everything the run needs:

```bash
PPTBLANK_SRC=/path/to/ai-ppt-dataset/dataset_v4 \
  python -m examples.aime_math.prepare_pptblank
```

That writes `data/pptblank.json` (pre-split records), `data/pptblank_gold.json` (id → gold, for post-hoc F1), and re-encoded slides under `data/pptblank/`. The script asserts its own output counts, so a source-data change fails loudly.

### Labels

Slides are labelled with whitespace boxes carrying a `severe` / `mild` severity, which map onto three gold states:

| Boxes on the slide | Gold | Count |
|---|---|---|
| any `severe` box | `yes` | 94 |
| only `mild` boxes | `either` — both answers score 1.0 | 65 |
| no boxes | `no` | 108 |

`either` slides are dropped from train/val (a constant 1.0 is not a learning signal and would consume minibatch budget) and kept in test for reporting.

**The labels do not follow the annotation standard in the source `DATASET_DESIGN_DOC.md.`** That document defines `severe` as a whitespace region above 30% of page area and `mild` as 15–30%. Measured from `index.json`, `severe` boxes have a median area of **4.4%** and a maximum of 25%; **no box anywhere exceeds 30%**, and the `severe` and `mild` size ranges overlap completely — severity is an independent human judgement, not an area threshold. In practice the annotator marked *small local gaps*, not large empty regions. Prompts written against the documented standard therefore ask about something absent from the labels and score near-zero recall; framing the task around local gaps that break the slide's own spacing rhythm is what works.

### Splits

By source `.pptx`, not by slide — slides from one deck share a template, so a slide-level split leaks. Two decks per style go to train, two to val, the remaining four per style to test:

| Split | Decks | Slides | pos | neg | mild |
|---|---|---|---|---|---|
| train | 6 | 45 | 20 | 25 | (22 dropped) |
| val | 6 | 48 | 24 | 24 | (14 dropped) |
| test | 12 | 138 | 50 | 59 | 29 |

The deck assignment came from an exhaustive search over all `C(8,2)·C(6,2)` per-style options, minimizing each split's deviation from the global 0.465 positive share. One known imbalance: `flat_illustration` has only 19 positives across all eight of its decks, so its test slice is 5 pos / 28 neg — report that style's F1 with the caveat, or report only its accuracy.

### Run

```bash
export DEEPSEEK_API_KEY=...          # any key for the api.luminai.cc gateway
bash scripts/formal/run_pptblank_gepa_captransfer.sh
```

Defaults: solver `gemini-3-flash`, reflector `claude-opus-4-6-thinking`, 8 workers (every call ships a ~75 KB base64 image, and the upstream pool returns transient rate-limit errors under heavier concurrency; both LM paths retry 3×).

### Scoring, and why F1 is computed afterwards

The loop optimizes per-example 0/1 correctness. F1 cannot be the in-loop objective: GEPA aggregates by averaging per-example scores, and `F1 = 2TP/(2TP+FP+FN)` is not a mean of per-example values — true negatives never enter the denominator and no single example knows the corpus-wide counts.

So F1 is reconstructed after the run, exactly rather than approximately:

```bash
python scripts/analysis/pptblank_f1.py <run_dir>
```

This reads `prog_candidate_val_subscores` from `gepa_state.bin`, joins it against the gold labels, prints a per-candidate confusion matrix and F1, and reports whether selecting on accuracy (what `GEPAResult.best_idx` does) picks a different candidate than selecting on F1. Because every `either` slide was removed from val, each val example is unambiguously positive or negative, so `score == 1` on a positive example is exactly one true positive.

Note that val ids are **positional indices** into the valset — GEPA's `ListDataLoader` keys examples by list position — so the script rebuilds the valset through the same loader call and needs the same `AIME_VAL_K` / `AIME_SEED` the run used. It verifies coverage and refuses to print numbers if the split does not match.

F1's asymmetry does reach the loop, but through the reflection text rather than the score: `score_yesno` emits distinct feedback for a miss versus a false alarm, telling the reflector that misses cost more.

### Scoring the test split

`eval_dataset.py` works on `pptblank` (pass both `AIME_EVAL_PROMPT` and `AIME_OPTIMIZED_PROMPT`, since its defaults are the math prompts) but reports only mean accuracy. For F1, use the dedicated evaluator, which keeps per-example verdicts:

```bash
python -m examples.aime_math.eval_pptblank \
    --optimized-file /path/to/optimized_prompt.txt \
    --output /tmp/pptblank_test.json
```

It prints precision / recall / F1 overall and per design style, flags styles with too few positives for F1 to mean much, counts unparseable verdicts as wrong (a non-answer is a miss on a positive, a false alarm on a negative — dropping them would flatter the result), and shows how the `mild` slides were answered as a calibration signal.

### Budget and parent selection: two traps on this task

A 500-call budget does **not** buy a multi-generation search here. Measured on the first full run (`n_parallel=5`, `valset=48`):

| | calls |
|---|---|
| seed's full valset eval | 48 |
| 6 accepted children × full valset eval | 288 |
| minibatch calls (parent + acceptance gate) over 2 iterations | ~60 |
| **total** | **378 / 500**, then `BudgetExhausted` |

Full-valset evaluations consumed 89% of the budget, so only 2 iterations finished and **every candidate was a child of the seed** — no candidate was ever improved on top of another, which is the entire point of GEPA's hill-climbing. Iteration 3 did select a child as parent (so the mechanism works), but hit the budget ceiling before its proposals were scored.

Cost per iteration is roughly `n_parallel × minibatch × 2 + accepted × |valset|`. To get real generations, cut whichever term dominates:

| `n_parallel` | `valset` | ≈ calls/iter | iterations in 500 |
|---|---|---|---|
| 5 | 48 | 126 | 3 (flat, one generation) |
| 2 | 48 | 60 | 7 |
| 5 | 24 | 78 | 6 |
| 1 | 24 | 30 | 15 |

Second trap: the default `"pareto"` parent selector barely discriminates on this task. It samples a parent in proportion to how many val examples the candidate sits on the Pareto front for — but 16 of 48 val examples are solved by *every* candidate, and each of those hands every candidate a front slot. Measured selection probabilities:

```
cand 1  val_acc=0.812  ->  16.3%
cand 3  val_acc=0.625  ->  12.7%
```

A 1.28× edge for a candidate 19 points better. The aggregate score only affects the pruning *order* inside `remove_dominated_programs` (`is_dominated` reads front membership, not scores), and nothing was pruned, so the score effectively did not participate in selection. `GEPA_CANDIDATE_SELECTOR=current_best` hill-climbs on the aggregate score instead; `top_k_pareto` is a middle ground that keeps some diversity. The default is left as `pareto` so runs stay comparable with the math tasks — change it deliberately, as an experiment, not as a fix.

### Reference points

On the val split (24 pos / 24 neg), always answering `yes` gives F1 = 0.667 and always `no` gives F1 = 0.0 — but the same all-`no` policy gets 61% *accuracy*. Any reported gain should be read against the all-`yes` F1, not against accuracy.

The two metrics genuinely diverge here. Measured on 24 test slides, the seed prompt and a gap-framed rewrite scored **identical accuracy (0.652)** and F1 **0.333 vs 0.692** — the seed buys precision 1.000 at recall 0.200, the rewrite runs recall 0.900 at precision 0.562. This is also why F1 selection is worth applying to the candidate pool after a run, not just to the final report.
