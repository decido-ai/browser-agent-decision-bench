# Browser-agent decision bench

How good is the model that picks a browser agent's next action? This benchmark answers that for
[Jev Ultrafast](https://github.com/browser-use/jev-ultrafast), Browser Use's open-source browser agent, with any
decision model that speaks its `/v1/systemone` request shape (TypeSafe's Jev, a self-hosted model, your own).

At every step, Jev Ultrafast asks its decision model two questions: which operation (click, type, select, scroll,
wait, done, blocked) and which element. We replay **1,500 steps recorded by people on real websites**
([Multimodal-Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) test sets) through the agent's own,
unmodified request builder, send the identical request to each model, and score whether it picked the recorded action.

## Results

Step accuracy (%): a step counts only if both the operation and the element are right. 500 steps per test set.

| decision model | new tasks | new websites | new domains | unseen (web + domain) | **all 1,500** | element | operation |
|---|---:|---:|---:|---:|---:|---:|---:|
| **[Decido-1 (web-trained)](https://huggingface.co/decido-ai/decido-1-web-26b-a4b)** | 79.4 | 79.6 | 76.8 | 78.2 | **78.6** | 80.7 | 94.1 |
| Jev 1.14 (TypeSafe API) | 62.4 | 59.0 | 58.2 | 58.6 | **59.9** | 62.3 | 86.8 |
| Jev 1.13 (TypeSafe API) | 55.8 | 52.0 | 55.4 | 53.7 | **54.4** | 56.9 | 80.0 |
| Decido-1 (released, no web training) | 42.0 | 39.8 | 45.8 | 42.8 | **42.5** | 44.8 | 65.2 |

Paired differences, bootstrap 95% CI over steps (`results/compare.txt`):

| comparison | all 1,500 | unseen websites + domains |
|---|---:|---:|
| Decido-1 (web-trained) − Jev 1.14 | +18.7 [+16.2, +21.3] | +19.6 [+16.4, +22.8] |
| Decido-1 (web-trained) − Jev 1.13 | +24.2 [+21.7, +26.9] | +24.5 [+21.3, +27.7] |

Failed requests count as wrong: the same 6 very long requests failed for every model (over the endpoint's input limit), plus 1 more for Jev. Jev was called
through the TypeSafe API in November 2026. Full write-up, including live-site results (where the two are on par,
74% vs 75% of 100 tasks) and a video: https://getdecido.ai/research/decido-1-web

**Steps where the answer stands out.** In 56 of the 1,500 steps the correct element is non-interactive and has no name
(or a name over 80 characters), while the builder drops such elements from the sampled alternatives. A model trained on
steps built this way can learn to spot that. `score` and `compare` therefore also report every result without those
steps: Decido-1 (web-trained) 79.1 vs Jev 1.14 61.5 (+17.6 [+15.0, +20.2]) and Jev 1.13 55.9 (+23.2 [+20.5, +25.8]) on
the remaining 1,444. Future versions of the builder will apply the same filter to the correct element.

**Add your model:** run it on `data/steps-final.zip` and open a pull request with your results file. We'll add a row.

## Reproduce

```bash
pip install -r requirements.txt            # pins jev-ultrafast to the commit used here
# the step set is data/steps-final.zip (password: mind2web); the commands read it directly

# 1) re-score the bundled results (no API key, no download)
python decision_bench.py score   --steps data/steps-final.zip results/*.jsonl
python decision_bench.py compare --steps data/steps-final.zip results/decido-1-web-trained.jsonl results/jev-1.14.jsonl

# 2) run any endpoint that speaks the /v1/systemone request shape
export TYPESAFE_API_KEY=...                # only for TypeSafe's API
python decision_bench.py run --steps data/steps-final.zip --url https://api.typesafe.ai/v1/systemone \
    --api-key-env TYPESAFE_API_KEY --model jev-1.14.0 --out results/my-jev.jsonl --workers 4

#    e.g. Decido-1 (web-trained), self-hosted on one 80 GB NVIDIA GPU (Jev-compatible server):
#    pip install "decido-ai[vllm,server]>=1.1.1"
#    DECIDO_N_CTX=32768 decido serve decido-ai/decido-1-web-26b-a4b --device cuda --engine vllm --port 8080
python decision_bench.py run --steps data/steps-final.zip --url http://127.0.0.1:8080/v1/systemone --out results/mine.jsonl

# 3) rebuild the step set from Multimodal-Mind2Web (download the test_* parquet files first)
python decision_bench.py build --data-dir <parquet dir> --out steps.jsonl --per-split 500 --k 50 --seed 1
shasum -a 256 steps.jsonl                  # compare with data/SHA256SUMS

pytest -q tests                            # re-scoring + an end-to-end run against a local mock endpoint
```

`data/steps-final.zip` contains the file built with the command in step 3 over all test shards. It is zipped with a
password because Mind2Web asks that its test data not be posted unzipped (to keep it out of web-crawled training data).

## How a step is built

- The recorded page's HTML is converted to Jev Ultrafast's observation format (a Python port of its `snapshot.js`): an
  element table with roles, labels, values and select options, plus visible page text.
- Each step offers the recorded element and up to 49 other elements sampled from the same page (`--k 50`), in page order, so the
  position of the right answer gives nothing away.
- History is the recorded earlier steps (offline replay), and the goal is the task description.
- The request body is produced by `jev_ultrafast.model.choose()` itself; only the endpoint URL is swapped.

## Limitations

- Offline replay: the model sees the true earlier steps and 50 candidate elements. A live agent sees its own history and
  every element on the page, and must also decide when to stop. Live task success can differ (see the write-up).
- Mind2Web has no wait/scroll/done labels, so those operations are never correct here.
- Decido-1 (web-trained)'s training data is separate from these test sets: no shared tasks or near-duplicate goals, and
  no websites from the "new websites" and "new domains" sets ("new tasks" uses websites it was trained on, by design).
  Jev's training data is not public. The web-trained model is open: https://huggingface.co/decido-ai/decido-1-web-26b-a4b.
- Results files for Jev contain only the chosen element and operation per step.

## Files

| path | contents |
|---|---|
| `decision_bench.py` | build, run, score, compare |
| `data/steps-final.zip` | the 1,500 test steps, password `mind2web` (`data/SHA256SUMS`, licence in `data/NOTICE.md`) |
| `results/*.jsonl` | one record per step and model |
| `results/scores.txt`, `results/compare.txt` | the tables above |
| `tests/` | checks that run without any API key |

## License

Code: Apache-2.0 (`LICENSE`). The step set is derived from Multimodal-Mind2Web and keeps its licence (`data/NOTICE.md`).
Jev Ultrafast is MIT-licensed by Browser Use.
