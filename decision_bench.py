"""Offline Mind2Web benchmark for the decision model inside Browser Use's Jev Ultrafast agent.

Each Mind2Web step (a recorded page plus the action a person took) is turned into the observation format of Jev
Ultrafast's snapshot.js (a Python port below). The request is then built by the agent's own, unmodified
`jev_ultrafast.model.choose()`, so every decision model receives the same request (only the model id differs).

    python decision_bench.py build   --data-dir <Multimodal-Mind2Web parquet dir> --out steps.jsonl --per-split 500 --k 50 --seed 1
    python decision_bench.py run     --steps data/steps-final.zip --url <.../v1/systemone> --out results/mine.jsonl [--model M] [--api-key-env VAR]
    python decision_bench.py score   --steps data/steps-final.zip results/*.jsonl
    python decision_bench.py compare --steps data/steps-final.zip results/A.jsonl results/B.jsonl [...]

A step is correct only if the chosen action is the recorded one: right operation AND right element.
The step set ships as a zip with password "mind2web" (Mind2Web asks that its test data not be posted unzipped).
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROLES = ['button', 'link', 'checkbox', 'radio', 'switch', 'tab', 'menuitem', 'menuitemradio', 'option', 'gridcell',
         'combobox', 'textbox', 'searchbox', 'spinbutton']
NONINTERACTIVE_ROLE = {"label": "label", "li": "listitem", "td": "cell", "th": "columnheader", "img": "img",
                       "svg": "img", "i": "generic", "span": "generic", "div": "generic", "p": "paragraph", "h1": "heading",
                       "h2": "heading", "h3": "heading", "h4": "heading", "tr": "row", "ul": "list", "nav": "navigation"}
SPLITS = ("test_task", "test_website", "test_domain")
GOLD_OP = {"CLICK": "CLICK", "TYPE": "TYPE_TEXT", "SELECT": "SELECT"}
WS = re.compile(r"\s+")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def clean(s, n=150):
    return WS.sub(" ", s or "").strip()[:n]


# ------------------------------------------------------------------------------ snapshot.js port (lxml), build steps
def role_of(e):
    tag, explicit = e.tag.lower(), (e.get("role") or "").lower()
    if explicit in ROLES:
        return explicit
    if tag in ("button", "summary"):
        return "button"
    if tag == "a":
        return "link"
    if tag == "select":
        return "combobox"
    if tag == "textarea":
        return "textbox"
    if tag == "input":
        t = (e.get("type") or "text").lower()
        if t in ("checkbox", "radio"):
            return t
        if t in ("button", "submit", "reset", "image"):
            return "button"
        if t == "search":
            return "searchbox"
        if t == "number":
            return "spinbutton"
        if t in ("text", "email", "url", "tel"):
            return "textbox"
        return None
    return None


def text_of(e):
    return clean(" ".join(t for t in e.itertext()), 300)


def name_of(e):
    tag = e.tag.lower()
    t = (e.get("type") or "").lower()
    return clean(e.get("aria_label") or e.get("label")
                 or (e.get("value") if tag == "input" and t in ("button", "submit", "reset") else "")
                 or e.get("alt") or ("" if tag == "input" else text_of(e)) or e.get("title") or e.get("placeholder") or "")


def actions_for(e, role):
    """Observed actions for one element, following snapshot.js (select -> one action per option; editable -> fill + open)."""
    tag = e.tag.lower()
    label = name_of(e) or role
    base = {"node": int(e.get("backend_node_id")), "role": role, "label": label}
    if role in ("checkbox", "radio") and e.get("input_checked") is not None:
        base["checked"] = str(e.get("input_checked")).lower() in ("true", "1", "checked")
    if tag == "select":
        out, opts = [], [o for o in e.iter("option")]
        cur = [clean(text_of(o)) for o in opts if o.get("option_selected") not in (None, "false")]
        for o in opts:
            if o.get("option_selected") not in (None, "false"):
                continue
            ol = clean(text_of(o)) or clean(o.get("value"))
            out.append({**base, "kind": "select", "value": o.get("value") or ol, "current_value": ", ".join(cur),
                        "label": label + " → " + ol, "_option": ol})
        return out
    editable = role in ("textbox", "searchbox", "spinbutton") or (role == "combobox" and tag in ("input", "textarea"))
    value = e.get("input_value") or (e.get("value") if tag in ("input", "textarea") else "") or ""
    if editable:
        return [{**base, "kind": "fill", "value": value}, {**base, "kind": "click", "value": value, "label": "Open " + label}]
    return [{**base, "kind": "click", "value": value}]


def page_text(root):
    words, n = [], 0
    for t in root.iter("text"):
        s = clean(t.text or "", 400)
        if s:
            words.append(s)
            n += len(s)
            if n >= 6000:
                break
    return "\n".join(words)[:6000]


def bbox_ok(cand):
    try:
        a = json.loads(cand["attributes"])
        x, y, w, h = (float(v) for v in a["bounding_box_rect"].split(","))
        return w > 0 and h > 0
    except Exception:
        return False


def parse_history(reprs):
    """'[textbox]  US City -> TYPE: 08817' -> Jev Ultrafast history entry."""
    out = []
    for r in reprs:
        m = re.match(r"\[(.*?)\]\s*(.*?)\s*->\s*(\w+)(?::\s*(.*))?$", r)
        if not m:
            continue
        _, label, op, val = m.groups()
        kind = {"TYPE": "fill", "SELECT": "select"}.get(op, "click")
        out.append({"action": clean(label) or "element", "kind": kind, "text": val if op == "TYPE" else None,
                    "page_changed": True})
    return out


def build_step(row, k, rng):
    """One Mind2Web step -> {state, goal, history, gold_ids}: the gold element plus k-1 sampled candidates, in page order."""
    from lxml import html as lhtml
    op = json.loads(row["operation"])
    gold_op = {"CLICK": "click", "TYPE": "fill", "SELECT": "select"}[op["op"]]
    pos = [json.loads(p) for p in row["pos_candidates"]]
    if not pos:
        return None, "no_pos"
    gold_id = int(pos[0]["backend_node_id"])
    root = lhtml.fromstring(row["cleaned_html"])
    by_id = {int(e.get("backend_node_id")): e for e in root.iter() if isinstance(e.tag, str) and e.get("backend_node_id")}
    if gold_id not in by_id:
        return None, "gold_not_in_html"
    cands = [json.loads(c) for c in row["neg_candidates"]]
    pool = []
    for c in cands:
        cid = int(c["backend_node_id"])
        e = by_id.get(cid)
        if e is None or not bbox_ok(c) or cid == gold_id:
            continue
        r = role_of(e)
        if r is None:
            if e.tag.lower() in ("html", "body", "form", "table", "tbody", "section", "header", "footer", "main"):
                continue
            nm = name_of(e)
            if not nm or len(nm) > 80:
                continue
            r = NONINTERACTIVE_ROLE.get(e.tag.lower(), "generic")
        pool.append((cid, r))
    ge = by_id[gold_id]
    grole = role_of(ge) or NONINTERACTIVE_ROLE.get(ge.tag.lower(), "generic")
    chosen = rng.sample(pool, min(k - 1, len(pool))) + [(gold_id, grole)]
    order = {int(e.get("backend_node_id")): i for i, e in enumerate(by_id.values())}
    chosen.sort(key=lambda x: order.get(x[0], 0))                       # page order: the gold position gives nothing away
    actions = []
    for cid, r in chosen:
        actions += actions_for(by_id[cid], r)
    gold = [a for a in actions if a["node"] == gold_id and a["kind"] == gold_op]
    if gold_op == "select":
        gold = [a for a in gold if norm(a.get("_option")) == norm(op["value"]) or norm(a.get("value")) == norm(op["value"])]
    if not gold:
        return None, f"gold_{gold_op}_not_available"
    actions = actions[:250]
    for i, a in enumerate(actions):
        a["id"] = f"e{i + 1}"
    gold_ids = [a["id"] for a in actions if a in gold]
    if not gold_ids:
        return None, "gold_truncated"
    actions += [{"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
                {"id": "wait", "kind": "wait", "label": "Wait for the page to update"}]
    idx = int(row["target_action_index"])
    state = {"url": f"https://www.{row['website']}.com/", "title": row["website"], "text": page_text(root),
             "actions": actions}
    return {"uid": row["action_uid"], "annotation_id": row["annotation_id"], "split": row["_split"],
            "website": row["website"], "domain": row["domain"], "goal": row["confirmed_task"], "step": idx,
            "gold_op": op["op"], "original_op": op["original_op"], "gold_ids": gold_ids, "gold_node": gold_id,
            "state": state, "history": parse_history(row["action_reprs"][:idx])}, "ok"


def cmd_build(a):
    import pyarrow.parquet as pq
    rng = random.Random(a.seed)
    out, reasons = [], {}
    for split in SPLITS:
        files = sorted(glob.glob(os.path.join(a.data_dir, f"{split}-*.parquet")))
        if not files:
            sys.exit(f"no {split}-*.parquet files in {a.data_dir}")
        rows = []
        for f in files:
            t = pq.read_table(f, columns=["action_uid", "cleaned_html", "operation", "pos_candidates", "neg_candidates",
                                          "website", "domain", "annotation_id", "confirmed_task", "action_reprs",
                                          "target_action_index"])
            rows += t.to_pylist()
        rng.shuffle(rows)
        got = 0
        for r in rows:
            if a.per_split and got >= a.per_split:
                break
            r["_split"] = split
            s, why = build_step(r, a.k, rng)
            reasons[why] = reasons.get(why, 0) + 1
            if s:
                out.append(s)
                got += 1
        print(split, "rows", len(rows), "kept", got, file=sys.stderr)
    with open(a.out, "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    print("built", len(out), "steps;", reasons, file=sys.stderr)


# --------------------------------------------------------------------------------------------------------------- run
def cmd_run(a):
    import httpx
    from jev_ultrafast import model as M
    key = os.environ.get(a.api_key_env, "") if a.api_key_env else "none"
    if a.api_key_env and not key:
        sys.exit(f"set {a.api_key_env}")
    os.environ.setdefault("TYPESAFE_API_KEY", "unused")          # read by choose(); the real key is passed below
    if a.model:
        os.environ["TYPESAFE_MODEL"] = a.model
    M.CLIENT = httpx.Client(http2=a.url.startswith("https://"), timeout=a.timeout)
    orig = M.post_json
    M.post_json = lambda _url, _key, body: orig(a.url, key, body)   # same request body; only the endpoint changes
    steps = read_steps(a.steps)
    if a.limit:
        steps = steps[:a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {r["uid"] for r in read_records(a.out) if "error" not in r}   # failed steps are retried on resume
    todo = [s for s in steps if s["uid"] not in done]
    lock = threading.Lock()
    print(f"{len(todo)} steps to run ({len(done)} already in {a.out})", file=sys.stderr)

    def one(s):
        rec = {"uid": s["uid"]}
        for attempt in range(3):
            try:
                d = M.choose(s["state"], s["goal"], s["history"])
                rec.update(choice=d["choice"], operation=d["operation"], target=d["target"], probabilities=d["probabilities"],
                           confidence=d["confidence"], operation_probabilities=d["operation_probabilities"],
                           target_probabilities=d["target_probabilities"], model=d["model"], latency_ms=d["latency_ms"])
                rec.pop("error", None)
                break
            except Exception as e:                                   # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
                transient = "connection failed" in str(e) or "unavailable" in str(e) or \
                    any(f"HTTP {c}" in str(e) for c in (408, 429, 500, 502, 503, 504, 529))
                if not transient:                                    # a rejected or invalid answer is final: no re-tries
                    break
                time.sleep(2 * (attempt + 1))
        with lock, open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    t0, n, failed = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(one, todo):
            n += 1
            failed += "error" in rec
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", file=sys.stderr)
    print(f"done: {n} steps, {failed} failed (failed steps count as wrong; re-run the same command to retry them)", file=sys.stderr)


# --------------------------------------------------------------------------------------------------- score / compare
def read_steps(path):
    """Steps from a .jsonl file, or from the password-protected .zip it ships in."""
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".jsonl"))
            lines = z.read(name, pwd=b"mind2web").decode().splitlines()
    else:
        lines = open(path).read().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_steps(path):
    return {s["uid"]: s for s in read_steps(path)}


def read_records(path):
    """Prediction records; a truncated last line (killed run) is ignored."""
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


INTERACTIVE = set(ROLES)


def unnamed_gold(s):
    """True when the gold element is non-interactive and unnamed (or its name is over 80 characters). The builder filters
    such elements out of the sampled candidates but not the gold one, so on these steps the answer can stand out."""
    g = next(a for a in s["state"]["actions"] if a.get("node") == s["gold_node"])
    return g["role"] not in INTERACTIVE and (g["label"] == g["role"] or len(g["label"]) > 80)


def judge(steps, path):
    """uid -> (step_ok, element_ok, op_ok, error). Missing or failed requests count as wrong."""
    recs = {}
    for r in read_records(path):
        if r.get("uid") in steps and ("error" not in r or r["uid"] not in recs):
            recs[r["uid"]] = r
    out = {}
    for uid, s in steps.items():
        r = recs.get(uid, {"error": "missing"})
        ch = None if "error" in r else r.get("choice")
        act = next((x for x in s["state"]["actions"] if x["id"] == ch), None)
        out[uid] = (ch in s["gold_ids"], act is not None and act.get("node") == s["gold_node"],
                    r.get("operation") == GOLD_OP[s["gold_op"]] and "error" not in r, "error" in r)
    return out


def groups(steps):
    yield "test_task", [u for u, s in steps.items() if s["split"] == "test_task"]
    yield "test_website", [u for u, s in steps.items() if s["split"] == "test_website"]
    yield "test_domain", [u for u, s in steps.items() if s["split"] == "test_domain"]
    yield "unseen (website+domain)", [u for u, s in steps.items() if s["split"] != "test_task"]
    yield "ALL", list(steps)
    yield "ALL excl. unnamed gold", [u for u, s in steps.items() if not unnamed_gold(s)]


def cmd_score(a):
    steps = load_steps(a.steps)
    print(f"{'file':38s} {'set':25s} {'n':>5s} {'step':>6s} {'elem':>6s} {'op':>6s} {'err':>4s}")
    for p in a.preds:
        j = judge(steps, p)
        for name, us in groups(steps):
            n = len(us)
            pct = lambda i: 100 * sum(j[u][i] for u in us) / n
            print(f"{os.path.basename(p):38s} {name:25s} {n:5d} {pct(0):6.1f} {pct(1):6.1f} {pct(2):6.1f} {sum(j[u][3] for u in us):4d}")


def cmd_compare(a):
    """Paired difference in step accuracy, first file vs each other file, with a bootstrap 95% CI over steps."""
    steps = load_steps(a.steps)
    A = judge(steps, a.preds[0])
    for bp in a.preds[1:]:
        B = judge(steps, bp)
        print(f"== {os.path.basename(a.preds[0])} − {os.path.basename(bp)}")
        for name, us in groups(steps):
            d = [int(A[u][0]) - int(B[u][0]) for u in us]
            n = len(d)
            rng = random.Random(0)
            bs = sorted(sum(rng.choice(d) for _ in range(n)) / n * 100 for _ in range(2000))
            print(f"  {name:25s} n={n:5d}  {sum(A[u][0] for u in us) / n * 100:5.1f} vs {sum(B[u][0] for u in us) / n * 100:5.1f}"
                  f"  diff {sum(d) / n * 100:+5.1f} [{bs[50]:+.1f}, {bs[1949]:+.1f}]"
                  f"  only-first {sum(x == 1 for x in d)}  only-second {sum(x == -1 for x in d)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a step set from Multimodal-Mind2Web parquet files")
    b.add_argument("--data-dir", required=True, help="folder with test_task-*.parquet, test_website-*, test_domain-*")
    b.add_argument("--out", required=True)
    b.add_argument("--k", type=int, default=50, help="candidate elements per step (gold + k-1 sampled)")
    b.add_argument("--per-split", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run", help="send every step to a decision endpoint that speaks the /v1/systemone request shape")
    r.add_argument("--steps", required=True)
    r.add_argument("--url", required=True, help="e.g. https://api.typesafe.ai/v1/systemone or http://127.0.0.1:8080/v1/systemone")
    r.add_argument("--api-key-env", default=None, help="environment variable holding the API key, if the endpoint needs one")
    r.add_argument("--model", default=None, help="model id sent in the request (e.g. jev-1.14.0)")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--limit", type=int, default=0, help="only the first N steps (smoke test)")
    for name in ("score", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--steps", required=True)
        s.add_argument("preds", nargs="+")
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "score": cmd_score, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
