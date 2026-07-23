"""
score_predictions.py
====================
Pure-CPU scorer for ASR predictions. NO model load, NO GPU, NO network.

Reads a predictions file and reports, each EXPLICITLY LABELED:
  * corpus-level WER and CER  (aggregate: total edits / total reference length)
  * macro-level  WER and CER  (unweighted mean of per-sample rates)
  * sample count n and total reference word count
  * seeded bootstrap 95% CIs (2000 resamples) for all four metrics
  * a drop count (records with a missing/failed hypothesis), with reasons

Why both statistics: the published 3.6%/2.5% figure is CORPUS-level; a prior version
of the evaluation script silently switched to MACRO. Emitting both, always, means the
headline figure and any future figure are never confused for one another again.

Accepted input schemas (auto-detected):
  * JSONL from generate_test_predictions.py : {"reference","hypothesis",...} per line
  * legacy output/evaluation_results.json   : [{"ground_truth","prediction",...}, ...]

--------------------------------------------------------------------------------
NORMALIZATION (pinned; applied IDENTICALLY to reference and hypothesis)
--------------------------------------------------------------------------------
  1. RemoveMultipleSpaces      collapse runs of whitespace to a single space
  2. Strip                     trim leading / trailing whitespace
  3. ReduceToListOf...Words    (WER) tokenize on whitespace into words
     ReduceToListOf...Chars    (CER) tokenize into characters (spaces included)

  NO case folding. NO punctuation stripping. This reproduces the provenance of the
  published number: the commit-115b6c4 evaluator scored the RAW strings. SpeechBrain
  references and this model's output are both uppercase, so casing is a no-op here;
  hypothesis punctuation (apostrophes, hyphen/space splits) is counted, exactly as it
  was when 3.6%/2.5% was produced. jiwer's bare defaults yield identical numbers on
  this data; the explicit transforms below PIN the behavior against future jiwer
  default drift. Do not "improve" this normalization without relabeling every figure
  it has ever produced.
"""

import argparse
import json
import os
import sys

import jiwer
import numpy as np

# --- pinned transforms (see module docstring) ---------------------------------
_WORD_TX = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])
# jiwer renamed the char reducer across majors; support both.
_CharReducer = getattr(jiwer, "ReduceToListOfListOfChars",
                       getattr(jiwer, "ReduceToListOfChars", None))
_CHAR_TX = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    _CharReducer(),
])

FAILED_MARKERS = {"[INFERENCE_FAILED]", "", None}


def load_pairs(path):
    """Return (pairs, drops). pairs = list of (id, reference, hypothesis).
    drops = list of (id, reason) for records with no usable hypothesis."""
    if path.endswith(".jsonl"):
        records = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    else:
        with open(path) as fh:
            records = json.load(fh)

    pairs, drops = [], []
    for i, r in enumerate(records):
        if r.get("_meta"):          # skip the JSONL provenance header line
            continue
        rid = r.get("id", f"row_{i}")
        ref = r.get("reference", r.get("ground_truth"))
        hyp = r.get("hypothesis", r.get("prediction"))
        if ref is None:
            drops.append((rid, "missing reference"))
            continue
        if hyp in FAILED_MARKERS:
            drops.append((rid, f"missing/failed hypothesis ({hyp!r})"))
            continue
        pairs.append((rid, ref, hyp))
    return pairs, drops


def per_sample_counts(pairs):
    """Per-sample (word_errors, word_reflen, char_errors, char_reflen, wer, cer)."""
    rows = []
    for _rid, ref, hyp in pairs:
        w = jiwer.process_words(ref, hyp, reference_transform=_WORD_TX,
                                hypothesis_transform=_WORD_TX)
        c = jiwer.process_characters(ref, hyp, reference_transform=_CHAR_TX,
                                     hypothesis_transform=_CHAR_TX)
        w_ref = w.substitutions + w.deletions + w.hits          # |reference| in words
        w_err = w.substitutions + w.deletions + w.insertions    # S+D+I
        c_ref = c.substitutions + c.deletions + c.hits
        c_err = c.substitutions + c.deletions + c.insertions
        rows.append((w_err, w_ref, c_err, c_ref, w.wer, c.cer))
    return np.array(rows, dtype=float)


def corpus(errs, reflens):
    total_ref = reflens.sum()
    return float(errs.sum() / total_ref) if total_ref else 0.0


def bootstrap_ci(rows, seed=1234, resamples=2000):
    """Seeded percentile 95% CIs for corpus & macro WER/CER by resampling samples."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    w_err, w_ref, c_err, c_ref, wer_i, cer_i = (rows[:, k] for k in range(6))
    cw, mw, cc, mc = [], [], [], []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        cw.append(w_err[idx].sum() / max(w_ref[idx].sum(), 1e-9))
        cc.append(c_err[idx].sum() / max(c_ref[idx].sum(), 1e-9))
        mw.append(wer_i[idx].mean())
        mc.append(cer_i[idx].mean())
    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"corpus_wer": ci(cw), "macro_wer": ci(mw),
            "corpus_cer": ci(cc), "macro_cer": ci(mc)}


def main():
    ap = argparse.ArgumentParser(description="Pure-CPU corpus+macro WER/CER scorer.")
    ap.add_argument("predictions", help="path to .jsonl or legacy .json predictions")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--json-out", default=None, help="optional path to write metrics JSON")
    args = ap.parse_args()

    pairs, drops = load_pairs(args.predictions)
    if not pairs:
        sys.exit("No scorable pairs found.")
    rows = per_sample_counts(pairs)
    w_err, w_ref, c_err, c_ref, wer_i, cer_i = (rows[:, k] for k in range(6))

    corpus_wer = corpus(w_err, w_ref)
    corpus_cer = corpus(c_err, c_ref)
    macro_wer = float(wer_i.mean())
    macro_cer = float(cer_i.mean())
    ci = bootstrap_ci(rows, seed=args.seed, resamples=args.resamples)

    n = len(pairs)
    total_ref_words = int(w_ref.sum())

    def pct(x):
        return f"{x*100:.2f}%"
    def band(key):
        lo, hi = ci[key]
        return f"[{lo*100:.2f}%, {hi*100:.2f}%]"

    print("=" * 72)
    print(f"SCORED: {args.predictions}")
    print(f"n (scored samples): {n}    dropped: {len(drops)}    "
          f"total reference words: {total_ref_words}")
    print(f"bootstrap: {args.resamples} resamples, seed={args.seed}")
    print("-" * 72)
    print(f"CORPUS WER : {pct(corpus_wer):>8}   95% CI {band('corpus_wer')}")
    print(f"MACRO  WER : {pct(macro_wer):>8}   95% CI {band('macro_wer')}")
    print(f"CORPUS CER : {pct(corpus_cer):>8}   95% CI {band('corpus_cer')}")
    print(f"MACRO  CER : {pct(macro_cer):>8}   95% CI {band('macro_cer')}")
    print("-" * 72)
    print("HEADLINE statistic = CORPUS (matches the published-figure provenance).")
    if drops:
        print("-" * 72)
        print(f"DROPPED {len(drops)} record(s) — NOT scored, NOT in any denominator:")
        for rid, reason in drops:
            print(f"  ! {rid}: {reason}")
    print("=" * 72)

    if args.json_out:
        metrics = {
            "predictions_file": os.path.abspath(args.predictions),
            "n_scored": n, "n_dropped": len(drops),
            "total_reference_words": total_ref_words,
            "normalization": "RemoveMultipleSpaces+Strip+tokenize; no case/punct change",
            "bootstrap": {"resamples": args.resamples, "seed": args.seed},
            "corpus_wer": corpus_wer, "macro_wer": macro_wer,
            "corpus_cer": corpus_cer, "macro_cer": macro_cer,
            "ci_95": ci,
            "drops": [{"id": r, "reason": why} for r, why in drops],
        }
        with open(args.json_out, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"metrics JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
