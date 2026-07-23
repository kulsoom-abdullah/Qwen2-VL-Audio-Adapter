"""
generate_test_predictions.py
============================
INFERENCE ONLY. Streams per-sample predictions to JSONL. NO metrics, NO scoring.
Score the output separately with scripts/score_predictions.py (pure CPU).

This is the generation half of the split of the old scripts/05_evaluate.py. It exists
because the old evaluator (a) computed a macro statistic mislabeled against a corpus
figure, (b) read the validation split while claiming "test", and (c) dropped the
transformers-fork sys.path insert, which would silently load stock Qwen2-VL and ignore
`input_features` — measuring text priors, not audio. The pre-flight assertions below
abort BEFORE any generation if the audio pathway is not provably live.

Durability (this run can be multi-hour on the full 8,087-row split):
  * each successful record is written and fsync'd IMMEDIATELY (never buffered to the end);
  * --resume reads completed ids from an existing output and skips them, so a killed or
    crashed run continues instead of restarting;
  * without --resume the script refuses to touch an existing output (no silent overwrite).

Smoke-test first (required before the full run):
    python scripts/generate_test_predictions.py --limit 50 \
        --output output/test_predictions_smoke.jsonl
  Confirm pre-flight passes, transcripts are sane, and note the reported seconds/sample.

Full run:
    python scripts/generate_test_predictions.py \
        --test-json data/audit_test/test.json \
        --output    output/test_predictions.jsonl        # --resume to continue
"""

import argparse
import inspect
import json
import os
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone

import torch
import librosa
from tqdm import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    WhisperFeatureExtractor,
)
import transformers

# --- SMART PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Add Fork (CRITICAL - your model needs custom transformers code!)
# Copied verbatim from the pattern in scripts/generate_audit_batch.py:26-32,
# which every other model-loading script in this repo uses. Its absence is exactly
# the defect this script's pre-flight refuses to let pass.
FORK_PATH = os.path.join(PROJECT_ROOT, "transformers_fork", "src")
if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)
    print(f"✅ Using transformers fork from: {FORK_PATH}")
else:
    print("⚠️  Transformers fork not found - model may fail to load!")

# Pinned dataset revision (parquet-metadata verified; see PRE_REGISTRATION.md)
DEFAULT_DATASET_REVISION = "0e84cdb9e4b826afaabca5d33ec9453b11aacef3"

# Constants (identical across train collator and both eval scripts)
AUDIO_TOKEN_ID = 151657
NUM_AUDIO_TOKENS = 1500
PAD_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645

# Fixed, reference-free prompt halves (must match 04_train_stage2.py:197-198).
PROMPT_PREFIX = "<|im_start|>user\n<|audio_bos|>"
PROMPT_SUFFIX = "<|audio_eos|>\nTranscribe this audio.<|im_end|>\n<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Pre-flight: abort BEFORE generation if the audio pathway is not provably live.
# ---------------------------------------------------------------------------
def preflight_model(model):
    reasons = []

    # (1) The loaded model class must come from the fork, not stock transformers
    #     and not a trust-remote-code cache. A stock load passes weight checks and is
    #     still audio-dead, so this is the load-time guard that actually matters.
    mod = sys.modules.get(type(model).__module__)
    mod_file = os.path.abspath(getattr(mod, "__file__", "") or "")
    if not mod_file.startswith(os.path.abspath(FORK_PATH)):
        reasons.append(f"model class loaded from {mod_file!r}, NOT under the fork "
                       f"{os.path.abspath(FORK_PATH)!r} — audio pathway would be dead")

    # (2) forward() must accept input_features (the grafted audio entry point)
    if "input_features" not in inspect.signature(model.forward).parameters:
        reasons.append("model.forward has no `input_features` parameter — "
                       "not the grafted architecture")

    # (3) audio encoder + projector present, not on meta, non-zero parameter norm
    for name in ("audio_encoder", "audio_projector"):
        sub = getattr(model, name, None)
        if sub is None:
            reasons.append(f"model has no `{name}`")
            continue
        params = list(sub.parameters())
        if not params:
            reasons.append(f"`{name}` has no parameters")
            continue
        if any(p.is_meta for p in params):
            reasons.append(f"`{name}` has parameters on the meta device (not loaded)")
            continue
        total_norm = float(sum(p.detach().float().norm() ** 2 for p in params) ** 0.5)
        if total_norm == 0.0:
            reasons.append(f"`{name}` has zero parameter norm (uninitialized/zeroed)")

    # (4) logging not suppressed below WARNING (would hide a load failure). This
    #     script deliberately installs NO warnings.filterwarnings / verbosity change.
    if transformers.logging.get_verbosity() > transformers.logging.WARNING:
        reasons.append("transformers logging verbosity is above WARNING — "
                       "load failures could be hidden")

    if reasons:
        print("\n❌ PRE-FLIGHT FAILED — refusing to generate:")
        for r in reasons:
            print(f"   - {r}")
        sys.exit(1)
    print("✅ Pre-flight passed: fork in use, input_features accepted, "
          "audio encoder+projector loaded & non-zero, logging not suppressed.")


def build_prompt(tokenizer, device):
    p1 = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False, return_tensors="pt").to(device)
    p2 = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False, return_tensors="pt").to(device)
    audio = torch.tensor([[AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS], device=device)
    return torch.cat([p1, audio, p2], dim=1)


def assert_input_clean(tokenizer, input_ids, reference):
    """Per-sample guard: audio tokens present AND reference NOT in the prompt."""
    if AUDIO_TOKEN_ID not in input_ids[0].tolist():
        raise AssertionError("prompt contains no audio tokens")
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    ref = (reference or "").strip()
    if ref and (ref in decoded or ref[:24] in decoded):
        raise AssertionError("reference transcript leaked into the prompt")


def read_completed_ids(path):
    """Return the set of ids already present in an existing JSONL (tolerant of a
    truncated final line from a killed run)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial trailing line from a crash — ignore
            if not rec.get("_meta") and "id" in rec:
                done.add(rec["id"])
    return done


def main():
    ap = argparse.ArgumentParser(description="Inference-only prediction generator (streaming, resumable).")
    ap.add_argument("--test-json", default=os.path.join(PROJECT_ROOT, "data", "audit_test", "test.json"))
    ap.add_argument("--output", default=os.path.join(PROJECT_ROOT, "output", "test_predictions.jsonl"))
    ap.add_argument("--model", default="kulsoom-abdullah/Qwen2-Audio-7B-Transcription")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of samples (smoke-test); default = FULL partition")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing output: skip ids already written")
    ap.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION,
                    help="recorded in output metadata for provenance; does not fetch")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    meta_path = args.output + ".meta.json"

    # Overwrite policy: never silently clobber; resume appends.
    if os.path.exists(args.output) and not args.resume:
        sys.exit(f"❌ Output exists, refusing to overwrite (use --resume): {args.output}")
    completed = read_completed_ids(args.output) if args.resume else set()
    if args.resume:
        print(f"↻ Resuming: {len(completed)} record(s) already present, will skip them.")

    data_dir = os.path.dirname(os.path.abspath(args.test_json))
    if not os.path.exists(args.test_json):
        sys.exit(f"❌ Test data not found: {args.test_json}  (run setup_audit_data.py)")
    with open(args.test_json) as fh:
        data = json.load(fh)
    if args.limit is not None:
        data = data[:args.limit]
    todo = [e for e in data if e.get("id") not in completed]
    print(f"📊 {len(data)} samples in scope; {len(todo)} to generate "
          f"({len(data) - len(todo)} already done).")

    print(f"📥 Loading model: {args.model}")
    with warnings.catch_warnings(record=True) as caught:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    for w in caught:
        print(f"   ⚠️ load warning: {w.category.__name__}: {w.message}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token_id = PAD_TOKEN_ID
    tokenizer.eos_token_id = EOS_TOKEN_ID
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    preflight_model(model)

    drops, per_sample_secs = [], []
    written = 0
    print("🚀 Generating (inference only, flushing each record)...")
    # Line-buffered append; each record is also fsync'd so a kill loses at most the
    # sample in flight, and --resume picks up from there.
    with open(args.output, "a", buffering=1) as out:
        for entry in tqdm(todo):
            rid = entry.get("id", f"sample_{len(completed) + written}")
            audio_rel = entry["audio"]
            audio_abs = os.path.join(data_dir, audio_rel)
            reference = entry.get("ground_truth", entry.get("reference"))
            t0 = time.time()
            try:
                y, _ = librosa.load(audio_abs, sr=16000, mono=True)
                feats = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
                input_features = feats.input_features.to(model.device).to(torch.bfloat16)

                input_ids = build_prompt(tokenizer, model.device)
                assert_input_clean(tokenizer, input_ids, reference)
                attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids, input_features=input_features,
                        attention_mask=attention_mask, max_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=PAD_TOKEN_ID, eos_token_id=EOS_TOKEN_ID,
                    )
                hypothesis = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

                out.write(json.dumps({
                    "id": rid,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "audio_path": audio_rel,
                    "n_ref_words": len((reference or "").split()),
                }) + "\n")
                out.flush()
                os.fsync(out.fileno())
                written += 1
                per_sample_secs.append(time.time() - t0)
            except Exception as e:  # a drop is a finding, not a footnote
                drops.append({"id": rid, "audio_path": audio_rel,
                              "reason": f"{type(e).__name__}: {e}"})
                print(f"\n   ⚠️ DROPPED {rid}: {type(e).__name__}: {e}")

    mean_s = statistics.mean(per_sample_secs) if per_sample_secs else 0.0
    med_s = statistics.median(per_sample_secs) if per_sample_secs else 0.0
    total_split = len(data) if args.limit is None else "?"
    with open(meta_path, "w") as fh:
        json.dump({
            "model": args.model,
            "test_json": os.path.abspath(args.test_json),
            "dataset_revision": args.dataset_revision,
            "n_in_scope": len(data),
            "n_written_this_run": written,
            "n_total_in_output": len(completed) + written,
            "n_dropped_this_run": len(drops),
            "drops": drops,
            "mean_sec_per_sample": round(mean_s, 3),
            "median_sec_per_sample": round(med_s, 3),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "note": "inference only; score with scripts/score_predictions.py",
        }, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"WROTE {written} new record(s) -> {args.output}  "
          f"(total in file: {len(completed) + written})")
    print(f"DROPPED {len(drops)} sample(s) this run" + (" — see meta.drops" if drops else ""))
    print(f"TIMING: mean {mean_s:.2f}s/sample, median {med_s:.2f}s/sample")
    if args.limit is not None and mean_s:
        proj = mean_s * 8087 / 3600.0
        print(f"PROJECTED full 8,087-row run ≈ {proj:.1f} h at this rate "
              f"(confirm before launching the full run)")
    print(f"metadata -> {meta_path}")
    print("Score with:  python scripts/score_predictions.py " + args.output)
    print("=" * 72)


if __name__ == "__main__":
    main()
