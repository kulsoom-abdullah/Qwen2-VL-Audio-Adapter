"""
05_evaluate.py  — SUPERSEDED (deprecation shim)
===============================================
This script has been split into two, and must not be used to produce a number:

  * scripts/generate_test_predictions.py  — inference only -> per-sample JSONL,
                                             with a pre-flight that aborts if the
                                             audio pathway is not provably live.
  * scripts/score_predictions.py          — pure CPU -> corpus AND macro WER/CER,
                                             each labeled, with bootstrap CIs.

The fork insert is restored below (pattern copied verbatim from
scripts/generate_audit_batch.py:26-32) so that importing this module is harmless, but
the script refuses to run and points you at the replacement flow.
"""

import os
import sys

# Add Fork (CRITICAL - your model needs custom transformers code!)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FORK_PATH = os.path.join(PROJECT_ROOT, "transformers_fork", "src")
if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)
    print(f"✅ Using transformers fork from: {FORK_PATH}")
else:
    print("⚠️  Transformers fork not found - model may fail to load!")

_MESSAGE = """
scripts/05_evaluate.py is superseded. Run the two-step flow instead:

  # 1) inference -> JSONL (on the GPU pod)
  python scripts/generate_test_predictions.py \\
      --test-json data/audit_test/test.json \\
      --output    output/test_predictions.jsonl \\
      --dataset-revision 0e84cdb9e4b826afaabca5d33ec9453b11aacef3

  # 2) score -> corpus AND macro WER/CER with CIs (pure CPU, offline)
  python scripts/score_predictions.py output/test_predictions.jsonl

To reproduce the published 50-sample validation figure from committed predictions:
  python scripts/score_predictions.py output/evaluation_results.json
"""

if __name__ == "__main__":
    sys.stderr.write(_MESSAGE)
    sys.exit(2)
