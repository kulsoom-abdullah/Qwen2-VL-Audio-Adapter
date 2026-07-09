"""
summarize.py
============
Recomputes ALL audit counts from cases.json -- no hand-tallied numbers anywhere.

Modes:
  python3 summarize.py            adjudication progress + reconciliation table
  python3 summarize.py --final    strict gate; refuses unless adjudication complete
  python3 summarize.py --final --emit-typology
                                  additionally writes audit_typology.md

Disposition semantics (per Kulsoom's schema ruling, 2026-07-07):
  * original_disposition is the FROZEN record of the original audit -- never
    mutated, and always reported as its own reconciliation column.
  * adjudicated_disposition (nullable) overrides it for all recomputed counts.
  * effective disposition = adjudicated_disposition or original_disposition.

Gate policy:
  * Structural breaks (missing/duplicate rows, unknown codes, subcategory from
    the wrong family, nulls in --final mode) -> exit 1.
  * Deltas vs the PUBLISHED tally -> printed as FINDINGS, never an error. The
    published 37/36/14/11 summed to 98/100; the committed tracker covered 62/63
    rows; adjudication may move counts further (e.g. row 60 -> model error 15).
    Every such delta is reported, not hidden.
"""

import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_JSON = os.path.join(HERE, "cases.json")
TYPOLOGY_MD = os.path.join(HERE, "audit_typology.md")

# Published tally: README / HF card / notebook cell 17 of the original audit.
# Row 60 (the 63rd disagreement) was absent from every published breakdown.
PUBLISHED = {
    "perfect_match": 37,
    "label_wrong_model_right": 36,
    "true_model_error": 14,
    "ambiguous": 11,
    "normalization_only": 1,
}

FAMILY = {"label_wrong_model_right": "A", "true_model_error": "B"}
VALID_SUB = {
    "A": {"A1", "A2", "A3", "A4", "A5", "A6"},
    "B": {"B1", "B2", "B3", "B4", "B5"},
}
KNOWN_DISPOSITIONS = {"label_wrong_model_right", "true_model_error",
                      "ambiguous", "normalization_only", "uncategorized"}
DISPLAY_ORDER = ["label_wrong_model_right", "true_model_error", "ambiguous",
                 "normalization_only", "uncategorized"]


def effective_disposition(c):
    return c.get("adjudicated_disposition") or c["original_disposition"]


def load():
    with open(CASES_JSON) as fh:
        return json.load(fh)


def recompute(payload):
    cases = payload["cases"]
    original = Counter(c["original_disposition"] for c in cases)
    final = Counter(effective_disposition(c) for c in cases)
    sub = defaultdict(Counter)
    for c in cases:
        if c["subcategory"]:
            sub[effective_disposition(c)][c["subcategory"]] += 1
    return original, final, sub


def structural_errors(payload, final_mode):
    cases = payload["cases"]
    errs = []
    rows = [c["row"] for c in cases]
    if sorted(rows) != list(range(1, 64)):
        errs.append(f"rows are not exactly 1..63 (n={len(rows)}, dupes/gaps present)")
    if len({c["sample_id"] for c in cases}) != len(cases):
        errs.append("sample_ids are not unique")
    for c in cases:
        adj = c.get("adjudicated_disposition")
        if adj is not None:
            if adj not in KNOWN_DISPOSITIONS or adj == "uncategorized":
                errs.append(f"row {c['row']}: adjudicated_disposition {adj!r} invalid")
        eff = effective_disposition(c)
        fam = FAMILY.get(eff)
        s = c["subcategory"]
        if s:
            if fam is None:
                errs.append(f"row {c['row']}: subcategory {s} set but effective "
                            f"disposition '{eff}' is not A/B")
            elif s not in VALID_SUB[fam]:
                errs.append(f"row {c['row']}: subcategory {s} not valid for family {fam}")
        if final_mode:
            if fam and not s:
                errs.append(f"row {c['row']}: {eff} still unadjudicated")
            if eff == "ambiguous" and not (c.get("notes") or "").strip():
                errs.append(f"row {c['row']}: ambiguous case missing one-line reason in notes")
            if eff == "uncategorized":
                errs.append(f"row {c['row']}: still 'uncategorized' -- adjudicate it")
    return errs


def reconciliation(payload, original, final):
    perfect = payload["meta"]["perfect_match_count"]
    print("\nRECONCILIATION - published / original-committed / final-adjudicated")
    print("(deltas are FINDINGS to report, not errors to hide)")
    hdr = f"{'disposition':28s} {'published':>9s} {'original':>9s} {'adjudicated':>11s} {'delta':>6s}"
    print(hdr)
    findings = []
    pub_total, orig_total, fin_total = 0, perfect, perfect
    print(f"{'perfect_match':28s} {PUBLISHED['perfect_match']:>9d} {perfect:>9d} {perfect:>11d} {'0':>6s}")
    pub_total += PUBLISHED["perfect_match"]
    for key in DISPLAY_ORDER:
        pub = PUBLISHED.get(key)
        o, f = original.get(key, 0), final.get(key, 0)
        orig_total += o
        fin_total += f
        if pub is None:
            delta = ""
            if o or f:
                findings.append(f"{key}: {o} in the original committed audit, "
                                f"{f} after adjudication -- absent from every "
                                f"published tally")
        else:
            pub_total += pub
            delta = f"{f - pub:+d}" if f != pub else "0"
            if f != pub:
                findings.append(f"{key}: final-adjudicated {f} != published {pub}"
                                + (f" (original committed: {o})" if o != f else ""))
        print(f"{key:28s} {('-' if pub is None else str(pub)):>9s} {o:>9d} {f:>11d} {delta:>6s}")
    print(f"{'TOTAL':28s} {pub_total:>9d} {orig_total:>9d} {fin_total:>11d} {fin_total - pub_total:+6d}")
    if pub_total != 100:
        findings.append(f"published categories sum to {pub_total}/100 -- the original "
                        f"tally under-covered the audit: row 60 was never categorized "
                        f"(the four-category 37/36/14/11 version summed to 98, also "
                        f"omitting the normalization row)")
    print("\nFINDINGS:")
    for f in findings or ["none - final-adjudicated matches published exactly"]:
        print(f"  * {f}")
    return findings


def emit_typology(payload, final, sub, findings):
    cases = payload["cases"]
    meta = payload["meta"]
    malta_cases = [c for c in cases
                   if "malta" in (c["prediction_text"] + c["reference_text"]).casefold()]
    malta_ids = {c["sample_id"] for c in malta_cases}

    by_subcat = defaultdict(list)
    for c in cases:
        if c["subcategory"]:
            by_subcat[c["subcategory"]].append(c)
    for v in by_subcat.values():
        v.sort(key=lambda c: (c["sample_id"] not in malta_ids, -c["wer_pct"]))

    L = []
    add = L.append
    add("# Label-Noise Audit Typology")
    add("")
    add(f"**N = {meta['total_audited']} audited samples** from the SpeechBrain test "
        f"partition: {meta['perfect_match_count']} perfect matches (count from the "
        f"original run's saved output) and {meta['disagreement_count']} disagreements, "
        f"each individually recoverable from the committed notebook and adjudicated "
        f"below. Every count in this file is recomputed from "
        f"`analysis/label_audit/cases.json` by `summarize.py`; nothing is hand-tallied. "
        f"Counts use the FINAL adjudicated dispositions; the frozen original audit "
        f"and the published tally are reconciled at the end of this file.")
    add("")
    add("## Dispositions (final adjudicated)")
    add("")
    add("| Disposition | Count |")
    add("|---|---|")
    add(f"| Perfect match | {meta['perfect_match_count']} |")
    for key in DISPLAY_ORDER:
        if final.get(key):
            add(f"| {key} | {final[key]} |")
    add("")
    tax = meta["taxonomy"]
    for disp_key, title in (("label_wrong_model_right", "A - label wrong, model right"),
                            ("true_model_error", "B - true model errors")):
        add(f"## {title} (N={final.get(disp_key, 0)})")
        add("")
        add("| Code | Meaning | Count | Exemplar sample IDs |")
        add("|---|---|---|---|")
        for code, meaning in tax[disp_key].items():
            n = sub[disp_key].get(code, 0)
            ex = ", ".join(f"`{c['sample_id']}`" for c in by_subcat.get(code, [])[:3])
            add(f"| {code} | {meaning} | {n} | {ex or '-'} |")
        add("")
    amb = [c for c in cases if effective_disposition(c) == "ambiguous"]
    if amb:
        add(f"## C - ambiguous (N={len(amb)})")
        add("")
        for c in sorted(amb, key=lambda c: c["row"]):
            add(f"- `{c['sample_id']}` (row {c['row']}): "
                f"{(c.get('notes') or '').strip() or '(reason pending)'}")
        add("")
    if malta_cases:
        m = malta_cases[0]
        add(f"## The 'Malta' case (`{m['sample_id']}`, row {m['row']})")
        add("")
        add(f"Adjudicated **{m['subcategory'] or 'pending'} — fluent semantic "
            f"substitution** (original disposition: {m['original_disposition']}; "
            f"the deterministic rule had suggested "
            f"{m['provisional_subcategory']}). The output is audio-grounded in "
            f"later clip context — 'Malta' is spoken later in the same 30-second "
            f"window — not fabrication. It exemplifies the same full-clip "
            f"contextual mechanism as the A-category corrections, operating past "
            f"faithfulness rather than toward it. A4 remains reserved for cases "
            f"where the transcriber misheard an entity the speaker actually said.")
        add("")
    add("## Limitations")
    add("")
    add(f"- {meta['perfect_match_note']}")
    add("- Row numbers refer to the WER-sorted disagreement table in the committed "
        "notebook output (cell 14); the mapping row -> sample_id is preserved in "
        "cases.json.")
    add("- The original published tally (37/36/14/11) summed to 98/100: one "
        "normalization-only row and one never-categorized row (row 60) were "
        "omitted. Row 60 is adjudicated here (see reconciliation); "
        "normalization-only remains its own line, faithful to the original audit.")
    add("")
    add("## Reconciliation findings")
    add("")
    for f in findings or ["none - final-adjudicated matches published exactly"]:
        add(f"- {f}")
    add("")
    with open(TYPOLOGY_MD, "w") as fh:
        fh.write("\n".join(L))
    print(f"\n📄 wrote {TYPOLOGY_MD}")


def main():
    final_mode = "--final" in sys.argv
    payload = load()
    original, final, sub = recompute(payload)

    n_ab = sum(1 for c in payload["cases"] if effective_disposition(c) in FAMILY)
    n_adj = sum(1 for c in payload["cases"] if c["subcategory"])
    print(f"adjudication progress: {n_adj}/{n_ab} A/B cases have a final subcategory")

    print("\nSUBCATEGORY SUMS vs PARENT DISPOSITIONS (gate; parents = final adjudicated)")
    ok = True
    for disp_key in ("label_wrong_model_right", "true_model_error"):
        parent = final.get(disp_key, 0)
        total = sum(sub[disp_key].values())
        status = "OK" if total == parent else ("incomplete" if not final_mode else "FAIL")
        if final_mode and total != parent:
            ok = False
        print(f"  {disp_key:28s} parent={parent:3d}  subcategorized={total:3d}  [{status}]")
        for code, n in sorted(sub[disp_key].items()):
            print(f"      {code}: {n}")

    findings = reconciliation(payload, original, final)

    errs = structural_errors(payload, final_mode)
    if errs:
        print("\nSTRUCTURAL ERRORS (gate -> nonzero exit):")
        for e in errs:
            print(f"  ! {e}")
    if errs or (final_mode and not ok):
        sys.exit(1)

    if "--emit-typology" in sys.argv:
        if not final_mode:
            sys.exit("refusing to emit typology without --final gate")
        emit_typology(payload, final, sub, findings)


if __name__ == "__main__":
    main()
