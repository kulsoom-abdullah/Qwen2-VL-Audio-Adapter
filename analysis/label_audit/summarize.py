"""
summarize.py
============
Recomputes ALL audit counts from cases.json -- no hand-tallied numbers anywhere.

Modes:
  python3 summarize.py            adjudication progress + reconciliation table
  python3 summarize.py --final    strict gate; refuses unless adjudication complete
  python3 summarize.py --final --emit-typology
                                  additionally writes audit_typology.md

Gate policy (per Kulsoom's Phase-2 spec):
  * Structural breaks (missing/duplicate rows, unknown codes, subcategory from the
    wrong family, nulls in --final mode) -> exit 1.
  * Deltas vs the PUBLISHED tally -> printed as FINDINGS, never an error. The
    published 37/36/14/11 sums to 98/100 and the committed tracker covered 62/63
    rows; any recomputed difference is reported, not hidden.
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


def load():
    with open(CASES_JSON) as fh:
        return json.load(fh)


def recompute(payload):
    cases = payload["cases"]
    disp = Counter(c["original_disposition"] for c in cases)
    sub = defaultdict(Counter)
    for c in cases:
        if c["subcategory"]:
            sub[c["original_disposition"]][c["subcategory"]] += 1
    return disp, sub


def structural_errors(payload, final):
    cases = payload["cases"]
    errs = []
    rows = [c["row"] for c in cases]
    if sorted(rows) != list(range(1, 64)):
        errs.append(f"rows are not exactly 1..63 (n={len(rows)}, dupes/gaps present)")
    if len({c["sample_id"] for c in cases}) != len(cases):
        errs.append("sample_ids are not unique")
    for c in cases:
        fam = FAMILY.get(c["original_disposition"])
        s = c["subcategory"]
        if s:
            if fam is None:
                # ambiguous/normalization/uncategorized carrying a subcategory is
                # only legal if Kulsoom re-dispositioned -- flag as error to force
                # the disposition field to change too.
                errs.append(f"row {c['row']}: subcategory {s} set but disposition "
                            f"'{c['original_disposition']}' is not A/B")
            elif s not in VALID_SUB[fam]:
                errs.append(f"row {c['row']}: subcategory {s} not valid for family {fam}")
        if final:
            if fam and not s:
                errs.append(f"row {c['row']}: {c['original_disposition']} still unadjudicated")
            if c["original_disposition"] == "ambiguous" and not c["notes"].strip():
                errs.append(f"row {c['row']}: ambiguous case missing one-line reason in notes")
            if c["original_disposition"] == "uncategorized":
                errs.append(f"row {c['row']}: still 'uncategorized' -- adjudicate disposition")
    return errs


def reconciliation(payload, disp):
    perfect = payload["meta"]["perfect_match_count"]
    print("\nRECONCILIATION - published vs recomputed (deltas are FINDINGS, not errors)")
    print(f"{'disposition':32s} {'published':>9s} {'recomputed':>10s} {'delta':>6s}")
    total_recomputed = perfect
    rows = [("perfect_match", perfect)]
    for key in ("label_wrong_model_right", "true_model_error", "ambiguous",
                "normalization_only", "uncategorized"):
        n = disp.get(key, 0)
        rows.append((key, n))
        total_recomputed += n
    findings = []
    for key, n in rows:
        pub = PUBLISHED.get(key)
        delta = "" if pub is None else f"{n - pub:+d}" if n != pub else "0"
        pub_s = "-" if pub is None else str(pub)
        print(f"{key:32s} {pub_s:>9s} {n:>10d} {delta:>6s}")
        if pub is not None and n != pub:
            findings.append(f"{key}: recomputed {n} != published {pub}")
        if pub is None and n:
            findings.append(f"{key}: {n} case(s) exist but were absent from every "
                            f"published tally")
    pub_total = sum(PUBLISHED.values())
    print(f"{'TOTAL':32s} {pub_total:>9d} {total_recomputed:>10d} "
          f"{total_recomputed - pub_total:+6d}")
    if pub_total != 100:
        findings.append(f"published categories sum to {pub_total}/100 -- the original "
                        f"tally under-covered the audit: row 60 was never categorized "
                        f"(the four-category 37/36/14/11 version summed to 98, also "
                        f"omitting the normalization row)")
    print("\nFINDINGS:")
    for f in findings or ["none - recomputed matches published exactly"]:
        print(f"  * {f}")
    return findings


def emit_typology(payload, disp, sub, findings):
    cases = payload["cases"]
    meta = payload["meta"]
    by_subcat = defaultdict(list)
    for c in cases:
        if c["subcategory"]:
            by_subcat[c["subcategory"]].append(c)
    for v in by_subcat.values():
        v.sort(key=lambda c: -c["wer_pct"])

    L = []
    add = L.append
    add("# Label-Noise Audit Typology")
    add("")
    add(f"**N = {meta['total_audited']} audited samples** from the SpeechBrain test "
        f"partition: {meta['perfect_match_count']} perfect matches (count from the "
        f"original run's saved output) and {meta['disagreement_count']} disagreements, "
        f"each individually recoverable from the committed notebook and adjudicated "
        f"below. Every count in this file is recomputed from "
        f"`analysis/label_audit/cases.json` by `summarize.py`; nothing is hand-tallied.")
    add("")
    add("## Dispositions (recomputed)")
    add("")
    add("| Disposition | Count |")
    add("|---|---|")
    add(f"| Perfect match | {meta['perfect_match_count']} |")
    for key in ("label_wrong_model_right", "true_model_error", "ambiguous",
                "normalization_only", "uncategorized"):
        if disp.get(key):
            add(f"| {key} | {disp[key]} |")
    add("")
    tax = meta["taxonomy"]
    for disp_key, title in (("label_wrong_model_right", "A - label wrong, model right"),
                            ("true_model_error", "B - true model errors")):
        add(f"## {title} (N={disp.get(disp_key, 0)})")
        add("")
        add("| Code | Meaning | Count | Exemplar sample IDs |")
        add("|---|---|---|---|")
        for code, meaning in tax[disp_key].items():
            n = sub[disp_key].get(code, 0)
            ex = ", ".join(f"`{c['sample_id']}`" for c in by_subcat.get(code, [])[:3])
            add(f"| {code} | {meaning} | {n} | {ex or '-'} |")
        add("")
    amb = [c for c in cases if c["original_disposition"] == "ambiguous"]
    if amb:
        add(f"## C - ambiguous (N={len(amb)})")
        add("")
        for c in sorted(amb, key=lambda c: c["row"]):
            add(f"- `{c['sample_id']}` (row {c['row']}): {c['notes'].strip() or '(reason pending)'}")
        add("")
    malta = [c for c in cases
             if "malta" in (c["prediction_text"] + c["reference_text"]).casefold()]
    if malta:
        m = malta[0]
        add("## The 'Malta' case")
        add("")
        add(f"- `{m['sample_id']}` (row {m['row']}) - adjudicated "
            f"**{m['original_disposition']} / {m['subcategory'] or 'pending'}**. "
            f"A4 is *named* for this mechanism (context-resolved entity), but the "
            f"eponymous case itself was counted against the model in the original "
            f"audit; see its entry in review.md.")
        add("")
    add("## Limitations")
    add("")
    add(f"- {meta['perfect_match_note']}")
    add("- Row numbers refer to the WER-sorted disagreement table in the committed "
        "notebook output (cell 14); the mapping row -> sample_id is preserved in "
        "cases.json.")
    add("- The original published tally (37/36/14/11) summed to 98/100: one "
        "normalization-only row and one never-categorized row (row 60) were "
        "omitted. This typology re-adjudicates both; deltas are listed below.")
    add("")
    add("## Reconciliation findings")
    add("")
    for f in findings or ["none - recomputed matches published exactly"]:
        add(f"- {f}")
    add("")
    with open(TYPOLOGY_MD, "w") as fh:
        fh.write("\n".join(L))
    print(f"\n📄 wrote {TYPOLOGY_MD}")


def main():
    final = "--final" in sys.argv
    payload = load()
    disp, sub = recompute(payload)

    n_adj = sum(1 for c in payload["cases"] if c["subcategory"])
    n_ab = sum(1 for c in payload["cases"]
               if c["original_disposition"] in FAMILY)
    print(f"adjudication progress: {n_adj}/{n_ab} A/B cases have a final subcategory")

    print("\nSUBCATEGORY SUMS vs PARENT DISPOSITIONS (gate)")
    ok = True
    for disp_key in ("label_wrong_model_right", "true_model_error"):
        parent = disp.get(disp_key, 0)
        total = sum(sub[disp_key].values())
        status = "OK" if total == parent else ("incomplete" if not final else "FAIL")
        if final and total != parent:
            ok = False
        print(f"  {disp_key:28s} parent={parent:3d}  subcategorized={total:3d}  [{status}]")
        for code, n in sorted(sub[disp_key].items()):
            print(f"      {code}: {n}")

    findings = reconciliation(payload, disp)

    errs = structural_errors(payload, final)
    if errs:
        print("\nSTRUCTURAL ERRORS (gate -> nonzero exit):")
        for e in errs:
            print(f"  ! {e}")
    if errs or (final and not ok):
        sys.exit(1)

    if "--emit-typology" in sys.argv:
        if not final:
            sys.exit("refusing to emit typology without --final gate")
        emit_typology(payload, disp, sub, findings)


if __name__ == "__main__":
    main()
