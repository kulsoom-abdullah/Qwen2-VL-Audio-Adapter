"""
extract_cases.py
================
Reconstructs the 100-sample label-noise audit's 63 disagreement cases from the
COMMITTED notebook `notebooks/01_View_Results_Highlighted.ipynb`.

Provenance chain (no other sources used):
  - cell 12 saved output : "Loaded 100 audit samples / 63 disagreements / 37 agreements"
                           -> total_audited=100, perfect_match_count=37
  - cell 14 saved output : rendered disagreement table (28.25 MB HTML) containing, per row:
                           display row # (1..63), sample_id, WER badge, diff-highlighted
                           ground truth + prediction, and base64-embedded WAV audio
  - cell 16 source       : Kulsoom's original adjudication -- row-number-keyed disposition
                           lists (36 label-wrong / 14 model-error / 11 ambiguous /
                           1 normalization; row 60 uncategorized)

Outputs (all under analysis/label_audit/):
  - cases.json  : one record per disagreement, subcategory=null awaiting adjudication,
                  plus deterministic PROVISIONAL subcategory suggestions
  - audio/*.wav : the 63 clips decoded from the notebook (regenerable; gitignored)
  - review.md   : adjudication surface -- row 60 first, then A / B / C / normalization

Deterministic: same committed notebook in -> same cases.json out.
"""

import base64
import difflib
import html
import json
import os
import re
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NOTEBOOK = os.path.join(ROOT, "notebooks", "01_View_Results_Highlighted.ipynb")
AUDIO_DIR = os.path.join(HERE, "audio")
CASES_JSON = os.path.join(HERE, "cases.json")
REVIEW_MD = os.path.join(HERE, "review.md")

# ---------------------------------------------------------------------------
# Original dispositions, transcribed verbatim from committed notebook cell 16
# (row numbers refer to the WER-sorted disagreement table in cell 14's output).
# ---------------------------------------------------------------------------
LABEL_ERROR_ROWS = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 16, 17, 22, 23,
                    25, 27, 28, 29, 32, 33, 37, 39, 40, 42, 43, 44, 46,
                    48, 49, 52, 53, 56, 57, 59, 62]          # 36 rows
MODEL_ERROR_ROWS = [14, 19, 21, 24, 26, 31, 34, 38, 41, 47, 51, 55, 61, 63]  # 14 rows
AMBIGUOUS_ROWS = [10, 15, 18, 20, 30, 35, 36, 45, 50, 54, 58]               # 11 rows
NORMALIZATION_ROWS = [7]                                                     # 1 row
# Row 60 was never categorized in cell 16 -> disposition "uncategorized".

DISPOSITION_BY_ROW = {}
for _r in LABEL_ERROR_ROWS:
    DISPOSITION_BY_ROW[_r] = "label_wrong_model_right"
for _r in MODEL_ERROR_ROWS:
    DISPOSITION_BY_ROW[_r] = "true_model_error"
for _r in AMBIGUOUS_ROWS:
    DISPOSITION_BY_ROW[_r] = "ambiguous"
for _r in NORMALIZATION_ROWS:
    DISPOSITION_BY_ROW[_r] = "normalization_only"

TAXONOMY = {
    "label_wrong_model_right": {
        "A1": "spelling/orthography fix",
        "A2": "missing word restored",
        "A3": "grammar/agreement normalization",
        "A4": "contextual entity resolution (Malta pattern)",
        "A5": "punctuation/format only",
        "A6": "other",
    },
    "true_model_error": {
        "B1": "acoustic confusion (phonetically similar substitution)",
        "B2": "omission",
        "B3": "insertion/hallucination",
        "B4": "fluent semantic substitution",
        "B5": "other",
    },
}

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def strip_tags(fragment):
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def norm_words(text):
    """casefold + strip punctuation per word, drop empties (for equivalence checks)."""
    return [w for w in (t.translate(PUNCT_TABLE) for t in text.casefold().split()) if w]


def word_opcodes(ref_text, pred_text):
    """difflib opcodes over casefolded words; returns diff with ORIGINAL-case words."""
    ref_words, pred_words = ref_text.split(), pred_text.split()
    matcher = difflib.SequenceMatcher(
        None, [w.casefold() for w in ref_words], [w.casefold() for w in pred_words]
    )
    diff = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        diff.append({"op": op, "ref": ref_words[a0:a1], "pred": pred_words[b0:b1]})
    return diff


def char_ratio(a, b):
    return difflib.SequenceMatcher(None, a.casefold(), b.casefold()).ratio()


def inline_diff(ref_text, pred_text):
    """Compact text markers for review.md: [-only in label-] {+only in prediction+}."""
    ref_words, pred_words = ref_text.split(), pred_text.split()
    matcher = difflib.SequenceMatcher(
        None, [w.casefold() for w in ref_words], [w.casefold() for w in pred_words]
    )
    parts = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            parts.append(" ".join(ref_words[a0:a1]))
        else:
            if a1 > a0:
                parts.append("[-" + " ".join(ref_words[a0:a1]) + "-]")
            if b1 > b0:
                parts.append("{+" + " ".join(pred_words[b0:b1]) + "+}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Deterministic PROVISIONAL classifiers (suggestions only -- Kulsoom adjudicates)
# ---------------------------------------------------------------------------

def _pair_facts(diff, full_text):
    """Facts about replace-pairs used by both classifiers."""
    replace_pairs = []
    for d in diff:
        if d["op"] == "replace" and len(d["ref"]) == len(d["pred"]):
            replace_pairs.extend(zip(d["ref"], d["pred"]))
        elif d["op"] == "replace":
            replace_pairs.append((" ".join(d["ref"]), " ".join(d["pred"])))
    facts = []
    corpus = full_text.casefold().split()
    for ref_w, pred_w in replace_pairs:
        facts.append({
            "pair": f"{ref_w}->{pred_w}",
            "ratio": round(char_ratio(ref_w, pred_w), 2),
            "punct_only": ref_w.translate(PUNCT_TABLE).casefold()
                          == pred_w.translate(PUNCT_TABLE).casefold(),
            "shared_prefix4": ref_w.casefold()[:4] == pred_w.casefold()[:4]
                              and len(ref_w) >= 4,
            "pred_elsewhere": corpus.count(pred_w.translate(PUNCT_TABLE).casefold()) >= 2,
        })
    return facts


def classify_label_wrong(ref, pred, diff):
    if norm_words(ref) == norm_words(pred):
        return "A5", "texts identical after punctuation/case stripping"
    ops = {d["op"] for d in diff}
    facts = _pair_facts(diff, ref + " " + pred)
    if ops == {"insert"}:
        added = [w for d in diff for w in d["pred"]]
        return "A2", f"model adds word(s) absent from label: {added}"
    if ops == {"replace"} and facts:
        if all(f["punct_only"] for f in facts):
            return "A5", f"replacements differ only in punctuation: {[f['pair'] for f in facts]}"
        entity = [f for f in facts if f["pred_elsewhere"] and f["ratio"] < 0.75]
        if entity:
            return "A4", (f"replacement token also appears elsewhere in clip text "
                          f"(Malta pattern): {[f['pair'] for f in entity]}")
        if all(f["shared_prefix4"] for f in facts):
            return "A3", f"morphological variants (shared stems): {[f['pair'] for f in facts]}"
        if all(f["ratio"] >= 0.75 for f in facts):
            return "A1", f"near-identical word forms: {[f['pair'] for f in facts]}"
    shape = ", ".join(f"{d['op']}({len(d['ref'])}->{len(d['pred'])})" for d in diff)
    return "A6", f"mixed/unclassified diff shape: {shape}"


def classify_model_error(ref, pred, diff):
    ops = {d["op"] for d in diff}
    facts = _pair_facts(diff, ref + " " + pred)
    if ops == {"delete"}:
        dropped = [w for d in diff for w in d["ref"]]
        return "B2", f"model omits label word(s): {dropped}"
    if ops == {"insert"}:
        added = [w for d in diff for w in d["pred"]]
        return "B3", f"model inserts word(s) not in label: {added}"
    if ops == {"replace"} and facts:
        if all(f["ratio"] >= 0.5 for f in facts):
            return "B1", f"phonetically/orthographically close substitutions: {[f['pair'] for f in facts]}"
        return "B4", f"fluent but dissimilar substitutions: {[f['pair'] for f in facts]}"
    shape = ", ".join(f"{d['op']}({len(d['ref'])}->{len(d['pred'])})" for d in diff)
    return "B5", f"mixed/unclassified diff shape: {shape}"


def diff_shape_summary(diff):
    n_ins = sum(len(d["pred"]) for d in diff if d["op"] == "insert")
    n_del = sum(len(d["ref"]) for d in diff if d["op"] == "delete")
    n_rep = sum(max(len(d["ref"]), len(d["pred"])) for d in diff if d["op"] == "replace")
    return f"{n_rep} replaced, {n_del} dropped, {n_ins} added"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract():
    with open(NOTEBOOK) as fh:
        nb = json.load(fh)
    cells = nb["cells"]

    # -- provenance: cell 12 saved output --------------------------------------
    cell12_text = "".join(t for o in cells[12]["outputs"] for t in o.get("text", []))
    m_total = re.search(r"Loaded (\d+) audit samples", cell12_text)
    m_dis = re.search(r"Found (\d+) disagreements", cell12_text)
    m_agree = re.search(r"Agreements: (\d+)", cell12_text)
    if not (m_total and m_dis and m_agree):
        sys.exit("FATAL: cell 12 saved output missing expected run counts")
    total_audited, n_disagreements, n_perfect = (
        int(m_total.group(1)), int(m_dis.group(1)), int(m_agree.group(1)))
    assert (total_audited, n_disagreements, n_perfect) == (100, 63, 37), \
        f"cell 12 counts {total_audited}/{n_disagreements}/{n_perfect} != published 100/63/37"

    # -- cell 14 saved output: the rendered table ------------------------------
    table_html = None
    for out in cells[14]["outputs"]:
        if "text/html" in out.get("data", {}):
            table_html = "".join(out["data"]["text/html"])
    if table_html is None:
        sys.exit("FATAL: cell 14 has no saved text/html output")

    os.makedirs(AUDIO_DIR, exist_ok=True)
    cases = []
    for row_html in re.finditer(r"<tr>(.*?)</tr>", table_html, re.S):
        body = row_html.group(1)
        m_idx = re.search(r"color: #666;'>(\d+)</td>", body)
        if not m_idx:            # header row
            continue
        row = int(m_idx.group(1))
        sample_id = re.search(r"<small[^>]*>(test_\d+)</small>", body).group(1)
        wer_pct = float(re.search(r'class="wer-badge">([\d.]+)% WER', body).group(1))
        b64 = re.search(r'base64,([A-Za-z0-9+/=]+)"', body)

        tds = re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)
        ref_text, pred_text = strip_tags(tds[-2]), strip_tags(tds[-1])

        audio_file = f"row{row:02d}_{sample_id}.wav"
        if b64:
            with open(os.path.join(AUDIO_DIR, audio_file), "wb") as fh:
                fh.write(base64.b64decode(b64.group(1)))
        else:
            audio_file = None

        diff = word_opcodes(ref_text, pred_text)
        disposition = DISPOSITION_BY_ROW.get(row, "uncategorized")

        provisional, rationale = None, None
        if disposition == "label_wrong_model_right":
            provisional, rationale = classify_label_wrong(ref_text, pred_text, diff)
        elif disposition == "true_model_error":
            provisional, rationale = classify_model_error(ref_text, pred_text, diff)
        elif disposition in ("ambiguous", "uncategorized", "normalization_only"):
            rationale = f"diff shape: {diff_shape_summary(diff)}"

        cases.append({
            "row": row,
            "sample_id": sample_id,
            "audio_path": f"analysis/label_audit/audio/{audio_file}" if audio_file else None,
            "wer_pct": wer_pct,
            "reference_text": ref_text,
            "prediction_text": pred_text,
            "word_diff": diff,
            "original_disposition": disposition,
            "subcategory": None,
            "provisional_subcategory": provisional,
            "provisional_rationale": rationale,
            "notes": "",
        })

    cases.sort(key=lambda c: c["row"])
    assert len(cases) == 63, f"expected 63 disagreement rows, extracted {len(cases)}"
    assert len({c["sample_id"] for c in cases}) == 63, "sample_ids not unique"

    payload = {
        "meta": {
            "source_notebook": "notebooks/01_View_Results_Highlighted.ipynb",
            "source_cells": {
                "run_counts": "cell 12 saved output",
                "per_case_content": "cell 14 saved output (WER-sorted disagreement table)",
                "dispositions": "cell 16 source (row-number-keyed lists)",
            },
            "extraction_script": "analysis/label_audit/extract_cases.py",
            "total_audited": total_audited,
            "perfect_match_count": n_perfect,
            "disagreement_count": n_disagreements,
            "perfect_match_note": (
                "The 37 perfect-match samples' identities are NOT individually "
                "recoverable: output/audit_batch_results.json and data/ were not "
                "preserved, and the source dataset revision is unpinned. Count is "
                "taken from cell 12's saved output of the original run."
            ),
            "row_semantics": "row = position in cell 14's WER-sorted disagreement table (1-63)",
            "diff_semantics": "word-level difflib opcodes, case-insensitive match, original case preserved",
            "taxonomy": TAXONOMY,
            "label_status": "PROVISIONAL - subcategory fields are null until Kulsoom adjudicates",
        },
        "cases": cases,
    }
    with open(CASES_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


# ---------------------------------------------------------------------------
# review.md
# ---------------------------------------------------------------------------

def row60_equivalence(case):
    ref, pred = case["reference_text"], case["prediction_text"]
    return {
        "exact_equal": ref == pred,
        "casefold_equal": ref.casefold() == pred.casefold(),
        "punct_and_case_stripped_equal": norm_words(ref) == norm_words(pred),
    }


def write_review(payload):
    cases = {c["row"]: c for c in payload["cases"]}
    lines = []
    add = lines.append
    add("# Label-Noise Audit — Adjudication Review")
    add("")
    add("**How to adjudicate:** for each case set `subcategory` in `cases.json` "
        "(and optionally `notes`). Provisional labels below are deterministic "
        "suggestions only. For ambiguous cases write a one-line reason in `notes`; "
        "leave `subcategory` null unless you re-disposition them.")
    add("")
    add("Taxonomy — A1 spelling/orthography · A2 missing word restored · "
        "A3 grammar/agreement · A4 contextual entity resolution · "
        "A5 punctuation/format only · A6 other || B1 acoustic confusion · "
        "B2 omission · B3 insertion/hallucination · B4 fluent semantic "
        "substitution · B5 other")
    add("")

    def emit(case, header_extra=""):
        add("---")
        add(f"## Row {case['row']:02d} — `{case['sample_id']}` "
            f"({case['original_disposition']}){header_extra}")
        add("")
        add(f"*WER {case['wer_pct']}%* · audio: `{case['audio_path']}`")
        add("")
        add(f"**LABEL:** {case['reference_text']}")
        add("")
        add(f"**MODEL:** {case['prediction_text']}")
        add("")
        add(f"**DIFF:** {inline_diff(case['reference_text'], case['prediction_text'])}")
        add("")
        if case["provisional_subcategory"]:
            add(f"**PROVISIONAL:** `{case['provisional_subcategory']}` — "
                f"{case['provisional_rationale']}")
        elif case["provisional_rationale"]:
            add(f"**SHAPE:** {case['provisional_rationale']}")
        add("")
        add("**FINAL (set in cases.json):** `subcategory: ____`   notes: ____")
        add("")

    # --- Row 60 first, with the explicit question -----------------------------
    r60 = cases[60]
    eq = row60_equivalence(r60)
    add("# ⚠️ FIRST: Row 60 — the uncategorized disagreement")
    add("")
    add(f"Equivalence checks — exact: **{eq['exact_equal']}**, "
        f"case-insensitive: **{eq['casefold_equal']}**, "
        f"punctuation+case stripped: **{eq['punct_and_case_stripped_equal']}**.")
    add("")
    if eq["punct_and_case_stripped_equal"]:
        add("> **Flag:** the difference is punctuation/whitespace/case only. "
            "DECISION FOR KULSOOM: was this ever a real disagreement? Options: "
            "re-disposition as `normalization_only`, or fold into A/B/C. "
            "Not forced into the taxonomy.")
    else:
        add("> Texts differ beyond punctuation/case — adjudicate disposition "
            "AND subcategory (it was never assigned one in the original audit).")
    add("")
    emit(r60)

    for title, rows in (
        ("A — label wrong, model right (36)", LABEL_ERROR_ROWS),
        ("B — true model errors (14)", MODEL_ERROR_ROWS),
        ("C — ambiguous (11)", AMBIGUOUS_ROWS),
        ("Normalization (1)", NORMALIZATION_ROWS),
    ):
        add(f"# {title}")
        add("")
        for row in rows:
            emit(cases[row])

    with open(REVIEW_MD, "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    payload = extract()
    write_review(payload)
    n_prov = sum(1 for c in payload["cases"] if c["provisional_subcategory"])
    print(f"✅ extracted {len(payload['cases'])} cases -> cases.json")
    print(f"   audio decoded -> audio/ ({len(os.listdir(AUDIO_DIR))} wav files)")
    print(f"   provisional labels proposed: {n_prov} "
          f"(A: {sum(1 for c in payload['cases'] if (c['provisional_subcategory'] or '').startswith('A'))}, "
          f"B: {sum(1 for c in payload['cases'] if (c['provisional_subcategory'] or '').startswith('B'))})")
    print(f"   review surface -> review.md (row 60 first)")
