#!/usr/bin/env python3
"""Builds docs/index.html from dataset/results/loco_eval_results.csv.

Every number on the page is recomputed from the CSV — nothing here is
hand-picked per run, so it stays correct as new rows are appended.
Run it directly, or let .github/workflows/benchmark-report.yml run it
on every push that touches the CSV.
"""
import csv
import math
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "dataset" / "results" / "loco_eval_results.csv"
OUT_PATH = ROOT / "docs" / "index.html"

FAMILY_RULES = [
    ("sam3", "SAM3", "cat-sam3"),
    ("owlv2", "OWLv2", "cat-owl"),
    ("mm_grounding_dino", "Grounding DINO · heavy", "cat-gdh"),
    ("grounding-dino-tiny", "Grounding DINO · tiny", "cat-gdt"),
    ("grounding_dino_tiny", "Grounding DINO · tiny", "cat-gdt"),
]
FALLBACK_COLOR_VARS = ["cat-x1", "cat-x2", "cat-x3", "cat-x4"]


def family_of(model: str):
    low = model.lower()
    for needle, name, css_var in FAMILY_RULES:
        if needle in low:
            return name, css_var
    return None, None


def to_float(s):
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def to_int(s):
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def fnum(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {
                "run": r["run"].strip(),
                "date": r["date"].strip(),
                "model": r["model"].strip(),
                "sample_images": to_int(r["sample_images"]),
                "category": r["category"].strip(),
                "prompt": r["prompt"].strip(),
                "gt_boxes": to_int(r["gt_boxes"]) or 0,
                "predictions": to_int(r["predictions"]) or 0,
                "ap_50_95": to_float(r["AP_50_95"]),
                "ap_50": to_float(r["AP_50"]),
                "ap_75": to_float(r["AP_75"]),
                "ap_95": to_float(r["AP_95"]),
                "ar_100": to_float(r["AR_100"]),
                "sec_per_image": to_float(r["sec_per_image"]),
                "images_per_sec": to_float(r["images_per_sec"]),
                "message": (r["message"] or "").strip(),
            }
            row["family"], row["color_var"] = family_of(row["model"])
            rows.append(row)
    return rows


def assign_fallback_colors(rows):
    seen_order = []
    for r in rows:
        if r["family"] is None:
            r["family"] = r["model"][:32]
        if r["color_var"] is None:
            if r["family"] not in seen_order:
                seen_order.append(r["family"])
    for i, fam in enumerate(seen_order):
        var = FALLBACK_COLOR_VARS[i % len(FALLBACK_COLOR_VARS)]
        for r in rows:
            if r["family"] == fam and r["color_var"] is None:
                r["color_var"] = var


def esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def jstr(s):
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
    )


def build():
    rows = load_rows()
    assign_fallback_colors(rows)

    categories = list(
        dict.fromkeys(r["category"] for r in rows if r["category"] != "OVERALL")
    )
    run_ids = list(dict.fromkeys(r["run"] for r in rows))
    families = list(dict.fromkeys(r["family"] for r in rows))

    overall_rows = [r for r in rows if r["category"] == "OVERALL"]
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)

    # ---- anomaly detection: same exact model string, wildly different AP50 ----
    by_model = defaultdict(list)
    for r in overall_rows:
        by_model[r["model"]].append(r)
    anomalies = []
    for model, group in by_model.items():
        if len(group) < 2:
            continue
        best = max(group, key=lambda r: r["ap_50"] or 0)
        worst = min(group, key=lambda r: r["ap_50"] or 0)
        b, w = best["ap_50"] or 0.0, worst["ap_50"] or 0.0
        if b <= 0:
            continue
        if w == 0:
            anomalies.append({"model": model, "best": best, "worst": worst})
        elif b / w >= 3 and b - w >= 0.01:
            anomalies.append({"model": model, "best": best, "worst": worst})
    anomalies.sort(key=lambda a: (a["best"]["ap_50"] or 0) - (a["worst"]["ap_50"] or 0), reverse=True)
    anomaly_runs = {a["best"]["run"] for a in anomalies} | {a["worst"]["run"] for a in anomalies}

    # ---- top 4 runs by AP50 drive FIG 01 / FIG 02 — ranked by run, not deduped by model ----
    TOP_N = 4
    top_runs = sorted(overall_rows, key=lambda r: (-(r["ap_50"] or 0), r["run"]))[:TOP_N]

    def short_tag(msg, limit=42):
        msg = msg or ""
        return msg if len(msg) <= limit else msg[: limit - 1].rstrip() + "…"

    bars = []
    for r in top_runs:
        ratio = (r["predictions"] / r["gt_boxes"]) if r["gt_boxes"] else 0
        tip = (
            f"{r['model']}\\nrun: {r['run']}\\n"
            f"AP50 {fnum(r['ap_50'])} · AP75 {fnum(r['ap_75'])}\\n"
            f"AR100 {fnum(r['ar_100'])} · {r['predictions']:,} preds / {r['gt_boxes']:,} GT"
        )
        bars.append(
            {
                "name": r["run"],
                "fam": f"{r['family']} · {short_tag(r['message'])}",
                "color_var": r["color_var"],
                "ap50": r["ap_50"] or 0.0,
                "ap75": r["ap_75"] or 0.0,
                "flagged": r["run"] in anomaly_runs,
                "tip": tip,
            }
        )
    scale_max = max([b["ap50"] for b in bars] + [0.01]) * 1.05

    # legend: unique (family, color) pairs among the top runs, in rank order
    legend_items = list({(r["family"], r["color_var"]) for r in top_runs})
    legend_items.sort(key=lambda fc: next(i for i, r in enumerate(top_runs) if r["family"] == fc[0]))

    # ---- heatmap: per-category AP50 for the same top-4 runs ----
    heat_rows = []
    all_cell_vals = []
    for r in top_runs:
        cat_vals = {}
        for cr in by_run[r["run"]]:
            if cr["category"] != "OVERALL":
                cat_vals[cr["category"]] = cr["ap_50"] or 0.0
        vals = [cat_vals.get(c, 0.0) for c in categories]
        all_cell_vals.extend(vals)
        heat_rows.append(
            {
                "name": f"{r['run']} · {r['family']}",
                "color_var": r["color_var"],
                "vals": vals,
                "flagged": r["run"] in anomaly_runs,
            }
        )
    heat_max = max(all_cell_vals) if all_cell_vals else 0.0
    for hr in heat_rows:
        hr["buckets"] = [
            0 if heat_max <= 0 else min(5, int(v / heat_max * 5 + 1e-9)) for v in hr["vals"]
        ]

    # ---- diverging chart: predictions/GT for every OVERALL run ----
    divs = []
    for r in overall_rows:
        if not r["gt_boxes"]:
            continue
        ratio = r["predictions"] / r["gt_boxes"]
        if ratio <= 0:
            continue
        divs.append(
            {
                "name": r["family"],
                "fam": f"{short_tag(r['message'], 30)} · {r['run']}",
                "color_var": "div-under" if ratio < 1 else "div-over",
                "ratio": ratio,
                "flagged": r["run"] in anomaly_runs,
            }
        )
    divs.sort(key=lambda d: d["ratio"])
    half = 2.0
    if divs:
        half = max(2.0, math.ceil(max(abs(math.log2(d["ratio"])) for d in divs)))

    # ---- KPIs ----
    best_ap50_row = max(overall_rows, key=lambda r: r["ap_50"] or -1)
    best_ap75_row = max(overall_rows, key=lambda r: r["ap_75"] or -1)
    over_pred_rows = [
        (r, r["predictions"] / r["gt_boxes"]) for r in overall_rows if r["gt_boxes"]
    ]
    worst_over = max(over_pred_rows, key=lambda t: t[1]) if over_pred_rows else None
    speed_rows = [r for r in overall_rows if r["images_per_sec"]]
    fastest = max(speed_rows, key=lambda r: r["images_per_sec"]) if speed_rows else None

    # ---- title block meta ----
    sample_mode = Counter(r["sample_images"] for r in rows if r["sample_images"]).most_common(1)
    sample_n = sample_mode[0][0] if sample_mode else "—"
    generated = date.today().isoformat()

    # ---- "reading these numbers" / "next up" — mechanically derived ----
    reading_bullets = [
        "Everything here is zero-shot: models prompted with text only, no fine-tuning, no reference crops. "
        "AP is not directly comparable to COCO-scale numbers — LOCO's vocabulary is out-of-distribution for most pretraining captions.",
    ]
    if best_ap50_row["ap_50"] and best_ap75_row["ap_75"]:
        gap_row = best_ap50_row
        reading_bullets.append(
            f"Best run's AP50 → AP75 gap ({fnum(gap_row['ap_50'])} → {fnum(gap_row['ap_75'])}, run {gap_row['run']}) "
            "points at how loose the boxes are, separate from whether the object is found at all."
        )
    cat_avg = {}
    for c in categories:
        vals = [
            (cr["ap_50"] or 0.0)
            for r in top_runs
            for cr in by_run[r["run"]]
            if cr["category"] == c
        ]
        if vals:
            cat_avg[c] = statistics.mean(vals)
    if cat_avg:
        worst_cat = min(cat_avg, key=cat_avg.get)
        reading_bullets.append(
            f"'{worst_cat.replace('_', ' ')}' is the weakest class on average across the top {TOP_N} runs by AP50 "
            f"(mean AP50 {cat_avg[worst_cat]:.3f}) — the one most worth iterating prompts/thresholds on next."
        )

    next_up = []
    if anomalies:
        a = anomalies[0]
        next_up.append(
            f"Root-cause <code>{esc(a['model'][:60])}</code>: run {a['best']['run']} scored AP50 {fnum(a['best']['ap_50'])} "
            f"while run {a['worst']['run']} (same config) scored {fnum(a['worst']['ap_50'])} — treat both as provisional until resolved."
        )
    else:
        next_up.append("No same-config re-runs disagree beyond the noise threshold right now — configs look reproducible.")
    if cat_avg:
        worst_cat = min(cat_avg, key=cat_avg.get)
        next_up.append(f"Prioritize prompt/threshold work on '{worst_cat.replace('_', ' ')}' — it's dragging overall AP the most.")
    latest_run_rows = sorted(overall_rows, key=lambda r: (r["date"], r["run"]))
    if latest_run_rows:
        latest = latest_run_rows[-1]
        if latest["message"]:
            next_up.append(f"Latest logged experiment ({latest['run']}, {latest['date']}): {esc(latest['message'])}")
    next_up.append("If zero-shot plateaus, fine-tuning is on the table — as a separate, clearly-labeled track, not a continuation of these run numbers.")

    return {
        "rows": rows,
        "categories": categories,
        "run_ids": run_ids,
        "families": families,
        "bars": bars,
        "legend_items": legend_items,
        "top_n": TOP_N,
        "scale_max": scale_max,
        "heat_rows": heat_rows,
        "categories_display": categories,
        "divs": divs,
        "half": half,
        "best_ap50_row": best_ap50_row,
        "best_ap75_row": best_ap75_row,
        "worst_over": worst_over,
        "fastest": fastest,
        "sample_n": sample_n,
        "generated": generated,
        "anomalies": anomalies,
        "reading_bullets": reading_bullets,
        "next_up": next_up,
    }


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LOCO Zero-Shot Benchmark</title>
<meta name="description" content="Auto-generated from dataset/results/loco_eval_results.csv — regenerates on every push that touches the CSV.">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root{{
    --bg:#EEF1F3; --surface:#FFFFFF; --surface-2:#F1F4F6; --surface-hover:#E6ECEF;
    --border:#D7DEE2; --border-strong:#B9C4C9; --ink:#171B1E; --ink-secondary:#48545A; --ink-muted:#7C8890;
    --accent:#D65A1F; --status-good:#2F8F5B; --status-warning:#B87A15; --status-critical:#C0392E;
    --status-good-bg:#E5F2EA; --status-warning-bg:#F8EEDC; --status-critical-bg:#F8E6E3;
    --cat-sam3:#00909E; --cat-gdt:#8054B8; --cat-gdh:#B23A5B; --cat-owl:#2E6FBE;
    --cat-x1:#B87A15; --cat-x2:#4C7A3E; --cat-x3:#8C5A2B; --cat-x4:#5A5FA6;
    --seq-0:#EAEEF0; --seq-1:#D3DADE; --seq-2:#A9B6BC; --seq-3:#78898F; --seq-4:#46595F; --seq-5:#17272C;
    --div-under:#2E6FBE; --div-neutral:#9A9C9A; --div-over:#C0392E;
    --shadow:0 1px 2px rgba(23,27,30,.05), 0 8px 24px rgba(23,27,30,.06);
  }}
  @media (prefers-color-scheme: dark){{
    :root:not([data-theme="light"]){{
      --bg:#14181A; --surface:#1B2124; --surface-2:#202829; --surface-hover:#26302F;
      --border:#303A3E; --border-strong:#425053; --ink:#EDEFEF; --ink-secondary:#B7C0C3; --ink-muted:#7E8B8F;
      --accent:#F0864A; --status-good:#5CBE8B; --status-warning:#E0A83D; --status-critical:#E1594C;
      --status-good-bg:#1B2E24; --status-warning-bg:#332A17; --status-critical-bg:#331E1A;
      --cat-sam3:#2FB6C4; --cat-gdt:#A084D6; --cat-gdh:#D8688A; --cat-owl:#6C9FE0;
      --cat-x1:#E0A83D; --cat-x2:#7FBE6B; --cat-x3:#C68A55; --cat-x4:#8C90D6;
      --seq-0:#1F2729; --seq-1:#2B3739; --seq-2:#3D4F52; --seq-3:#587277; --seq-4:#86A6AB; --seq-5:#CFE6E9;
      --div-under:#6C9FE0; --div-neutral:#7E8B8F; --div-over:#E1594C;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px rgba(0,0,0,.35);
    }}
  }}
  :root[data-theme="dark"]{{
    --bg:#14181A; --surface:#1B2124; --surface-2:#202829; --surface-hover:#26302F;
    --border:#303A3E; --border-strong:#425053; --ink:#EDEFEF; --ink-secondary:#B7C0C3; --ink-muted:#7E8B8F;
    --accent:#F0864A; --status-good:#5CBE8B; --status-warning:#E0A83D; --status-critical:#E1594C;
    --status-good-bg:#1B2E24; --status-warning-bg:#332A17; --status-critical-bg:#331E1A;
    --cat-sam3:#2FB6C4; --cat-gdt:#A084D6; --cat-gdh:#D8688A; --cat-owl:#6C9FE0;
    --cat-x1:#E0A83D; --cat-x2:#7FBE6B; --cat-x3:#C68A55; --cat-x4:#8C90D6;
    --seq-0:#1F2729; --seq-1:#2B3739; --seq-2:#3D4F52; --seq-3:#587277; --seq-4:#86A6AB; --seq-5:#CFE6E9;
    --div-under:#6C9FE0; --div-neutral:#7E8B8F; --div-over:#E1594C;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px rgba(0,0,0,.35);
  }}
  *{{box-sizing:border-box;}}
  body{{ margin:0; background:var(--bg); color:var(--ink); font-family:"IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; -webkit-font-smoothing:antialiased; line-height:1.5; }}
  ::selection{{ background: color-mix(in srgb, var(--accent) 30%, transparent); }}
  .sheet{{ max-width:1180px; margin:0 auto; padding:2.5rem 1.5rem 5rem; }}
  a{{ color:var(--accent); }}
  code, .mono{{ font-family:"IBM Plex Mono", ui-monospace, "SF Mono", monospace; font-variant-numeric:tabular-nums; }}
  h1,h2,h3{{ font-family:"Archivo", ui-sans-serif, sans-serif; text-wrap:balance; margin:0; }}
  .eyebrow{{ font-family:"IBM Plex Mono", monospace; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-muted); }}
  .titleblock{{ display:grid; grid-template-columns:1.5fr 1fr; gap:1.5rem; background:var(--surface); border:1px solid var(--border); border-top:3px solid var(--accent); border-radius:3px; padding:1.75rem 1.9rem; box-shadow:var(--shadow); }}
  .titleblock h1{{ font-size:clamp(1.7rem, 1.35rem + 1.6vw, 2.5rem); font-weight:800; letter-spacing:-.01em; margin:.3rem 0 .55rem; }}
  .titleblock .dek{{ color:var(--ink-secondary); font-size:.95rem; max-width:46ch; }}
  .tb-meta{{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--border); border:1px solid var(--border); border-radius:3px; overflow:hidden; align-content:start; }}
  .tb-field{{ background:var(--surface); padding:.6rem .8rem; }}
  .tb-field .k{{ font-family:"IBM Plex Mono",monospace; font-size:.64rem; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-muted); display:block; margin-bottom:.15rem; }}
  .tb-field .v{{ font-family:"IBM Plex Mono",monospace; font-size:.92rem; font-weight:500; }}
  .kpis{{ display:grid; grid-template-columns:repeat(4, 1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:3px; margin-top:1.25rem; overflow:hidden; }}
  .kpi{{ background:var(--surface); padding:1.1rem 1.25rem; display:flex; flex-direction:column; gap:.35rem; }}
  .kpi .val{{ font-family:"IBM Plex Mono",monospace; font-size:1.9rem; font-weight:600; letter-spacing:-.01em; }}
  .kpi .val small{{ font-size:1rem; font-weight:500; color:var(--ink-muted); margin-left:.15rem; }}
  .kpi .lbl{{ font-size:.78rem; color:var(--ink-secondary); }}
  .kpi.flag .val{{ color:var(--status-critical); }}
  .kpi .sub{{ font-size:.72rem; color:var(--ink-muted); }}
  .panel{{ background:var(--surface); border:1px solid var(--border); border-radius:3px; padding:1.6rem 1.75rem 1.8rem; margin-top:1.5rem; box-shadow:var(--shadow); }}
  .panel-head{{ display:flex; align-items:baseline; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:1.1rem; border-bottom:1px solid var(--border); padding-bottom:.85rem;}}
  .panel-head .fig{{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; color:var(--accent); letter-spacing:.08em; }}
  .panel-head h2{{ font-size:1.15rem; font-weight:700; margin-top:.15rem; }}
  .panel-head p{{ margin:.3rem 0 0; color:var(--ink-secondary); font-size:.88rem; max-width:58ch; }}
  .panel-note{{ font-size:.8rem; color:var(--ink-muted); margin-top:1rem; }}
  .legend{{ display:flex; flex-wrap:wrap; gap:.9rem; font-size:.8rem; color:var(--ink-secondary); }}
  .legend .item{{ display:flex; align-items:center; gap:.4rem; }}
  .swatch{{ width:.65rem; height:.65rem; border-radius:2px; flex:none; }}
  .barchart{{ display:flex; flex-direction:column; gap:.9rem; }}
  .barrow{{ display:grid; grid-template-columns:168px 1fr 84px; align-items:center; gap:.9rem; }}
  .barrow .name{{ font-size:.85rem; font-weight:500; }}
  .barrow .name .fam{{ display:block; font-size:.72rem; color:var(--ink-muted); font-weight:400; }}
  .track{{ position:relative; height:1.6rem; background:var(--surface-2); border-radius:2px; overflow:visible; }}
  .fill{{ position:absolute; top:0; left:0; height:100%; border-radius:2px; transition:filter .15s; cursor:pointer; }}
  .fill:hover{{ filter:brightness(1.08); }}
  .marker{{ position:absolute; top:-3px; width:2px; height:calc(100% + 6px); background:var(--ink); opacity:.55; cursor:pointer; }}
  .marker::after{{ content:''; position:absolute; top:-4px; left:50%; transform:translateX(-50%); width:6px; height:6px; border-radius:50%; background:var(--ink); opacity:.7; }}
  .barrow .val{{ font-family:"IBM Plex Mono",monospace; font-size:.92rem; text-align:right; font-weight:600; }}
  .barrow .val small{{ display:block; font-size:.68rem; font-weight:400; color:var(--ink-muted); }}
  .barrow.flagged .name .fam{{ color:var(--status-critical); }}
  .axis{{ display:grid; grid-template-columns:168px 1fr 84px; margin-top:.15rem; }}
  .axis .ticks{{ position:relative; height:1rem; }}
  .axis .tick{{ position:absolute; top:0; font-family:"IBM Plex Mono",monospace; font-size:.66rem; color:var(--ink-muted); transform:translateX(-50%); }}
  .axis .tick.first{{ transform:translateX(0); }}
  .heatwrap{{ overflow-x:auto; }}
  .heatmap{{ display:grid; grid-template-columns:150px repeat({n_cats}, minmax(84px,1fr)); gap:2px; min-width:640px; }}
  .hcell{{ background:var(--surface-2); padding:.55rem .5rem; font-size:.8rem; border-radius:2px; }}
  .hcell.colhead{{ background:transparent; font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.03em; text-transform:uppercase; color:var(--ink-muted); padding:.2rem .5rem .6rem; text-align:center; }}
  .hcell.rowhead{{ display:flex; align-items:center; gap:.45rem; background:transparent; font-weight:500; padding-left:.15rem; }}
  .hcell.val{{ display:flex; align-items:center; justify-content:center; font-family:"IBM Plex Mono",monospace; font-weight:600; cursor:pointer; min-height:2.6rem; }}
  .hcell.corner{{ background:transparent; }}
  .divchart{{ display:flex; flex-direction:column; gap:.7rem; }}
  .divrow{{ display:grid; grid-template-columns:210px 1fr 76px; align-items:center; gap:.9rem; }}
  .divrow .name{{ font-size:.83rem; }}
  .divrow .name .fam{{ display:block; font-size:.7rem; color:var(--ink-muted); }}
  .divtrack{{ position:relative; height:1.5rem; background:var(--surface-2); border-radius:2px; }}
  .divtrack .zero{{ position:absolute; top:-4px; left:50%; width:1px; height:calc(100% + 8px); background:var(--border-strong); }}
  .divbar{{ position:absolute; top:0; height:100%; border-radius:2px; cursor:pointer; }}
  .divbar:hover{{ filter:brightness(1.1); }}
  .divrow .val{{ font-family:"IBM Plex Mono",monospace; font-size:.85rem; text-align:right; font-weight:600; }}
  .divaxis{{ display:grid; grid-template-columns:210px 1fr 76px; margin-top:.2rem; }}
  .divaxis .ticks{{ position:relative; height:1rem; }}
  .divaxis .tick{{ position:absolute; top:0; font-family:"IBM Plex Mono",monospace; font-size:.64rem; color:var(--ink-muted); transform:translateX(-50%); }}
  .badge{{ display:inline-flex; align-items:center; gap:.3rem; font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.03em; text-transform:uppercase; padding:.15rem .45rem; border-radius:2px; font-weight:500; }}
  .badge.critical{{ background:var(--status-critical-bg); color:var(--status-critical); }}
  .callout{{ margin-top:1.5rem; border:1px solid var(--status-critical); background:var(--status-critical-bg); border-radius:3px; padding:1.1rem 1.3rem; display:grid; grid-template-columns:auto 1fr; gap:.9rem; align-items:start; }}
  .callout .ico{{ font-family:"IBM Plex Mono",monospace; font-weight:700; color:var(--status-critical); font-size:.95rem; line-height:1.4; }}
  .callout h3{{ font-size:.95rem; font-weight:700; color:var(--status-critical); margin-bottom:.35rem; }}
  .callout p{{ margin:.3rem 0; font-size:.86rem; color:var(--ink); }}
  .callout .rundiff{{ display:flex; gap:1.5rem; margin-top:.6rem; flex-wrap:wrap; }}
  .callout .rundiff div{{ font-size:.8rem; }}
  .callout .rundiff .k{{ display:block; font-family:"IBM Plex Mono",monospace; font-size:.66rem; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-muted); }}
  .callout .rundiff .v{{ font-family:"IBM Plex Mono",monospace; font-weight:600; }}
  .tablewrap{{ overflow:auto; max-height:26rem; border:1px solid var(--border); border-radius:3px; }}
  table{{ border-collapse:collapse; width:100%; min-width:900px; font-size:.82rem; }}
  thead th{{ position:sticky; top:0; background:var(--surface-2); z-index:1; text-align:left; padding:.55rem .7rem; font-family:"IBM Plex Mono",monospace; font-size:.66rem; text-transform:uppercase; letter-spacing:.05em; color:var(--ink-muted); border-bottom:1px solid var(--border); white-space:nowrap; }}
  tbody td{{ padding:.5rem .7rem; border-bottom:1px solid var(--border); white-space:nowrap; }}
  tbody td.num{{ font-family:"IBM Plex Mono",monospace; text-align:right; font-variant-numeric:tabular-nums; }}
  tbody td.msg{{ white-space:normal; color:var(--ink-secondary); min-width:22ch; }}
  tbody tr.overall{{ background:var(--surface-2); font-weight:600; }}
  tbody tr:hover{{ background:var(--surface-hover); }}
  .runtag{{ font-family:"IBM Plex Mono",monospace; font-size:.78rem; }}
  .notes{{ margin-top:1.5rem; display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }}
  .notes .block h3{{ font-size:.9rem; margin-bottom:.6rem; }}
  .notes ul{{ margin:0; padding-left:1.1rem; font-size:.85rem; color:var(--ink-secondary); display:flex; flex-direction:column; gap:.45rem; }}
  .notes li::marker{{ color:var(--accent); }}
  footer{{ margin-top:2.2rem; padding-top:1.2rem; border-top:1px solid var(--border); font-size:.76rem; color:var(--ink-muted); display:flex; justify-content:space-between; flex-wrap:wrap; gap:.5rem; }}
  #tooltip{{ position:fixed; pointer-events:none; z-index:50; background:var(--ink); color:var(--bg); font-family:"IBM Plex Mono",monospace; font-size:.72rem; padding:.4rem .6rem; border-radius:3px; line-height:1.45; max-width:240px; opacity:0; transform:translate(-50%,-100%); transition:opacity .1s; white-space:pre-line; }}
  #tooltip.show{{ opacity:1; }}
  .stale-banner{{ display:none; margin-top:1rem; border:1px solid var(--status-warning); background:var(--status-warning-bg); color:var(--status-warning); border-radius:3px; padding:.7rem 1rem; font-size:.82rem; }}
  @media (max-width: 760px){{
    .titleblock{{ grid-template-columns:1fr; }}
    .kpis{{ grid-template-columns:1fr 1fr; }}
    .barrow, .axis{{ grid-template-columns:104px 1fr 64px; }}
    .divrow, .divaxis{{ grid-template-columns:1fr; }}
    .divrow .val{{ text-align:left; }}
    .notes{{ grid-template-columns:1fr; }}
  }}
  @media (prefers-reduced-motion: reduce){{ .fill, .divbar, #tooltip{{ transition:none; }} }}
</style>
</head>
<body>
<div class="sheet">

  <header class="titleblock">
    <div>
      <div class="eyebrow">Evaluation report · dataset/results/loco_eval_results.csv · auto-generated</div>
      <h1>LOCO Zero-Shot Benchmark</h1>
      <p class="dek">Open-vocabulary detectors prompted with text descriptions of five logistics object classes — no fine-tuning, no visual exemplars. Scored with COCO-style AP/AR against LOCO ground truth. This page is rebuilt from the CSV by CI on every push.</p>
    </div>
    <div class="tb-meta">
      <div class="tb-field"><span class="k">Dataset</span><span class="v">LOCO · loco-all-v1</span></div>
      <div class="tb-field"><span class="k">Sample</span><span class="v">{sample_n} images</span></div>
      <div class="tb-field"><span class="k">Categories</span><span class="v">{n_cats}</span></div>
      <div class="tb-field"><span class="k">Runs logged</span><span class="v">{n_runs}</span></div>
      <div class="tb-field"><span class="k">Prompting</span><span class="v">Zero-shot / text</span></div>
      <div class="tb-field"><span class="k">Generated</span><span class="v">{generated}</span></div>
    </div>
  </header>

  <section class="kpis">
    <div class="kpi">
      <div class="val">{best_ap50:.3f}</div>
      <div class="lbl">Best AP<span class="mono">50</span> overall</div>
      <div class="sub">{best_ap50_sub}</div>
    </div>
    <div class="kpi">
      <div class="val">{best_ap75:.3f}</div>
      <div class="lbl">Best AP<span class="mono">75</span> overall</div>
      <div class="sub">{best_ap75_sub}</div>
    </div>
    <div class="kpi flag">
      <div class="val">{worst_over_val}<small>{worst_over_unit}</small></div>
      <div class="lbl">Worst over-prediction</div>
      <div class="sub">{worst_over_sub}</div>
    </div>
    <div class="kpi">
      <div class="val">{fastest_val}<small>{fastest_unit}</small></div>
      <div class="lbl">Fastest inference</div>
      <div class="sub">{fastest_sub}</div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <div>
        <div class="fig">FIG. 01</div>
        <h2>Accuracy by run — top {top_n} by AP50, with AP75 marker</h2>
        <p>The {top_n} highest-scoring runs, ranked by AP<span class="mono">50</span> (IoU ≥ 0.5). Tick marker = AP<span class="mono">75</span> (IoU ≥ 0.75) on the same scale. Re-runs of the same config both count as separate runs.</p>
      </div>
      <div class="legend">{legend_html}</div>
    </div>
    <div class="barchart" id="barchart"></div>
    <div class="axis"><div></div><div class="ticks" id="axis-ticks"></div><div></div></div>
    <p class="panel-note">Scale max = {scale_max:.3f} (105% of the current best AP50). Flagged rows have a same-config re-run that disagrees — see the callout below.</p>
  </section>

  <section class="panel">
    <div class="panel-head">
      <div>
        <div class="fig">FIG. 02</div>
        <h2>AP50 by category — same top {top_n} runs</h2>
        <p>Per-category AP50 for the same {top_n} runs as FIG. 01. Darker cell = higher AP50 for that class.</p>
      </div>
      <div class="legend">
        <div class="item"><span class="swatch" style="background:var(--seq-1)"></span>low</div>
        <div class="item"><span class="swatch" style="background:var(--seq-3)"></span>mid</div>
        <div class="item"><span class="swatch" style="background:var(--seq-5)"></span>high</div>
      </div>
    </div>
    <div class="heatwrap"><div class="heatmap" id="heatmap"></div></div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <div>
        <div class="fig">FIG. 03</div>
        <h2>Predictions vs. ground truth — the extrapolation check</h2>
        <p>predictions ÷ gt_boxes, log2 scale. Left of center = model stays quiet; right of center = model floods the image with boxes.</p>
      </div>
    </div>
    <div class="divchart" id="divchart"></div>
    <div class="divaxis"><div></div><div class="ticks" id="divaxis-ticks"></div><div></div></div>
  </section>

  {callout_html}

  <section class="panel">
    <div class="panel-head">
      <div>
        <div class="fig">FIG. 04</div>
        <h2>All logged runs</h2>
        <p>Every row from <code>loco_eval_results.csv</code>, per category and OVERALL.</p>
      </div>
    </div>
    <div class="tablewrap">
      <table>
        <thead>
          <tr>
            <th>Run</th><th>Model</th><th>Category</th>
            <th style="text-align:right">GT</th><th style="text-align:right">Pred</th>
            <th style="text-align:right">AP50-95</th><th style="text-align:right">AP50</th>
            <th style="text-align:right">AP75</th><th style="text-align:right">AR100</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </section>

  <section class="notes">
    <div class="block">
      <h3>Reading these numbers</h3>
      <ul>{reading_html}</ul>
    </div>
    <div class="block">
      <h3>Next up</h3>
      <ul>{next_up_html}</ul>
    </div>
  </section>

  <footer>
    <span>Source: <code>dataset/results/loco_eval_results.csv</code> · <code>dataset/eval_loco.py</code> · <code>dataset/eval_loco_grounding_dino.py</code></span>
    <span>{n_runs} runs · {n_families} model families · {n_cats} LOCO categories</span>
  </footer>

</div>

<div id="tooltip"></div>

<script>
(function(){{
  var SCALE_MAX = {scale_max};
  var bars = {bars_js};
  var barchart = document.getElementById('barchart');
  bars.forEach(function(b){{
    var row = document.createElement('div');
    row.className = 'barrow' + (b.flagged ? ' flagged' : '');
    var wAp50 = Math.min(100, b.ap50 / SCALE_MAX * 100);
    var wAp75 = Math.min(100, b.ap75 / SCALE_MAX * 100);
    row.innerHTML =
      '<div class="name">' + b.name + '<span class="fam">' + b.fam + '</span></div>' +
      '<div class="track">' +
        '<div class="fill" data-tip="' + b.tip + '" style="width:' + wAp50 + '%; background:var(--' + b.color_var + ')"></div>' +
        '<div class="marker" data-tip="AP75 = ' + b.ap75.toFixed(3) + '" style="left:' + wAp75 + '%"></div>' +
      '</div>' +
      '<div class="val">' + b.ap50.toFixed(3) + '<small>AP75 ' + b.ap75.toFixed(3) + '</small></div>';
    barchart.appendChild(row);
  }});
  var ticks = document.getElementById('axis-ticks');
  [0, 1/3, 2/3, 1].forEach(function(f, i){{
    var s = document.createElement('span');
    s.className = 'tick' + (i === 0 ? ' first' : '');
    s.style.left = (f * 100) + '%';
    s.textContent = (f * SCALE_MAX).toFixed(2);
    ticks.appendChild(s);
  }});

  var cats = {cats_js};
  var heatRows = {heat_js};
  var seq = ['var(--seq-0)','var(--seq-1)','var(--seq-2)','var(--seq-3)','var(--seq-4)','var(--seq-5)'];
  var heatmap = document.getElementById('heatmap');
  var corner = document.createElement('div'); corner.className='hcell corner'; heatmap.appendChild(corner);
  cats.forEach(function(c){{
    var h = document.createElement('div'); h.className='hcell colhead'; h.style.whiteSpace='pre-line'; h.textContent=c.replace(/_/g,'\\n'); heatmap.appendChild(h);
  }});
  heatRows.forEach(function(r){{
    var rh = document.createElement('div'); rh.className='hcell rowhead';
    rh.innerHTML = '<span class="swatch" style="background:var(--'+r.color_var+')"></span>' + r.name + (r.flagged ? ' <span class="badge critical" style="margin-left:.3rem">flag</span>' : '');
    heatmap.appendChild(rh);
    r.vals.forEach(function(v, i){{
      var bucket = r.buckets[i];
      var cell = document.createElement('div');
      cell.className = 'hcell val';
      cell.style.background = seq[bucket];
      cell.style.color = bucket >= 4 ? 'var(--seq-0)' : 'var(--ink)';
      cell.textContent = v.toFixed(3);
      cell.setAttribute('data-tip', r.name + ' × ' + cats[i].replace(/_/g,' ') + '\\nAP50 = ' + v.toFixed(3));
      heatmap.appendChild(cell);
    }});
  }});

  var HALF = {half};
  var divs = {divs_js};
  var divchart = document.getElementById('divchart');
  divs.forEach(function(d){{
    var log2 = Math.log2(d.ratio);
    var pct = Math.max(-HALF, Math.min(HALF, log2)) / HALF * 50;
    var row = document.createElement('div');
    row.className = 'divrow';
    var left = pct < 0 ? (50 + pct) : 50;
    var width = Math.abs(pct);
    var ratioLabel = d.ratio >= 1 ? d.ratio.toFixed(1) + '×' : (1/d.ratio).toFixed(1) + '× under';
    row.innerHTML =
      '<div class="name">' + d.name + (d.flagged ? ' <span class="badge critical">flag</span>' : '') + '<span class="fam">' + d.fam + '</span></div>' +
      '<div class="divtrack"><div class="zero"></div>' +
        '<div class="divbar" data-tip="' + d.name + ' (' + d.fam + ')\\npredictions/GT = ' + d.ratio.toFixed(3) + '×" style="left:' + left + '%; width:' + width + '%; background:var(--' + d.color_var + ')"></div>' +
      '</div>' +
      '<div class="val">' + ratioLabel + '</div>';
    divchart.appendChild(row);
  }});
  var dticks = document.getElementById('divaxis-ticks');
  [-HALF, -HALF/2, 0, HALF/2, HALF].forEach(function(v){{
    var s = document.createElement('span');
    s.className = 'tick';
    s.style.left = ((v / HALF) * 50 + 50) + '%';
    var label = v === 0 ? '1×' : (v > 0 ? Math.pow(2, v).toFixed(v % 1 === 0 ? 0 : 1) + '×' : (1/Math.pow(2, v)).toFixed(-v % 1 === 0 ? 0 : 1) + '× under');
    s.textContent = label;
    dticks.appendChild(s);
  }});

  var DATA = {table_js};
  var tbody = document.getElementById('tbody');
  DATA.forEach(function(r){{
    var tr = document.createElement('tr');
    if (r[2] === 'OVERALL') tr.className = 'overall';
    tr.innerHTML =
      '<td class="runtag">' + r[0] + '</td>' +
      '<td>' + r[1] + '</td>' +
      '<td>' + r[2].replace(/_/g,' ') + '</td>' +
      '<td class="num">' + r[3].toLocaleString() + '</td>' +
      '<td class="num">' + r[4].toLocaleString() + '</td>' +
      '<td class="num">' + r[5] + '</td>' +
      '<td class="num">' + r[6] + '</td>' +
      '<td class="num">' + r[7] + '</td>' +
      '<td class="num">' + r[8] + '</td>' +
      '<td class="msg">' + r[9] + '</td>';
    tbody.appendChild(tr);
  }});

  var tip = document.getElementById('tooltip');
  document.addEventListener('mouseover', function(e){{
    var el = e.target.closest('[data-tip]');
    if (!el) return;
    tip.textContent = el.getAttribute('data-tip');
    tip.classList.add('show');
  }});
  document.addEventListener('mousemove', function(e){{
    if (!tip.classList.contains('show')) return;
    tip.style.left = e.clientX + 'px';
    tip.style.top = (e.clientY - 12) + 'px';
  }});
  document.addEventListener('mouseout', function(e){{
    var el = e.target.closest('[data-tip]');
    if (!el) return;
    tip.classList.remove('show');
  }});
}})();
</script>
</body>
</html>
"""


def render(ctx: dict) -> str:
    import json

    def num_or_dash(v, digits=3):
        return "—" if v is None else f"{v:.{digits}f}"

    legend_html = "".join(
        f'<div class="item"><span class="swatch" style="background:var(--{color_var})"></span>{esc(family)}</div>'
        for family, color_var in ctx["legend_items"]
    )

    if ctx["worst_over"]:
        row, ratio = ctx["worst_over"]
        worst_over_val, worst_over_unit = f"{ratio:.1f}", "×"
        worst_over_sub = f"{esc(row['family'])} · run {esc(row['run'])}"
    else:
        worst_over_val, worst_over_unit, worst_over_sub = "—", "", "no OVERALL rows with GT boxes"

    if ctx["fastest"]:
        r = ctx["fastest"]
        fastest_val, fastest_unit = f"{r['images_per_sec']:.1f}", "img/s"
        fastest_sub = f"{esc(r['family'])} · run {esc(r['run'])}"
    else:
        fastest_val, fastest_unit, fastest_sub = "—", "", "no speed data logged yet"

    if ctx["anomalies"]:
        a = ctx["anomalies"][0]
        callout_html = f"""
  <section class="callout">
    <div class="ico">!</div>
    <div>
      <h3>Unresolved — same config, different results</h3>
      <p><code>{esc(a['model'])}</code> was run more than once with identical settings and produced meaningfully different AP50. This has not been root-caused; treat both numbers as provisional until it is.</p>
      <div class="rundiff">
        <div><span class="k">{esc(a['best']['run'])}</span><span class="v">{a['best']['predictions']:,} preds · AP50 {num_or_dash(a['best']['ap_50'])} · AR100 {num_or_dash(a['best']['ar_100'])}</span></div>
        <div><span class="k">{esc(a['worst']['run'])}</span><span class="v">{a['worst']['predictions']:,} preds · AP50 {num_or_dash(a['worst']['ap_50'])} · AR100 {num_or_dash(a['worst']['ar_100'])}</span></div>
      </div>
    </div>
  </section>"""
    else:
        callout_html = ""

    reading_html = "".join(f"<li>{b}</li>" for b in ctx["reading_bullets"])
    next_up_html = "".join(f"<li>{b}</li>" for b in ctx["next_up"])

    table_js = []
    for r in ctx["rows"]:
        table_js.append(
            [
                r["run"],
                r["model"],
                r["category"],
                r["gt_boxes"],
                r["predictions"],
                num_or_dash(r["ap_50_95"]),
                num_or_dash(r["ap_50"]),
                num_or_dash(r["ap_75"]),
                num_or_dash(r["ar_100"]),
                r["message"],
            ]
        )

    bars_for_js = [
        {
            "name": esc(b["name"]),
            "fam": esc(b["fam"]),
            "color_var": b["color_var"],
            "ap50": b["ap50"],
            "ap75": b["ap75"],
            "flagged": b["flagged"],
            "tip": b["tip"],
        }
        for b in ctx["bars"]
    ]
    heat_for_js = [
        {
            "name": esc(h["name"]),
            "color_var": h["color_var"],
            "vals": h["vals"],
            "buckets": h["buckets"],
            "flagged": h["flagged"],
        }
        for h in ctx["heat_rows"]
    ]
    divs_for_js = [
        {
            "name": esc(d["name"]),
            "fam": esc(d["fam"]),
            "color_var": d["color_var"],
            "ratio": d["ratio"],
            "flagged": d["flagged"],
        }
        for d in ctx["divs"]
    ]

    return TEMPLATE.format(
        n_cats=len(ctx["categories"]),
        n_runs=len(ctx["run_ids"]),
        n_families=len(ctx["families"]),
        top_n=ctx["top_n"],
        sample_n=ctx["sample_n"],
        generated=ctx["generated"],
        best_ap50=ctx["best_ap50_row"]["ap_50"] or 0.0,
        best_ap50_sub=f"{esc(ctx['best_ap50_row']['family'])} · run {esc(ctx['best_ap50_row']['run'])}",
        best_ap75=ctx["best_ap75_row"]["ap_75"] or 0.0,
        best_ap75_sub=f"{esc(ctx['best_ap75_row']['family'])} · run {esc(ctx['best_ap75_row']['run'])}",
        worst_over_val=worst_over_val,
        worst_over_unit=worst_over_unit,
        worst_over_sub=worst_over_sub,
        fastest_val=fastest_val,
        fastest_unit=fastest_unit,
        fastest_sub=fastest_sub,
        legend_html=legend_html,
        scale_max=ctx["scale_max"],
        callout_html=callout_html,
        reading_html=reading_html,
        next_up_html=next_up_html,
        bars_js=json.dumps(bars_for_js),
        cats_js=json.dumps(ctx["categories"]),
        heat_js=json.dumps(heat_for_js),
        divs_js=json.dumps(divs_for_js),
        half=ctx["half"],
        table_js=json.dumps(table_js),
    )


def main():
    ctx = build()
    html = render(ctx)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(html):,} bytes) from {len(ctx['rows'])} CSV rows, {len(ctx['run_ids'])} runs")


if __name__ == "__main__":
    main()
