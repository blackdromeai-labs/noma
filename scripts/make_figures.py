"""Draw the README figures as SVG, from the numbers in docs/EVALUATION.md.

    python scripts/make_figures.py --out media
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

REEF, TEAL, MARINE, ABYSS, PAPER = "#7DD0C0", "#2E8FA6", "#12507A", "#08243B", "#F7F5EF"
INK2 = "#A9C6D2"
SERIF = "Playfair Display, Georgia, 'Times New Roman', serif"
SANS = "Quicksand, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace"


def frame(w: int, h: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="0.35" y2="1">
    <stop offset="0" stop-color="{ABYSS}"/><stop offset="0.6" stop-color="{MARINE}"/><stop offset="1" stop-color="#1C6E8E"/>
  </linearGradient>
  <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{REEF}"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.85" cy="0.1" r="0.7">
    <stop offset="0" stop-color="{REEF}" stop-opacity="0.22"/><stop offset="1" stop-color="{REEF}" stop-opacity="0"/>
  </radialGradient>
</defs>
<rect width="{w}" height="{h}" rx="18" fill="url(#bg)"/>
<rect width="{w}" height="{h}" rx="18" fill="url(#glow)"/>
{body}
</svg>
"""


def text(x, y, s, size=14, fill=PAPER, font=SANS, anchor="start", weight=400, extra=""):
    s = str(s).replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{s}</text>')


def eyebrow(x, y, s):
    return text(x, y, s.upper(), 12, REEF, SANS, weight=600, extra='letter-spacing="3"')


def banner() -> str:
    b = [eyebrow(64, 92, "Blackdrome AI Labs · open decision model"),
         text(64, 190, "Noma", 104, PAPER, SERIF),
         f'<text x="64" y="250" font-family="{SERIF}" font-size="34" fill="{PAPER}">Decisions '
         f'<tspan font-style="italic" fill="{REEF}">in</tspan> milliseconds.</text>',
         text(64, 296, "State in. Typed, calibrated answers out. No generated text.", 18, INK2)]
    # A small answer card on the right, drawn from a real response (ticket triage).
    x, y = 800, 70
    b.append(f'<rect x="{x}" y="{y}" width="410" height="230" rx="16" fill="{ABYSS}" fill-opacity="0.55" stroke="{REEF}" stroke-opacity="0.25"/>')
    b.append(text(x + 24, y + 40, "team", 13, INK2, MONO))
    b.append(text(x + 76, y + 44, "Billing and refunds", 24, REEF, SERIF))
    for i, (k, p) in enumerate((("billing", 0.866), ("identity", 0.133), ("sales", 0.001), ("shipping", 0.0))):
        yy = y + 82 + i * 30
        b.append(text(x + 24, yy + 5, k, 13, INK2, MONO))
        b.append(f'<rect x="{x + 120}" y="{yy - 4}" width="200" height="8" rx="4" fill="{PAPER}" fill-opacity="0.1"/>')
        b.append(f'<rect x="{x + 120}" y="{yy - 4}" width="{max(4, 200 * p):.0f}" height="8" rx="4" fill="url(#bar)"/>')
        b.append(text(x + 386, yy + 5, f"{p * 100:.1f}%", 13, PAPER, MONO, "end"))
    b.append(text(x + 24, y + 210, "abstain 0.3%   ·   uncertainty 0.001", 12, INK2, MONO))
    return frame(1280, 360, "\n".join(b))


def hbars(small, title, sub, rows, fmt, lo, hi, ticks, tick_fmt, w=960):
    """Log-scale horizontal bars. rows: (name, value, ours)."""
    b = [eyebrow(48, 56, small), text(48, 96, title, 30, PAPER, SERIF), text(48, 124, sub, 14, INK2)]
    x0, x1, llo, lhi = 250, w - 130, math.log10(lo), math.log10(hi)
    bottom = 170 + len(rows) * 40
    for t in ticks:
        x = x0 + (math.log10(t) - llo) / (lhi - llo) * (x1 - x0)
        b.append(f'<line x1="{x:.0f}" y1="150" x2="{x:.0f}" y2="{bottom}" stroke="{PAPER}" stroke-opacity="0.08"/>')
        b.append(text(f"{x:.0f}", bottom + 24, tick_fmt(t), 12, INK2, MONO, "middle"))
    for i, (name, v, ours) in enumerate(rows):
        y = 166 + i * 40
        wd = (math.log10(v) - llo) / (lhi - llo) * (x1 - x0)
        b.append(text(x0 - 18, y + 18, name, 15.5, PAPER if ours else INK2, SANS, "end", 600 if ours else 400))
        b.append(f'<rect x="{x0}" y="{y}" width="{wd:.0f}" height="26" rx="13" '
                 f'fill="{"url(#bar)" if ours else PAPER}" fill-opacity="{"1" if ours else "0.22"}"/>')
        b.append(text(x0 + wd + 12, y + 18, fmt(v), 14.5, REEF if ours else INK2, MONO, weight=500))
    return frame(w, bottom + 50, "\n".join(b))


def latency() -> str:
    rows = [("Noma", 16, True), ("decider-4b v2", 17, False), ("Cygnet", 35, False),
            ("NInfer Flash-Next", 79, False), ("JevOne", 87, False), ("Nimble 9B", 389, False),
            ("OpenJev (thinking)", 463, False), ("Jev 1.13.0", 652, False)]
    return hbars("Latency", "Median time per decision",
                 "End to end over HTTP, JevBench client, one question per request. Log scale. Lower is better.",
                 rows, lambda v: f"{v} ms", 8, 1000, (10, 30, 100, 300, 1000), lambda t: f"{t} ms")


def cost() -> str:
    rows = [("decider-4b v2", 0.020, False), ("Noma", 0.023, True), ("Cygnet", 0.037, False),
            ("Jev 1.13.0", 0.040, False), ("Nimble 9B", 0.166, False)]
    return hbars("Cost", "Cost per 1,000 decisions",
                 "JevBench method: input tokens x hosted price for the size class. Log scale. Lower is better.",
                 rows, lambda v: f"${v:.3f}", 0.01, 0.25, (0.01, 0.02, 0.05, 0.1, 0.2), lambda t: f"${t:g}")


def panel(x, y, w, title, sub, groups, series, vmax=100):
    """Grouped vertical bars. groups: (label, [values]); series: [(name, colour, opacity)]."""
    h = 190
    b = [f'<rect x="{x}" y="{y}" width="{w}" height="{h + 150}" rx="14" fill="{ABYSS}" fill-opacity="0.5" '
         f'stroke="{REEF}" stroke-opacity="0.2"/>',
         text(x + 20, y + 32, title, 17, PAPER, SANS, weight=600), text(x + 20, y + 52, sub, 12, INK2, MONO)]
    base = y + 80 + h
    gw = (w - 40) / len(groups)
    bw = min(46, gw / (len(series) + 1))
    for gi, (label, vals) in enumerate(groups):
        cx = x + 20 + gw * (gi + 0.5)
        for si, v in enumerate(vals):
            bx = cx - bw * len(vals) / 2 + si * bw
            bh = h * v / vmax
            _, col, op = series[si]
            b.append(f'<rect x="{bx + 3:.0f}" y="{base - bh:.0f}" width="{bw - 6:.0f}" height="{bh:.0f}" rx="5" fill="{col}" fill-opacity="{op}"/>')
            b.append(text(f"{bx + bw / 2:.0f}", f"{base - bh - 7:.0f}", f"{v:g}", 11.5, PAPER, MONO, "middle"))
        for li, ln in enumerate(label.split("|")):
            b.append(text(f"{cx:.0f}", base + 20 + li * 16, ln, 12.5, INK2, SANS, "middle"))
    b.append(f'<line x1="{x + 20}" y1="{base}" x2="{x + w - 20}" y2="{base}" stroke="{PAPER}" stroke-opacity="0.25"/>')
    return "\n".join(b)


def ablations() -> str:
    S2 = [("Sealed set", "url(#barv)", 1), ("Hard tier", PAPER, 0.3)]
    b = [f'<defs><linearGradient id="barv" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="{TEAL}"/>'
         f'<stop offset="1" stop-color="{REEF}"/></linearGradient></defs>',
         eyebrow(48, 56, "Ablations"), text(48, 96, "What moved the numbers, and what did not", 30, PAPER, SERIF),
         text(48, 124, "Accuracy in %. One change per panel; recipe, data and budget otherwise fixed.", 14, INK2),
         f'<rect x="760" y="80" width="14" height="14" rx="3" fill="url(#barv)"/>', text(782, 92, "Sealed set", 13, INK2),
         f'<rect x="880" y="80" width="14" height="14" rx="3" fill="{PAPER}" fill-opacity="0.3"/>',
         text(902, 92, "Hard tier (multi-step)", 13, INK2),
         panel(48, 150, 400, "Depth and size", "same 6,000 items",
               [("4B|18 of 32 layers", [79.5, 50.5]), ("4B|32 of 32 layers", [80.1, 53.2]), ("9B|16 of 32 layers", [76.2, 53.2])], S2),
         panel(468, 150, 280, "Training set size", "4B, 18 layers",
               [("6,000|items", [79.5, 50.5]), ("32,000|items", [79.8, 50.5])], S2),
         panel(768, 150, 300, "Targeted multi-step data", "hard tier = held-out half",
               [("before", [79.8, 46.4]), ("+3,500|generated items", [82.6, 46.4])], S2),
         text(48, 520, "Half the depth and half the size lose nothing. More data lifts the sealed set, not the multi-step tier.", 13.5, INK2)]
    return frame(1116, 548, "\n".join(b))


def speed_path() -> str:
    rows = [("First working server", 2300, False), ("Fast path (released)", 14, True)]
    return hbars("Serving ablation", "What the fast path buys",
                 "Time per decision before and after: prefix fork, length buckets, CUDA graph per bucket. Log scale.",
                 rows, lambda v: f"{v:,} ms", 8, 4000, (10, 100, 1000), lambda t: f"{t:,} ms")


def decision_head() -> str:
    def tok(x, y, w, label, hot=False):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="7" fill="{REEF if hot else PAPER}" '
                f'fill-opacity="{0.9 if hot else 0.1}"/>' + text(x + w / 2, y + 20, label, 12, ABYSS if hot else INK2, MONO, "middle"))
    b = [ARROW_DEF, eyebrow(48, 56, "Decision head"), text(48, 96, "From tokens to a calibrated answer", 30, PAPER, SERIF),
         text(48, 124, "Hidden states are read only at the marked positions. Nothing is decoded.", 14, INK2),
         text(48, 172, "prefix (run once, cache forked)", 12, REEF, MONO), text(560, 172, "one block per question", 12, REEF, MONO),
         tok(48, 184, 110, "ref time"), tok(164, 184, 90, "facts"), tok(260, 184, 280, "state ..."),
         tok(560, 184, 130, "instructions"), tok(696, 184, 44, "/q", True), tok(746, 184, 90, "option A"), tok(842, 184, 50, "mark", True),
         tok(898, 184, 90, "option B"), tok(994, 184, 50, "mark", True), tok(1050, 184, 60, "abstain", True),
         box(48, 250, 1062, 64, "Backbone: Qwen3.5-4B, layers 1-18 (Gated DeltaNet + attention), LoRA merged", []),
         arrow(718, 214, 718, 248), arrow(867, 214, 867, 248), arrow(1019, 214, 1019, 248), arrow(1080, 214, 1080, 248)]
    for i in range(4):
        b.append(box(560 + i * 140, 356, 126, 70, f"Scorer {i + 1}", ["listwise"], True))
        b.append(arrow(623 + i * 140, 314, 623 + i * 140, 354))
    b += [box(48, 356, 470, 70, "Evidence head", ["per token: which span of the state decides the answer"]),
          arrow(283, 314, 283, 354),
          box(560, 470, 266, 96, "Mean of the four", ["option probabilities (sum to 1)", "temperature per question type"], True),
          box(844, 470, 266, 96, "Their disagreement", ["uncertainty", "abstain: its own probability"], True),
          arrow(693, 426, 693, 468), arrow(977, 426, 977, 468),
          text(48, 500, "Each scorer trains on its own Poisson", 13, INK2), text(48, 520, "bootstrap of the data; all four share", 13, INK2),
          text(48, 540, "one backbone pass.", 13, INK2)]
    return frame(1158, 600, "\n".join(b))


def bars(title_small, title, sub, rows, w=960) -> str:
    """rows: (label, right-hand note, fraction)."""
    b = [eyebrow(48, 56, title_small), text(48, 96, title, 30, PAPER, SERIF), text(48, 124, sub, 14, INK2)]
    x0, x1 = 300, w - 150
    for i, (name, note, frac) in enumerate(rows):
        y = 158 + i * 40
        b.append(text(x0 - 18, y + 17, name, 15, PAPER, SANS, "end"))
        b.append(f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="24" rx="12" fill="{PAPER}" fill-opacity="0.08"/>')
        b.append(f'<rect x="{x0}" y="{y}" width="{(x1 - x0) * frac:.0f}" height="24" rx="12" fill="url(#bar)"/>')
        b.append(text(x1 + 14, y + 17, note, 14, REEF, MONO, weight=500))
    return frame(w, 158 + len(rows) * 40 + 30, "\n".join(b))


def box(x, y, w, h, title, lines, accent=False):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{ABYSS}" fill-opacity="0.55" '
           f'stroke="{REEF}" stroke-opacity="{0.7 if accent else 0.22}"/>',
           text(x + 18, y + 30, title, 16, REEF if accent else PAPER, SANS, weight=600)]
    for i, ln in enumerate(lines):
        out.append(text(x + 18, y + 54 + i * 19, ln, 12.5, INK2, MONO))
    return "\n".join(out)


def arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{REEF}" stroke-width="1.6" '
            f'stroke-opacity="0.8" marker-end="url(#ah)"/>')


ARROW_DEF = (f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{REEF}"/></marker></defs>')


def architecture() -> str:
    b = [ARROW_DEF, eyebrow(48, 56, "Architecture"),
         text(48, 96, "One forward pass, no decoding", 30, PAPER, SERIF),
         text(48, 124, "The state is read once; every question forks from its cache.", 14, INK2),
         box(48, 160, 210, 150, "Request", ["state (text or JSON)", "questions", "  choice · yes/no · score", "reference_time"]),
         box(298, 160, 210, 150, "Fact channel", ["dates, durations,", "totals, thresholds", "as short fact lines", "deterministic"]),
         box(548, 160, 250, 150, "Backbone", ["Qwen3.5-4B, layers 1-18", "of 32 (56% of depth)", "LoRA merged", "state cache forked per question"], True),
         box(838, 160, 250, 150, "Decision heads", ["listwise option scorer", "4-head ensemble", "abstain head", "evidence head"], True),
         box(838, 350, 250, 130, "Answer", ["probabilities (sum to 1)", "abstain probability", "ensemble uncertainty", "0 output tokens"]),
         box(548, 350, 250, 130, "Fast path", ["length buckets (128)", "CUDA graph per bucket", "no host-device syncs", "14 ms model time (H100)"]),
         arrow(258, 235, 296, 235), arrow(508, 235, 546, 235), arrow(798, 235, 836, 235),
         arrow(963, 310, 963, 348), arrow(673, 348, 673, 312)]
    return frame(1136, 520, "\n".join(b))


def agent_loop() -> str:
    b = [ARROW_DEF, eyebrow(48, 56, "Where it fits"),
         text(48, 96, "The decision layer of an agent", 30, PAPER, SERIF),
         text(48, 124, "A large model does the work. Noma answers the small questions around it, in milliseconds.", 14, INK2),
         box(48, 170, 220, 120, "Incoming request", ["ticket, message,", "tool result, document"]),
         box(318, 170, 240, 120, "Noma: route", ["which queue? which model?", "needs tools? urgent?"], True),
         box(608, 170, 220, 120, "Agent / LLM", ["plans, writes, calls tools", "(seconds, tokens)"]),
         box(878, 170, 240, 120, "Noma: verify", ["did the step succeed?", "safe to continue? done?"], True),
         arrow(268, 230, 316, 230), arrow(558, 230, 606, 230), arrow(828, 230, 876, 230),
         f'<path d="M998 290 C 998 350, 718 350, 718 292" fill="none" stroke="{REEF}" stroke-width="1.6" '
         f'stroke-opacity="0.8" stroke-dasharray="5 5" marker-end="url(#ah)"/>',
         text(858, 352, "retry, escalate or stop when abstain or uncertainty is high", 12.5, INK2, MONO, "middle")]
    return frame(1166, 390, "\n".join(b))


FIGURES = {
    "banner.svg": banner,
    "latency.svg": latency,
    "cost.svg": cost,
    "ablations.svg": ablations,
    "speed-path.svg": speed_path,
    "decision-head.svg": decision_head,
    "architecture.svg": architecture,
    "agent-loop.svg": agent_loop,
    "accuracy.svg": lambda: bars(
        "Accuracy", "Single-pass decisions", "Share of decisions correct. Sealed set: 386 human-reviewed decisions never used in training.",
        [("JevBench easy", "100%  (48/48)", 1.0), ("JevBench original", "98.6%  (71/72)", 0.986),
         ("Sealed set, 12 families", "82.6%  (319/386)", 0.826)]),
    "calibration.svg": lambda: bars(
        "Calibration", "Probabilities you can threshold", "Expected calibration error (ECE). Bar shows 1 - ECE; closer to full is better.",
        [("JevBench easy", "ECE 0.011", 1 - 0.0111), ("Sealed set", "ECE 0.042", 1 - 0.0421),
         ("JevBench original", "ECE 0.084", 1 - 0.0844)]),
    "families.svg": lambda: bars(
        "Sealed set", "Accuracy by decision family", "386 held-out, human-reviewed decisions.",
        [("Ambiguous / out of scope", "16/17", 16 / 17), ("Intent", "32/34", 32 / 34), ("Enum pick", "31/33", 31 / 33),
         ("Tone and safety", "30/32", 30 / 32), ("Severity", "31/34", 31 / 34), ("Adversarial", "29/34", 29 / 34),
         ("Policy", "29/34", 29 / 34), ("Routing", "28/33", 28 / 33), ("Adequacy", "27/34", 27 / 34),
         ("Relevance", "25/33", 25 / 33), ("Trade-off", "24/34", 24 / 34), ("Temporal / numeric", "17/34", 17 / 34)]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="media")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (out / name).write_text(fn(), encoding="utf-8")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
