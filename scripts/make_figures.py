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


def latency() -> str:
    rows = [("Noma", 16, True), ("decider-4b v2", 17, False), ("Cygnet", 35, False),
            ("Nimble 9B", 389, False), ("Jev 1.13.0", 652, False)]
    b = [eyebrow(48, 56, "Latency"), text(48, 96, "Median time per decision", 30, PAPER, SERIF),
         text(48, 124, "End to end over HTTP, JevBench client, public tasks. Log scale. Lower is better.", 14, INK2)]
    x0, x1, lo, hi = 230, 860, math.log10(8), math.log10(1000)
    for t in (10, 30, 100, 300, 1000):
        x = x0 + (math.log10(t) - lo) / (hi - lo) * (x1 - x0)
        b.append(f'<line x1="{x:.0f}" y1="150" x2="{x:.0f}" y2="400" stroke="{PAPER}" stroke-opacity="0.08"/>')
        b.append(text(f"{x:.0f}", 424, f"{t} ms", 12, INK2, MONO, "middle"))
    for i, (name, ms, ours) in enumerate(rows):
        y = 170 + i * 46
        wd = (math.log10(ms) - lo) / (hi - lo) * (x1 - x0)
        b.append(text(x0 - 18, y + 19, name, 16, PAPER if ours else INK2, SANS, "end", 600 if ours else 400))
        fill = "url(#bar)" if ours else PAPER
        op = "1" if ours else "0.22"
        b.append(f'<rect x="{x0}" y="{y}" width="{wd:.0f}" height="28" rx="14" fill="{fill}" fill-opacity="{op}"/>')
        b.append(text(x0 + wd + 12, y + 19, f"{ms} ms", 15, REEF if ours else INK2, MONO, weight=500))
    return frame(960, 450, "\n".join(b))


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
