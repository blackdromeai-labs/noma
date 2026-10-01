"""Fact channel: deterministic code that turns dates and quantities into
short fact lines the model can read. No parameters; the version is part of the model card
because changing it changes model inputs.

    facts(state_text, question_texts, reference_time) -> ["F1: ...", ...]

Facts are hints, not overrides: they state what the text says and how quantities compare,
never which option is right.
"""

from __future__ import annotations

import datetime as dt
import re

VERSION = "facts-0.2"
MAX_FACTS = 16

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], 1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|" \
       r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}))?")
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\.?,?\s+(\d{{4}})\b", re.I)
_MDY = re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_REL = re.compile(r"\b(\d+)\s+(day|week|month|year)s?\s+(ago|from now|later|earlier)\b|"
                  r"\bin\s+(\d+)\s+(day|week|month|year)s?\b|\b(yesterday|today|tomorrow)\b", re.I)
_CUR = {"$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY", "₹": "INR"}
# Numbers: a left boundary and bounded digit groups keep matching linear; the unbounded
# \d[\d,]* form retried from every digit and took over an hour on one 80k-character state.
_QTY = re.compile(
    r"(?P<cur>[$£€¥₹])\s?(?P<a>(?:\d{1,3}(?:,\d{3})+|\d{1,15})(?:\.\d{1,6})?)\s?(?P<mult>k|m|bn|million|billion|thousand)?\b"
    r"|(?<![\d,.])(?P<b>(?:\d{1,3}(?:,\d{3})+|\d{1,15})(?:\.\d{1,6})?)(?!\d)\s?(?P<unit>%|percent\b|usd\b|gbp\b|eur\b|ms\b|kg\b|km\b|gb\b|mb\b|"
    r"tb\b|hours?\b|hrs?\b|minutes?\b|mins?\b|seconds?\b|days?\b|weeks?\b|months?\b|years?\b)",
    re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
_UNIT = {"percent": "%", "hour": "h", "hours": "h", "hr": "h", "hrs": "h", "minute": "min",
         "minutes": "min", "mins": "min", "min": "min", "second": "s", "seconds": "s",
         "day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month",
         "months": "month", "year": "year", "years": "year"}


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if x == int(x) and abs(x) >= 1000 else f"{x:g}"


def parse_reference(ref) -> dt.datetime | None:
    if not ref:
        return None
    try:
        return dt.datetime.fromisoformat(str(ref).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def dates(text: str, ref: dt.datetime | None) -> list[tuple[dt.date, str]]:
    out = []
    for m in _ISO.finditer(text):
        try:
            out.append((dt.date(int(m[1]), int(m[2]), int(m[3])), m[0]))
        except ValueError:
            pass
    for m in _DMY.finditer(text):
        try:
            out.append((dt.date(int(m[3]), _MONTHS[m[2].lower()[:3]], int(m[1])), m[0]))
        except (ValueError, KeyError):
            pass
    for m in _MDY.finditer(text):
        try:
            out.append((dt.date(int(m[3]), _MONTHS[m[1].lower()[:3]], int(m[2])), m[0]))
        except (ValueError, KeyError):
            pass
    if ref:
        days = {"day": 1, "week": 7, "month": 30, "year": 365}
        for m in _REL.finditer(text):
            if m[6]:
                off = {"yesterday": -1, "today": 0, "tomorrow": 1}[m[6].lower()]
            elif m[1]:
                off = int(m[1]) * days[m[2].lower()] * (-1 if m[3].lower() in ("ago", "earlier") else 1)
            else:
                off = int(m[4]) * days[m[5].lower()]
            out.append(((ref + dt.timedelta(days=off)).date(), m[0]))
    seen, uniq = set(), []
    for d, s in out:
        if (d, s) not in seen:
            seen.add((d, s))
            uniq.append((d, s))
    return uniq


_CODE = re.compile(r"\b(USD|EUR|GBP|JPY|INR|CHF|CAD|AUD|SEK|NOK|DKK)\s?((?:\d{1,3}(?:,\d{3})+|\d{1,15})(?:\.\d{1,6})?)"
                   r"\s?(k|m|bn|million|billion|thousand)?\b")


def quantities(text: str) -> list[tuple[float, str, str]]:
    """(value, kind, surface) where kind is a currency code or a normalized unit, in document
    order."""
    out = []
    for m in _CODE.finditer(text):
        out.append((m.start(), _num(m[2]) * _MULT.get((m[3] or "").lower(), 1), m[1], m[0].strip()))
    for m in _QTY.finditer(text):
        if m["cur"]:
            v = _num(m["a"]) * _MULT.get((m["mult"] or "").lower(), 1)
            out.append((m.start(), v, _CUR[m["cur"]], m[0].strip()))
        else:
            u = m["unit"].lower()
            if u in ("usd", "gbp", "eur") and any(s <= m.start() < s + len(x) for s, _, _, x in out):
                continue
            out.append((m.start(), _num(m["b"]), _UNIT.get(u, u.upper() if len(u) == 3 else u),
                        m[0].strip()))
    return [(v, k, s) for _, v, k, s in sorted(out, key=lambda x: x[0])]


def _quantities_v01(text: str) -> list[tuple[float, str, str]]:
    out = []
    for m in _QTY.finditer(text):
        if m["cur"]:
            v = _num(m["a"]) * _MULT.get((m["mult"] or "").lower(), 1)
            out.append((v, _CUR[m["cur"]], m[0].strip()))
        else:
            u = m["unit"].lower()
            out.append((_num(m["b"]), _UNIT.get(u, u.upper() if len(u) == 3 else u), m[0].strip()))
    return out


_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_BIN = {"tib": 1024 ** 4, "gib": 1024 ** 3, "mib": 1024 ** 2}
_DEC = {"tb": 1e12, "gb": 1e9, "mb": 1e6}
_BYTES = re.compile(r"(?<![\d,.])((?:\d{1,3}(?:,\d{3})+|\d{1,15})(?:\.\d{1,6})?)(?!\d)\s?(TiB|GiB|MiB|TB|GB|MB)\b")
_SENT = re.compile(r"(?:[^.;\n]|\.(?=\d))+")  # sentence pieces; keeps decimal points
_RULE = re.compile(r"\b(budget|limit|cap|allowance|threshold|maximum|minimum|up to|at most|"
                   r"at least|exceed|above|below|more than|less than|no more|no less|must not)\b",
                   re.I)


def _dated_positions(text: str) -> list[tuple[int, int, dt.date]]:
    out = []
    for rx, f in ((_ISO, lambda m: dt.date(int(m[1]), int(m[2]), int(m[3]))),
                  (_DMY, lambda m: dt.date(int(m[3]), _MONTHS[m[2].lower()[:3]], int(m[1]))),
                  (_MDY, lambda m: dt.date(int(m[3]), _MONTHS[m[1].lower()[:3]], int(m[2])))):
        for m in rx.finditer(text):
            try:
                out.append((m.start(), m.end(), f(m)))
            except (ValueError, KeyError):
                pass
    return sorted(out)


def timestamps(text: str) -> list[tuple[dt.datetime, str]]:
    """Clock times joined to the nearest date within 60 characters, in document order."""
    ds = _dated_positions(text)
    out = []
    for m in _TIME.finditer(text):
        near = [(min(abs(m.start() - e), abs(s - m.end())), d) for s, e, d in ds
                if min(abs(m.start() - e), abs(s - m.end())) <= 60]
        if near:
            d = min(near, key=lambda x: x[0])[1]
            lo, hi = max(0, m.start() - 30), min(len(text), m.end() + 10)
            out.append((dt.datetime(d.year, d.month, d.day, int(m[1]), int(m[2])),
                        " ".join(text[lo:hi].split())))
    return out


def _v02(state: str, s_q, q_q) -> list[tuple[int, str]]:
    """facts-0.2: elapsed time between timestamps, day spans,
    running totals and threshold crossings, percent-of-amount, binary/decimal data units."""
    lines: list[tuple[int, str]] = []
    ts = timestamps(state)
    for (a, sa), (b, sb) in list(zip(ts, ts[1:]))[:6]:
        h = (b - a).total_seconds() / 3600
        lines.append((1, f"{_fmt(round(h, 2))} h elapsed from {a:%Y-%m-%d %H:%M} to {b:%Y-%m-%d %H:%M} "
                         f"(local times as written; time zones not applied)"))
    ds = []
    for _, _, d in _dated_positions(state):
        if not ds or ds[-1] != d:
            ds.append(d)
    for a, b in list(zip(ds, ds[1:]))[:4]:
        n = abs((b - a).days)
        lines.append((2, f"{min(a, b).isoformat()} to {max(a, b).isoformat()}: {n} days apart, "
                         f"{n + 1} days counting both ends"))
    # Percent of an amount stated in the same sentence ("80% of the USD 12,000 budget").
    derived = []
    for sent in _SENT.findall(state):
        qs = quantities(sent)
        pcts = [v for v, k, _ in qs if k == "%"]
        amts = [(v, k) for v, k, _ in qs if k not in ("%",) and k.isupper()]
        for p in pcts[:2]:
            for v, k in amts[:2]:
                x = v * p / 100
                derived.append((x, k))
                lines.append((1, f"{_fmt(p)}% of {_fmt(v)} {k} = {_fmt(round(x, 2))} {k}"))
    # Ledger amounts vs rule amounts: sentences that state a limit are thresholds, the rest are
    # entries to accumulate in document order.
    by_kind: dict[str, list] = {}
    rule_amts: list[tuple[float, str]] = []
    for sent in _SENT.findall(state):
        is_rule = bool(_RULE.search(sent))
        for v, k, _ in quantities(sent):
            if k.isupper():
                (rule_amts.append((v, k)) if is_rule else by_kind.setdefault(k, []).append(v))
    for k, vs in by_kind.items():
        if len(vs) < 3:
            continue
        run, tot = [], 0.0
        for v in vs[:12]:
            tot += v
            run.append(tot)
        lines.append((2, f"running total {k} of listed amounts in document order: " +
                      " -> ".join(_fmt(round(x, 2)) for x in run[:8])))
        thresholds = list(dict.fromkeys([(v, kk) for v, kk, _ in q_q] + derived + rule_amts))
        for t, kt in thresholds[:4]:
            if kt == k:
                hit = next((i for i, x in enumerate(run) if x >= t), None)
                lines.append((1, f"running total {k} first reaches {_fmt(round(t, 2))} at value "
                                 f"#{hit + 1} ({_fmt(vs[hit])})" if hit is not None else
                                 f"running total {k} never reaches {_fmt(round(t, 2))}"))
    for m in list(_BYTES.finditer(state))[:4]:
        v, u = _num(m[1]), m[2].lower()
        if u in _BIN:
            lines.append((2, f"{m[0]} = {_fmt(round(v * _BIN[u] / 1e9, 3))} GB (decimal; "
                             f"1 {m[2]} = {_fmt(round(_BIN[u] / 1e9, 6))} GB)"))
    return lines


def _cmp(a: float, b: float) -> str:
    return ">" if a > b else "<" if a < b else "="


def facts(state_text: str, question_texts: list[str], reference_time=None) -> list[str]:
    ref = parse_reference(reference_time)
    q_text = "\n".join(question_texts)
    lines: list[tuple[int, str]] = []  # (priority, text); lower = more relevant
    if ref:
        lines.append((0, f"reference time {ref.date().isoformat()} ({ref.strftime('%A')})"))

    s_dates, q_dates = dates(state_text, ref), dates(q_text, ref)
    for d, s in s_dates[:8]:
        rel = ""
        if ref:
            n = (d - ref.date()).days
            rel = (f", {abs(n)} days {'after' if n > 0 else 'before'} reference" if n
                   else ", same day as reference")
        lines.append((2, f"date {d.isoformat()} ({d.strftime('%a')}) from \"{s}\"{rel}"))
    for dq, sq in q_dates[:4]:
        for ds, ss in s_dates[:6]:
            lines.append((1, f"\"{ss}\" ({ds.isoformat()}) {'before' if ds < dq else 'after' if ds > dq else 'same day as'} "
                             f"\"{sq}\" ({dq.isoformat()}) in question, {abs((dq - ds).days)} days apart"))

    s_q, q_q = quantities(state_text), quantities(q_text)
    for vq, kq, sq in q_q[:6]:
        for vs, ks, ss in s_q[:12]:
            if ks == kq:
                lines.append((1, f"{_fmt(vs)} {ks} (state \"{ss}\") {_cmp(vs, vq)} "
                                 f"{_fmt(vq)} {kq} (question \"{sq}\")"))
    by_kind: dict[str, list] = {}
    for v, k, s in s_q:
        by_kind.setdefault(k, []).append(v)
    for k, vs in by_kind.items():
        if len(vs) >= 2:
            lines.append((3, f"state {k} values: {', '.join(_fmt(v) for v in vs[:6])}; "
                             f"total {_fmt(sum(vs))}, max {_fmt(max(vs))}, min {_fmt(min(vs))}"))

    lines += _v02(state_text, s_q, q_q)

    seen, out = set(), []
    for _, text in sorted(lines, key=lambda x: x[0]):
        if text not in seen:
            seen.add(text)
            out.append(text)
    return [f"F{i}: {t}" for i, t in enumerate(out[:MAX_FACTS], 1)]
