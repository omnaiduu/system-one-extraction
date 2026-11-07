"""Read marked words the rules cannot: plain Gemma rewrites them, code checks the reading.

Used only after the tagger has marked a part and the parser has refused it
("one lakh twenty thousand", "the second of April, 2026", "fourteen").

1. Gemma, with the LoRA adapters switched off, gets a short prompt holding
   only the marked words, never the note, so it cannot pick another value
   from the document. Its job is the format, nothing else.
2. Decoding is forced: at every step only tokens made of digits, "." and "-"
   (or the end) are allowed, so the output is a number or an ISO date or
   nothing. About 6 to 10 tokens.
3. The reading is checked backwards: code writes the value out in words
   (English and Indian systems, several decimal and date styles) and compares
   with the marked words, ignoring filler ("rupees", "only", "and", "the").
   No match: the part stays unread and the field is empty.

`agrees` and `greedy_constrained` are plain Python and run in the tests with
a fake model. `make_rewriter` wires them to Gemma on Kaggle.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
           "nine": "ninth", "twelve": "twelfth", "twenty": "twentieth", "thirty": "thirtieth"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
FILLER = {"and", "only", "rupees", "rupee", "rs", "inr", "dollars", "dollar", "usd", "cents", "cent",
          "paise", "paisa", "the", "of", "on", "a", "sum", "amount", "total", "exactly", "day", "days"}

PROMPTS = {
    "number": "Rewrite this amount as a plain number in digits, with a dot for decimals and no commas, "
              "words or currency. Reply with the number only.",
    "count": "Rewrite this count as a whole number in digits. Reply with the number only.",
    "date": "Rewrite this date as YYYY-MM-DD. Reply with the date only.",
}
ALLOWED_CHARS = set("0123456789.-")


def _under_thousand(n: int) -> list[str]:
    words = []
    if n >= 100:
        words += [ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(TENS[n // 10])
        if n % 10:
            words.append(ONES[n % 10])
    elif n or not words:
        words.append(ONES[n])
    return words


def _western(n: int) -> list[str]:
    if n == 0:
        return ["zero"]
    words = []
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            words += _western(n // size) + [name]
            n %= size
    if n:
        words += _under_thousand(n)
    return words


def _indian(n: int) -> list[str]:
    words = []
    for size, name in ((10**7, "crore"), (10**5, "lakh")):
        if n >= size:
            words += _indian(n // size) + [name]
            n %= size
    if n or not words:
        words += _western(n) if n else []
    return words or ["zero"]


def _integer_forms(n: int) -> list[list[str]]:
    forms = [_western(n), _indian(n)]
    # "twelve hundred" for 1,100-9,999 not divisible into thousands neatly.
    if 1100 <= n < 10000 and (n // 100) % 10:
        forms.append(_under_thousand(n // 100) + ["hundred"] + (_under_thousand(n % 100) if n % 100 else []))
    out = []
    for form in forms:
        if form not in out:
            out.append(form)
    return out


def words(text: str) -> list[str]:
    """Lower-case word tokens with hyphens split and filler dropped ("eighty-six" -> eighty six)."""
    tokens = re.findall(r"\d+(?:st|nd|rd|th)?|[a-z]+", text.lower().replace("-", " "))
    return [t for t in tokens if t not in FILLER]


def number_forms(value: Decimal) -> list[list[str]]:
    whole = int(abs(value))
    frac = int((abs(value) - whole) * 100 + Decimal("0.5"))
    sign = ["minus"] if value < 0 else []
    out = []
    for head in _integer_forms(whole):
        if frac == 0:
            out.append(sign + head)
        digits = f"{frac:02d}"
        point = [ONES[int(d)] for d in digits]
        out.append(sign + head + ["point"] + point)
        if digits.endswith("0"):
            out.append(sign + head + ["point", ONES[int(digits[0])]])
        if frac:
            out.append(sign + head + _western(frac))
    return out


def _ordinal(day: int) -> list[str]:
    base = _under_thousand(day)
    last = base[-1]
    if last in ORDINAL:
        last = ORDINAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last = last + "th"
    return base[:-1] + [last]


def date_forms(d: date) -> list[list[str]]:
    month = MONTHS[d.month - 1]
    days = [[str(d.day)], _ordinal(d.day), [f"{d.day}{_suffix(d.day)}"]]
    months = [[month], [month[:3]]] + ([["sept"]] if d.month == 9 else [])
    years = [[str(d.year)], _western(d.year)]
    if d.year >= 2000:
        years.append(["two", "thousand"] + (_western(d.year - 2000) if d.year > 2000 else []))
        years.append(_under_thousand(d.year // 100) + _under_thousand(d.year % 100))  # twenty twenty-six
    return [day + mon + year for day in days for mon in months for year in years]


def _suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def agrees(text: str, value, kind: str) -> bool:
    """Does `value` read back as the marked words? Order is ignored for dates (day/month can swap)."""
    got = words(text)
    if not got:
        return False
    if kind == "date":
        if not isinstance(value, date):
            return False
        return any(sorted(got) == sorted(form) for form in date_forms(value))
    if not isinstance(value, Decimal):
        return False
    digits = re.sub(r"[^\d.]", "", text)
    if digits and not re.search(r"[a-z]{3,}", text.lower()):
        return False  # plain digits the rules refused: no words to check against
    return any(got == form for form in number_forms(value))


def parse_output(out: str, kind: str):
    out = out.strip()
    try:
        if kind == "date":
            return date.fromisoformat(out) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", out) else None
        if not re.fullmatch(r"-?\d+(\.\d+)?", out):
            return None
        value = Decimal(out)
        if kind == "count" and value != value.to_integral_value():
            return None
        return value
    except (ValueError, InvalidOperation):
        return None


def greedy_constrained(next_logits, prompt_ids: list[int], allowed: list[int], stops: set[int], max_new: int = 12) -> list[int]:
    """Greedy decoding over `allowed` + `stops` only. `next_logits(ids)` -> 1-D logits for the next token."""
    ids, out = list(prompt_ids), []
    candidates = list(allowed) + sorted(stops)
    for _ in range(max_new):
        logits = next_logits(ids)
        best = max(candidates, key=lambda t: float(logits[t]))
        if best in stops:
            break
        out.append(best)
        ids.append(best)
    return out


def make_rewriter(model, tokenizer, device: str, stats: dict | None = None):
    """A `read(text, kind)` that asks plain Gemma (adapters off) and checks the answer."""
    import torch

    stats = stats if stats is not None else {}
    vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    allowed = [i for i, tok in enumerate(vocab) if tok and tok.replace("▁", "") and set(tok.replace("▁", "")) <= ALLOWED_CHARS]
    stops = {tokenizer.eos_token_id}
    for name in ("<end_of_turn>", "<turn|>", "\n"):
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            stops.add(tid)
    stops.update(i for i, tok in enumerate(vocab) if tok in ("\n", "Ċ", "<0x0A>"))
    layers = [m for m in model.modules() if hasattr(m, "enable_adapters") and hasattr(m, "lora_A")]
    cache: dict = {}

    def next_logits(ids):
        out = model(input_ids=torch.tensor([ids], device=device), use_cache=False)
        return out.logits[0, -1].float()

    def read(text: str, kind: str):
        key = (text, kind)
        if key in cache:
            return cache[key]
        stats["asked"] = stats.get("asked", 0) + 1
        messages = [{"role": "user", "content": f"{PROMPTS[kind]}\n\n{text}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for layer in layers:
            layer.enable_adapters(False)
        try:
            with torch.no_grad():
                new = greedy_constrained(next_logits, ids, allowed, stops)
        except Exception as exc:  # never let a reading failure end a training run
            stats["errors"] = stats.get("errors", 0) + 1
            stats["last_error"] = repr(exc)[:300]
            new = []
        finally:
            for layer in layers:
                layer.enable_adapters(True)
        out = tokenizer.decode(new).replace(" ", "")
        value = parse_output(out, kind)
        checked = value is not None and agrees(text, value, "date" if kind == "date" else "number")
        stats["parsed"] = stats.get("parsed", 0) + (value is not None)
        stats["accepted"] = stats.get("accepted", 0) + checked
        log = stats.setdefault("log", [])
        if len(log) < 200:
            log.append({"text": text, "kind": kind, "gemma": out, "accepted": checked})
        cache[key] = value if checked else None
        return cache[key]

    read.stats = stats
    return read
