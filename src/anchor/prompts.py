"""Prompt templates — frozen Monday Nov 23. A paraphrased set can live in
this same file as a second named variant (Talha's job) without touching
the pipeline code.

Rules enforced here:
- BLIND/INFORMED/DECOY are identical except for one line.
- INFORMED and DECOY use *identical* wording for the price line. The
  model is never told whether the price is real or random.
- Same snapshot date in all three conditions.
- Asks for a single probability 0..100.
"""
from __future__ import annotations

from dataclasses import dataclass

CONDITIONS = ("blind", "informed", "decoy")

_BASE = """You are forecasting a binary event.

Question: {question}
Snapshot date: {snapshot_date}
{price_line}
Based on the information above, estimate the probability that the event
resolves YES. Reply with a single integer from 0 to 100 representing the
percent probability. Do not include any other text.

Probability (0-100):"""

_PRICE_LINE = "Current market price (implied probability, 0-100): {price}"


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    condition: str
    shown_price: int | None  # None for blind


def render(
    *,
    condition: str,
    question: str,
    snapshot_date: str,
    real_price: int | None,
    decoy_price: int | None,
) -> RenderedPrompt:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

    if condition == "blind":
        price_line = ""
        shown = None
    elif condition == "informed":
        if real_price is None:
            raise ValueError("informed condition requires real_price")
        shown = int(real_price)
        price_line = _PRICE_LINE.format(price=shown)
    else:  # decoy
        if decoy_price is None:
            raise ValueError("decoy condition requires decoy_price")
        shown = int(decoy_price)
        price_line = _PRICE_LINE.format(price=shown)

    text = _BASE.format(
        question=question.strip(),
        snapshot_date=snapshot_date,
        price_line=price_line,
    )
    # Collapse the blank line when blind so spacing stays consistent.
    text = text.replace("\n\n\nBased on", "\n\nBased on")
    return RenderedPrompt(text=text, condition=condition, shown_price=shown)
