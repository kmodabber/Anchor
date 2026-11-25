from pipeline.prompts import CONDITIONS, render


def _common():
    return dict(
        question="Will X happen by date Y?",
        snapshot_date="2024-10-01",
        real_price=70,
        decoy_price=30,
    )


def test_all_conditions_render():
    for c in CONDITIONS:
        r = render(condition=c, **_common())
        assert "Will X happen" in r.text
        assert "2024-10-01" in r.text
        assert "Probability (0-100):" in r.text


def test_blind_has_no_price():
    r = render(condition="blind", **_common())
    assert "market price" not in r.text.lower()
    assert r.shown_price is None


def test_informed_shows_real_price():
    r = render(condition="informed", **_common())
    assert "70" in r.text
    assert r.shown_price == 70


def test_decoy_shows_decoy_price():
    r = render(condition="decoy", **_common())
    assert "30" in r.text
    assert r.shown_price == 30


def test_informed_and_decoy_use_same_wording():
    i = render(condition="informed", **_common()).text
    d = render(condition="decoy", **_common()).text
    # Replace the integer and the strings should match exactly.
    assert i.replace("70", "X") == d.replace("30", "X")


def test_no_hint_of_real_vs_fake():
    for c in ("informed", "decoy"):
        r = render(condition=c, **_common())
        low = r.text.lower()
        for word in ("real", "fake", "decoy", "random", "actual", "true"):
            assert word not in low, f"{c} prompt leaks word: {word}"


# --- frozen wording and prompt variants -------------------------------------

import hashlib

import pytest

from pipeline.prompts import TEMPLATES

# sha256 of v1 output, recorded before variants were added. If one of these
# fails, the frozen main prompt changed: revert, don't update the hash.
V1_HASHES = {
    "blind": "8bb938c3e68e89649cc2fadae01a6cf91096102218b8170ff9375259dbaccced",
    "informed": "d9ac1c2fe20d4701990e2c43683ce7b85686eeb007eb52e60ca2dfb9ae7d0e5c",
    "decoy": "8097dcec5517866d7d48c63d39bb429c35e7ed493abc6b73c13f16e64f8fe46c",
}


@pytest.mark.parametrize("condition", CONDITIONS)
def test_v1_wording_is_frozen(condition):
    t = render(condition=condition, question="Will X happen by date Y?",
               snapshot_date="2025-10-01", real_price=70, decoy_price=30).text
    assert hashlib.sha256(t.encode()).hexdigest() == V1_HASHES[condition]


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="not defined"):
        render(condition="blind", variant="nope", **_common())


@pytest.mark.parametrize("variant", sorted(TEMPLATES))
def test_every_variant_follows_the_rules(variant):
    r = {c: render(condition=c, variant=variant, **_common()) for c in CONDITIONS}
    for c in CONDITIONS:
        assert "Will X happen" in r[c].text and "2024-10-01" in r[c].text
    assert r["blind"].shown_price is None and "70" not in r["blind"].text
    assert r["informed"].text.replace("70", "X") == r["decoy"].text.replace("30", "X")
    for c in ("informed", "decoy"):
        low = r[c].text.lower()
        for word in ("real", "fake", "decoy", "random", "actual", "true"):
            assert word not in low, f"{variant}/{c} prompt leaks word: {word}"
