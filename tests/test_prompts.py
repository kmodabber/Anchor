from anchor.prompts import CONDITIONS, render


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
