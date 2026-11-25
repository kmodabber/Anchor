from scripts.drop_same_event import check, drop_rows

MARKETS = (
    "market_id,question,volume\n"
    'a,"Will X, Y happen?",100\n'
    "b,Will Z happen?,50\n"
    "c,Will W happen?,10\n"
)


def test_drop_rows_keeps_other_lines_byte_for_byte(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(MARKETS, encoding="utf-8")
    assert drop_rows(p, {"b"}) == 1
    # the quoted line with a comma is untouched, header and newline kept
    assert p.read_text(encoding="utf-8") == (
        "market_id,question,volume\n"
        'a,"Will X, Y happen?",100\n'
        "c,Will W happen?,10\n")


def test_drop_rows_is_a_no_op_when_already_applied(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(MARKETS, encoding="utf-8")
    drop_rows(p, {"b"})
    after = p.read_bytes()
    assert drop_rows(p, {"b"}) == 0
    assert p.read_bytes() == after


def _markets():
    return {"a": {"volume": "100"}, "b": {"volume": "50"}, "c": {"volume": "10"}}


def test_check_accepts_a_valid_list():
    drops = [{"market_id": "b", "kept_market_id": "a"},
             {"market_id": "c", "kept_market_id": "a"}]
    assert check(drops, _markets()) == []


def test_check_rejects_keeping_the_lower_volume_market():
    assert check([{"market_id": "a", "kept_market_id": "b"}], _markets())


def test_check_rejects_a_kept_market_that_is_also_dropped():
    drops = [{"market_id": "b", "kept_market_id": "a"},
             {"market_id": "a", "kept_market_id": "c"}]
    assert any("kept and as dropped" in p for p in check(drops, _markets()))


def test_check_rejects_a_missing_kept_market():
    assert check([{"market_id": "b", "kept_market_id": "zzz"}], _markets())
