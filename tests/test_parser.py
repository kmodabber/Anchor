from anchor.parser import parse_probability


def test_plain_integer():
    assert parse_probability("65") == 65


def test_percent_sign():
    assert parse_probability("65%") == 65


def test_decimal_percent():
    assert parse_probability("65.5%") == 65.5


def test_word_percent():
    assert parse_probability("around 65 percent") == 65


def test_fraction_form():
    assert parse_probability("0.65") == 65.0


def test_fraction_with_prefix():
    assert parse_probability("Probability: .72") == 72.0


def test_extra_text():
    assert parse_probability("My best estimate is 42% based on priors.") == 42


def test_several_numbers_prefers_percent():
    # "in 2024" has 2024 (out of range) then 60% — should land on 60.
    assert parse_probability("Given data from 2024, I estimate 60%.") == 60


def test_out_of_range_integer_returns_first_in_range():
    assert parse_probability("12345 then 50") == 50


def test_empty():
    assert parse_probability("") is None


def test_none():
    assert parse_probability(None) is None


def test_unparsable():
    assert parse_probability("I cannot provide a guess.") is None


def test_leading_zero_fraction():
    assert parse_probability("0.05 maybe") == 5.0


def test_integer_zero():
    assert parse_probability("0") == 0


def test_integer_hundred():
    assert parse_probability("100%") == 100
