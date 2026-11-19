"""Output checks (arc/outchecks.py): number parsing, running a check, and the serve-path wiring.
Offline: no model calls."""

from decimal import Decimal

from arc import config, outchecks, serve
from arc.cache import Cache


def test_number_reads_common_forms():
    assert outchecks.number("$329,000") == 329000.0
    assert outchecks.number("$150k") == 150000.0
    assert outchecks.number("$2.5 million") == 2.5e6
    assert outchecks.number("7.5%") == 7.5
    assert outchecks.number("10 years") == 10.0
    assert outchecks.number("ten years") == 10.0
    assert outchecks.number("") is None
    assert outchecks.number(None) is None
    assert outchecks.number(Decimal("12.5")) == 12.5
    assert outchecks.number("$150K") == 150000.0
    assert outchecks.number("$2.5m") == 2.5e6
    assert outchecks.number("$3bn") == 3e9


def test_number_does_not_read_units_as_scale():
    assert outchecks.number("294.65 K") == 294.65          # kelvin, not thousand
    assert outchecks.number("5 m") == 5.0                  # metres
    assert outchecks.number("5m long") == 5.0              # metres: no currency, no million
    assert outchecks.number("12 kg") == 12.0
    assert outchecks.number("3 km") == 3.0
    assert outchecks.number("0.897 J/(g·°C)") == 0.897


SOURCE = '''
def check(inputs, outputs):
    bad = []
    bv = outputs.get("book value")
    cost = number(inputs.get("cost"))
    if isinstance(bv, float) and cost is not None and bv > cost + 0.01:
        bad.append(f"book value {bv} above cost {cost}")
    return bad
'''


def _cache(source=SOURCE, status="ok"):
    c = Cache("t")
    c.checks["e1"] = {"source": source, "status": status}
    return c


def test_run_passes_and_fails():
    c = _cache()
    assert outchecks.run(c, "e1", {"cost": "$1,000"}, {"book value": Decimal("400")}) == []
    broken = outchecks.run(c, "e1", {"cost": "$1,000"}, {"book value": Decimal("4000")})
    assert broken and "above cost" in broken[0]


def test_run_none_without_a_usable_check():
    assert outchecks.run(Cache("t"), "e1", {}, {}) is None
    assert outchecks.run(_cache(status="flagged_good_answer"), "e1", {}, {}) is None


def test_crashing_check_counts_as_broken():
    c = _cache("def check(inputs, outputs):\n    return 1 / 0\n")
    broken = outchecks.run(c, "e1", {}, {"x": 1.0})
    assert broken and broken[0].startswith("check crashed")


def test_check_cannot_import_os():
    c = _cache("import os\ndef check(inputs, outputs):\n    return []\n")
    broken = outchecks.run(c, "e1", {}, {})
    assert broken and "check crashed" in broken[0]


def test_serve_helper_is_off_by_default(monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_CHECKS", False)
    ep = type("EP", (), {"problem_id": "e1"})()
    trace = {}
    assert serve._output_check(_cache(), ep, {"cost": "$1"}, {"book value": 99.0}, trace) == []
    assert "output_check" not in trace


def test_serve_helper_reports_violations(monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_CHECKS", True)
    ep = type("EP", (), {"problem_id": "e1"})()
    trace = {}
    broken = serve._output_check(_cache(), ep, {"cost": "$1"}, {"book value": 99.0}, trace)
    assert broken and trace["output_check"] == "fail" and trace["output_violations"]


def test_cache_saves_and_loads_checks(tmp_path):
    c = _cache()
    c.save(tmp_path)
    assert Cache.load(tmp_path).checks["e1"]["source"] == SOURCE


def test_evidence_may_skip_a_few_words_but_not_jump():
    from arc.extraction import check_evidence_value as check
    q = "What are the monthly payments on a 20-year, $722,000 home loan at 6.5% per year?"
    assert check("a $722,000 home loan", "$722,000", q) is None          # dropped "20-year,"
    assert check("$722,000 home loan", "$722,000", q) is None            # exact
    assert check("a home loan at 6.5%", "6.5%", q) is not None           # skips too much
    assert check("a $722,000 loan", "$999", q) is not None               # number not in its evidence


def test_filter_accepts_numbers_written_as_words():
    from arc.chat import is_reusable
    assert is_reusable("Eight people sit at a circular table. How many seating arrangements are possible?")
    assert is_reusable("What pressure do 10 moles of methane exert in a 3 litre flask?")
    assert not is_reusable("thanks, that helps a lot")
    assert not is_reusable("can you explain that more simply please?")
