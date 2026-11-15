"""arc/checks.py: the model-free number check used by certificates and judges."""

import pytest

from arc.checks import number_check, numbers


def test_minus_before_currency_symbol_is_kept():
    # regression: "-£62,164.50" was read as +62164.5, so a correct NPV was rejected
    assert numbers("NPV = -£62,164.50") == [-62164.50]
    assert number_check("net present value: -62164.50", "NPV = -£62,164.50 (reject)")["status"] == "match"
    # Changed 2026-11-15 with the owner's exact-amount rule: this used to "match" under the 1%
    # floor, but -62,164.5075 is -£62,164.51 to the cent, one cent off the stated answer.
    assert number_check("net present value: -62164.50746433773", "NPV = -£62,164.50 (reject)")["status"] == "mismatch"


def test_currency_after_sign_variants():
    assert numbers("−$1,200") == [-1200.0]
    assert numbers("- € 300") == [-300.0]
    assert numbers("+$5") == [5.0]


def test_sign_mismatch_is_still_a_mismatch():
    assert number_check("NPV: 62164.5", "NPV = -£62,164.50")["status"] == "mismatch"


def test_scientific_and_prefix_scaling_still_work():
    assert numbers("1.2 × 10⁶") == [1.2e6]
    assert number_check("6.563e+03 Pa", "p = 6.56 kPa")["status"] == "match"


# ----------------------------------------------------------------------------- worked references
# Regression: a production response is full working. final_line() took its FIRST paragraph
# (the givens), so the chat loop's gas family got a false "mismatch" (record f4a9efff6e95).

GAS_RESPONSE = (
    '**Given:** n = 7.41 mol, V = 12.09 L, T = 96 °C\n\n**Step 1: Convert temperature to kelvin**\n'
    'T = 96 + 273.15 = 369.15 K\n\n**Step 2: Apply the ideal gas law**\n$$P = \\frac{nRT}{V}$$\n\n'
    'Using R = 0.082057 L·atm·mol⁻¹·K⁻¹:\n\n$$P = \\frac{(7.41)(0.082057)(369.15)}{12.09} = '
    '\\frac{224.46}{12.09} \\approx 18.57\\ \\text{atm}$$\n\n**Check in SI units:** With R = 8.314 '
    'J·mol⁻¹·K⁻¹ and V = 0.01209 m³, P ≈ 1.88 × 10⁶ Pa, which matches 18.57 atm × 101325 Pa.\n\n'
    '**Final answer:** P ≈ 18.6 atm (≈ 1.88 MPa)')


def test_worked_production_response_matches_its_resolve():
    assert number_check("18.57 atm", GAS_RESPONSE)["status"] == "match"
    assert number_check("42.0 atm", GAS_RESPONSE)["status"] == "mismatch"   # an input value must not match


def test_final_line_prefers_the_labelled_answer():
    from arc.checks import final_line
    assert final_line(GAS_RESPONSE) == "**Final answer:** P ≈ 18.6 atm (≈ 1.88 MPa)"
    assert final_line("Working: 2 + 2\n\nAnswer: 4\n\nHope that helps!") == "Answer: 4"
    assert final_line("x = 3\n## Final Answer\n\n**12 kg**") == "**12 kg**"          # label on its own line
    assert final_line("so the result is $\\boxed{7.5}$ m.") == "so the result is $\\boxed{7.5}$ m."


def test_final_line_without_a_label_uses_first_and_last_numbered_paragraphs():
    from arc.checks import final_line
    text = "n = 2 mol, V = 3 L\n\nP = 16.4 atm\n\nLet me know if you need more."
    assert final_line(text) == "n = 2 mol, V = 3 L\n\nP = 16.4 atm"


# Shapes seen in real production responses (chat-loop backlog, 2026-11-15).
ANSWER_FIRST = ("**Net book value at the end of Year 2: $99,562.50**\n\n**Setup:** DDB rate 25%.\n\n"
                "**Year 1:**\n- Ending book value = $132,750\n\n"
                "Book value remains above the $19,600 salvage value, so no adjustment is needed.")
PLURAL_BLOCK = ("**Step 4: Deadweight loss**\n\nDWL = 15\n\n**Answers**\n\n"
                "- **New price buyers pay:** $33 per unit\n- **Deadweight loss:** $15")


def test_answer_first_response_still_matches():
    assert number_check("99562.5", ANSWER_FIRST)["status"] == "match"
    assert number_check("70000", ANSWER_FIRST)["status"] == "mismatch"


def test_plural_answer_heading_takes_the_whole_block():
    from arc.checks import final_line
    assert final_line(PLURAL_BLOCK) == "- **New price buyers pay:** $33 per unit\n- **Deadweight loss:** $15"
    assert final_line("**Final answers:**\n- Payment: $2,176.86\n- Interest: $415,669") == \
        "- Payment: $2,176.86\n- Interest: $415,669"
    assert final_line("**Result:** $P \\approx 1.25$ MPa") == "**Result:** $P \\approx 1.25$ MPa"


def test_short_golden_labels_are_unchanged():
    from arc.checks import final_line
    for label in ["50.7 atm", "NPV = -£62,164.50 (reject)", "p = 6.56 kPa",
                  "Requires numerical integration with the drag model; the drag-free answer (81.0 m) is an overestimate"]:
        assert final_line(label) == label


def test_compare_tolerance_comes_from_the_final_answer_not_the_working(monkeypatch):
    # 19.4 atm is 4.3% from 18.6: inside the 5% that "Step 1" in the working gave, outside 1%.
    from types import SimpleNamespace
    from arc import certify, judges
    monkeypatch.setattr(judges, "judge", lambda *a, **k: SimpleNamespace(passed=True, critique=""))
    assert certify._compare("t", "q", GAS_RESPONSE, "18.57 atm", "t").agree
    assert not certify._compare("t", "q", GAS_RESPONSE, "19.4 atm", "t").agree


# ----------------------------------------------------------------------------- exact to the written precision
# Owner rule (2026-11-15): "If an answer requires an exact amount, then it requires an exact amount."
# Two numbers agree only to the precision they are written in (half a unit in the last stated
# digit of the coarser one). No 1% floor, no 5% cap. Regression: the chat run's mortgage
# production answer $5,053.27 "matched" the right value $5,053.07 under the old 1% floor.

@pytest.mark.parametrize("answer,reference,status", [
    ("$5,053.07", "$5,053.27", "mismatch"),          # money to the cent
    ("monthly_payment: 5053.067812", "$5,053.07", "match"),
    ("monthly_payment: 5053.067812", "$5,053.27", "mismatch"),
    ("18.57 atm", "18.6 atm", "match"),               # coarser is 18.6 -> +-0.05
    ("19.4 atm", "18.6 atm", "mismatch"),
    ("18.64 atm", "18.6 atm", "match"),
    ("18.66 atm", "18.6 atm", "mismatch"),
    ("1,009 people", "1,008 people", "mismatch"),
    ("1008", "1,008", "match"),
    ("1.88 × 10⁶ Pa", "1.884e6 Pa", "match"),         # mantissa decimals: +-0.005e6
    ("1.88 × 10⁶ Pa", "1.89e6 Pa", "mismatch"),
    ("5132.1 kPa", "5.13 MPa", "match"),              # SI prefix scaling still allowed
    ("5150 kPa", "5.13 MPa", "mismatch"),
    ("0.30000000000000004", "0.3", "match"),          # float noise
])
def test_agreement_is_exact_to_the_written_precision(answer, reference, status):
    assert number_check(answer, reference)["status"] == status


def test_precision_is_per_pair_so_a_rough_aside_cannot_loosen_it():
    # "(about 51 bar)" must not widen the tolerance for the 50.68 atm answer
    assert number_check("50.60 atm", "P = 50.68 atm (about 51 bar)")["status"] == "mismatch"
    assert number_check("50.68 atm", "P = 50.68 atm (about 51 bar)")["status"] == "match"


def test_a_conflicting_value_is_a_mismatch_even_if_another_number_matches():
    ref = "Monthly payment: $5,053.27; total interest: $95,392"
    assert number_check("payment 5053.27, interest 95368.14", ref)["status"] == "mismatch"
    assert number_check("payment 5053.27, interest 95392.00", ref)["status"] == "match"
    # an unrelated number in the reference (the year) is not a conflict
    assert number_check("54250", "Book value after 2 years = $54,250")["status"] == "match"


def test_written_numbers_report_their_precision():
    from arc.checks import written_numbers
    got = {(w.value, w.half_unit) for w in written_numbers("$5,053.27, 1008 people, 1.88 × 10⁶ Pa, 6.563e+03")}
    assert got == {(5053.27, 0.005), (1008.0, 0.5), (1.88e6, 5e3), (6563.0, 0.5)}


def test_values_agree_helper():
    from arc.checks import values_agree
    assert values_agree("18.57", "18.6") and not values_agree("19.4", "18.6")
    assert values_agree("$5,053.07", "5053.0678") and not values_agree("5053.27", "5053.07")
    assert values_agree("5.13 MPa", "5132.1 kPa")


def test_compare_vetoes_a_cent_level_money_difference(monkeypatch):
    # the judge alone stopped this in the chat run; the number check must stop it too
    from types import SimpleNamespace
    from arc import certify, judges
    monkeypatch.setattr(judges, "judge", lambda *a, **k: SimpleNamespace(passed=True, critique=""))
    q = "A $350,000 loan at 6% for 7 years: monthly payment and total interest?"
    assert not certify._compare("t", q, "Payment: $5,053.27; total interest $95,392", "$5,053.07 and $95,368.14", "t").agree
    assert certify._compare("t", q, "Payment: $5,053.07; total interest $95,368.14", "$5,053.07 and $95,368.14", "t").agree


def test_cert_judge_is_told_the_precision_rule(monkeypatch):
    from types import SimpleNamespace
    from arc import certify, judges
    seen = {}
    monkeypatch.setattr(judges, "judge", lambda prompt, **k: seen.update(prompt=prompt, **k) or
                        SimpleNamespace(passed=True, critique=""))
    certify._compare("t", "q", "18.6 atm", "18.57 atm", "t")
    text = seen["prompt"] + (seen.get("system") or "")
    assert "precision" in text and "to the cent" in text and "unit conversion" in text.lower()
    assert "up to rounding" not in text


def test_money_is_never_rescaled():
    # chat run: the mortgage entry read "$150k" as 150 and answered 0.92 instead of $921.13
    assert number_check("monthly_payment: 0.92", "Monthly payment: $921.13")["status"] == "mismatch"
    assert number_check("monthly_payment: 0.92113", "**$921.13**")["status"] == "mismatch"
    from arc.checks import values_agree
    assert not values_agree("0.92", "$921.13")


def test_rescaling_needs_a_written_si_unit():
    assert number_check("5132.1", "5.13")["status"] == "mismatch"           # no unit: no rescaling
    assert number_check("5132.1 kPa", "5.13 MPa")["status"] == "match"
    assert number_check("pressure_pa: 6563", "p = 6.56 kPa")["status"] == "match"


def test_paper_judges_keep_their_instructions(fake_llm):
    # paper mode must not change: judges called without `system` get the old text byte for byte
    from arc import judges
    fake_llm.queue(judges.Verdict(critique="", passed=True))
    judges.judge("q", role="judge")
    assert fake_llm.calls[0]["system"] == judges.DEFAULT_SYSTEM
    assert "agree up to rounding and unit conversion" in judges.DEFAULT_SYSTEM


def test_latex_dollar_signs_are_not_money_and_latex_units_are_read():
    from arc.checks import written_numbers
    assert [w.unit for w in written_numbers(r"$$p \approx 5.13\ \text{MPa} \;(\approx 50.7\ \text{atm})$$")] == ["mpa", "atm"]
    ref = "p = 5.13 MPa (about 5.13 x 10^6 Pa, i.e. 51.3 bar)"
    cand = r"$$p \approx 5.13\ \text{MPa} \;(\approx 5.13 \times 10^{6}\ \text{Pa}\approx 50.7\ \text{atm})$$"
    assert number_check(cand, ref)["status"] == "match"      # 51.3 bar vs 50.7 atm is not a conflict
    assert number_check("$911.56", "$911.55")["status"] == "mismatch"   # real money still is


def test_units_must_be_compatible():
    assert number_check("50.6 atm", "51 bar")["status"] == "mismatch"
    assert number_check("pressure in atm: 50.68", "50.7 atm")["status"] == "match"   # unit only on one side
    assert number_check("317.46 mm", "0.317 m (about 318 mm, i.e. 317.5 mm)")["status"] == "match"
