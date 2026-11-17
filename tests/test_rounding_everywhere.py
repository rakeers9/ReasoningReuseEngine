"""Every prompt that produces a final answer carries the one rounding policy, verbatim;
the paper's own prompts stay byte for byte (one rounding policy, arc/present.py)."""

from pydantic import BaseModel

from arc import certify, chat, explain, present, source


def test_the_chatbot_instructions_carry_the_policy():
    assert present.ROUNDING_POLICY in chat.PRODUCTION_SYSTEM


def test_the_explainer_blind_solve_carries_the_policy(fake_llm):
    fake_llm.queue(explain.Worked(working="w", final_answer="7.759 bar"))
    explain._blind_solve("What is the pressure?", "t")
    prompt = fake_llm.calls[0]["prompt"]
    assert present.ROUNDING_POLICY in prompt
    assert prompt.startswith(source.SOLVE_PROMPT.format(question="What is the pressure?"))   # paper prompt first, intact


def test_the_certificate_fresh_solve_carries_the_policy(fake_llm):
    fake_llm.queue(certify.Solved(working="w", final_answer="7.759 bar"))
    certify._fresh_solve("What is the pressure?", "t")
    prompt = fake_llm.calls[0]["prompt"]
    assert present.ROUNDING_POLICY in prompt
    assert prompt.startswith(source.SOLVE_PROMPT.format(question="What is the pressure?"))


def test_the_papers_solve_prompt_does_not_carry_it():
    assert "Rounding policy" not in source.SOLVE_PROMPT


def test_label_free_prompts_import_the_same_text():
    from arc import builder, labelfree
    assert labelfree.DEFAULT_CONVENTION == present.ROUNDING_POLICY
    assert present.ROUNDING_POLICY in builder.STRICT_UNITS_RULES          # as context: the code itself never rounds
