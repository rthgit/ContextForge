from __future__ import annotations

import os

os.environ["CONTEXTFORGE_ENABLE_MODEL"] = "0"
os.environ["CONTEXTFORGE_SKIP_UI_BUILD"] = "1"

import app


BASE = {
    "project_idea": "Build an issue triage agent.",
    "target_user": "Engineering teams",
    "build_target": "Agent workflow",
    "risk_level": "High",
    "output_language": "English",
    "user_context": "Reports may be incomplete.",
    "project_context": "Reduce triage time.",
    "technical_context": "Python and structured JSON.",
    "constraints": "Do not invent evidence.",
    "inputs_files": "Bug reports and logs.",
    "output_contract": "Return a prioritized ticket with evidence.",
    "failure_modes": "Hallucinated root cause.",
    "verification_criteria": "All ticket fields and evidence exist.",
}


def compile_for(topology: str) -> tuple[str, str, str, str, str, str]:
    return app.compile_context(
        BASE["project_idea"],
        BASE["target_user"],
        BASE["build_target"],
        topology,
        BASE["risk_level"],
        BASE["output_language"],
        app.REASONING_LAYERS,
        BASE["user_context"],
        BASE["project_context"],
        BASE["technical_context"],
        BASE["constraints"],
        BASE["inputs_files"],
        BASE["output_contract"],
        BASE["failure_modes"],
        BASE["verification_criteria"],
    )


def main() -> None:
    analysis = app.analyze_intake(BASE)
    topology = app.decide_topology(analysis, "Cascade")
    vital = app.extract_vital_structure(analysis, topology)
    reasoning = app.select_reasoning_architecture(analysis, topology, app.REASONING_LAYERS)
    pack = app.generate_prompt_pack(analysis, topology, vital, reasoning, BASE)
    qa = app.qa_repair_pass(pack)
    final = app.assemble_final_output(analysis, topology, vital, reasoning, qa)
    assert qa["pass"]
    assert final["prompt_pack"]

    expected_counts = {
        "Single Prompt": 1,
        "Cascade": 4,
        "Context Pack": 2,
        "Agent Workflow": 4,
    }
    for topology_name, expected_count in expected_counts.items():
        _, _, prompt_text, _, qa_text, runtime = compile_for(topology_name)
        assert prompt_text.count("[ROLE]") == expected_count
        for tag in app.REQUIRED_PROMPT_TAGS:
            assert prompt_text.count(f"[{tag}]") == expected_count
        assert "reveal your chain of thought" not in prompt_text.lower()
        assert "strategy | upside | risk | cost | selected" in prompt_text
        assert "No Chain Of Thought Leakage" in qa_text
        assert runtime.count("deterministic_fallback") >= 7

    print("ContextForge QA passed.")


if __name__ == "__main__":
    main()
