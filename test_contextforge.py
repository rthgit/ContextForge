from __future__ import annotations

import os
from unittest.mock import patch

os.environ["CONTEXTFORGE_ENABLE_MODEL"] = "0"
os.environ["CONTEXTFORGE_OPENBMB_ENABLE"] = "0"
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
    assert "pending" in app.render_metrics({}).lower()
    fast_updates = app.update_mode("Fast Compile")
    full_updates = app.update_mode("Full Control")
    assert all(update.get("visible") is False for update in fast_updates)
    assert all(update.get("visible") is True for update in full_updates)
    assert app.detect_generation_issue("", "intake_analysis") == "empty decoded continuation"
    assert "whitespace-only" in app.detect_generation_issue("   ", "intake_analysis")
    assert "special-token-only" in app.detect_generation_issue(
        "<eos>",
        "intake_analysis",
        generated_token_ids=[2],
        special_token_ids={2},
    )
    assert "repeated special tokens" in app.detect_generation_issue(
        "<|eos|><|eos|><|eos|><|eos|>",
        "intake_analysis",
        raw_with_special_tokens="<|eos|><|eos|><|eos|><|eos|>",
    )
    assert "output too short" in app.detect_generation_issue('{"ok": true}', "prompt_pack_generation")
    assert "non-target-language" in app.detect_generation_issue(
        '{"result": "' + ("测试输出" * 40) + '"}',
        "intake_analysis",
        "English",
    )
    assert app.detect_generation_issue(
        '{"domain":"software engineering","task_type":"design and implementation planning","risk_level":"medium"}',
        "intake_analysis",
        "English",
    ) is None
    assert app.detect_generation_issue(
        '{"dominio":"ingegneria software","tipo_attivita":"progettazione e implementazione","rischio":"medio"}',
        "intake_analysis",
        "Italiano",
    ) is None

    openbmb = app.RuntimeCandidate("openbmb_lightweight", "openbmb/MiniCPM5-1B", "openbmb_minicpm5")
    existing = app.RuntimeCandidate("public_cpu", "Qwen/test", "small_model")
    with patch.object(app, "runtime_candidates", return_value=[openbmb, existing]), patch.object(
        app,
        "generate_with_candidate",
        side_effect=[(None, "blank output"), ({"domain": "software engineering"}, "loaded")],
    ):
        candidate, attempted, note, source = app.generate_json("intake_analysis", "Classify.", BASE)
        assert candidate == {"domain": "software engineering"}
        assert attempted == "openbmb/MiniCPM5-1B -> Qwen/test"
        assert "blank output" in note
        assert source == "small_model"

    original_openbmb_enabled = app.OPENBMB_ENABLED
    original_model_enabled = app.MODEL_ENABLED
    try:
        app.OPENBMB_ENABLED = True
        app.MODEL_ENABLED = False
        app._RUNTIME_TRACE.clear()
        with patch.object(
            app,
            "load_candidate_model",
            return_value=(None, None, "simulated OpenBMB unavailable"),
        ):
            app.analyze_intake(BASE)
        runtime_row = app._RUNTIME_TRACE[-1]
        assert runtime_row["source"] == "deterministic_fallback"
        assert "openbmb/MiniCPM5-1B" in runtime_row["model_attempted"]
        assert "simulated OpenBMB unavailable" in runtime_row["fallback_reason"]
    finally:
        app.OPENBMB_ENABLED = original_openbmb_enabled
        app.MODEL_ENABLED = original_model_enabled
        app._RUNTIME_TRACE.clear()

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
        assert "| Stage | Model attempted | Source | Fallback reason | Duration ms |" in runtime

    print("ContextForge QA passed.")


if __name__ == "__main__":
    main()
