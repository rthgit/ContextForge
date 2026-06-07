from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable


APP_TITLE = "ContextForge"
APP_SUBTITLE = "Agent Prompt Compiler"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MID_MODEL_ID = "RthItalia/nano_compact_3b_qkvfp16"
DEFAULT_HIGH_MODEL_ID = "Qwen/Qwen3-32B"
REQUIRED_PROMPT_TAGS = [
    "ROLE",
    "COGNITIVE_LAYERS",
    "KAHNEMAN_SYSTEM2",
    "PARETO_80_20",
    "VITAL_SPOT",
    "REASONING_PROTOCOL",
    "AGENTIC_LOOP",
    "ACTION",
    "FORMAT_AND_TARGET",
    "QA_CHECKS",
]
TOPOLOGIES = ["Auto", "Single Prompt", "Cascade", "Context Pack", "Agent Workflow"]
REASONING_LAYERS = [
    "CRAFT",
    "Kahneman System 2",
    "Pareto 80/20",
    "Agentic Loop",
    "Tree of Thought controlled",
    "Private CoT",
    "Self-Correction",
    "Sentinel Recovery",
]
STAGE_NAMES = [
    "intake_analysis",
    "topology_decision",
    "vital_structure",
    "reasoning_architecture",
    "prompt_pack_generation",
    "qa_repair",
    "final_assembly",
]
STAGE_TOKEN_BUDGETS = {
    "intake_analysis": 180,
    "topology_decision": 140,
    "vital_structure": 180,
    "reasoning_architecture": 240,
    "prompt_pack_generation": 520,
    "qa_repair": 260,
    "final_assembly": 260,
}


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


MODEL_ENABLED = parse_bool_env("CONTEXTFORGE_ENABLE_MODEL", False)
MODEL_ID = os.getenv("CONTEXTFORGE_MODEL_ID", DEFAULT_MODEL_ID)
MID_MODEL_ID = os.getenv("CONTEXTFORGE_MID_MODEL_ID", DEFAULT_MID_MODEL_ID)
HIGH_MODEL_ID = os.getenv("CONTEXTFORGE_HIGH_MODEL_ID", DEFAULT_HIGH_MODEL_ID)
MAX_NEW_TOKENS = parse_int_env("CONTEXTFORGE_MAX_NEW_TOKENS", 1800, 256, 4096)
MAX_INPUT_CHARS = parse_int_env("CONTEXTFORGE_MAX_INPUT_CHARS", 12000, 2000, 40000)


@dataclass
class StageResult:
    data: dict[str, Any]
    source: str
    model_id: str
    elapsed_ms: int
    note: str = ""

    def runtime_row(self, stage: str) -> dict[str, Any]:
        return {
            "stage": stage,
            "source": self.source,
            "model_id": self.model_id,
            "elapsed_ms": self.elapsed_ms,
            "note": self.note,
        }


_RUNTIME_TRACE: list[dict[str, Any]] = []


def clean_text(value: Any, limit: int = 4000) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit]


def clean_list(value: Any, limit: int = 8) -> list[str]:
    if isinstance(value, str):
        candidates = re.split(r"[,;\n]+", value)
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []
    result = []
    for item in candidates:
        cleaned = clean_text(item, 240)
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result[:limit]


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def parse_json_object(raw: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw or ""):
        try:
            parsed, _ = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def merge_known(fallback: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return fallback
    merged = dict(fallback)
    for key, fallback_value in fallback.items():
        candidate_value = candidate.get(key)
        if candidate_value is None:
            continue
        if isinstance(fallback_value, list):
            items = clean_list(candidate_value, max(3, len(fallback_value) + 3))
            if items:
                merged[key] = items
        elif isinstance(fallback_value, dict) and isinstance(candidate_value, dict):
            merged[key] = {**fallback_value, **candidate_value}
        elif isinstance(fallback_value, int):
            try:
                merged[key] = int(candidate_value)
            except (TypeError, ValueError):
                pass
        else:
            cleaned = clean_text(candidate_value, 16000)
            if cleaned:
                merged[key] = cleaned
    return merged


def model_candidates() -> list[tuple[str, str, bool]]:
    candidates = [
        ("high", HIGH_MODEL_ID, True),
        ("mid", MID_MODEL_ID, True),
        ("public_cpu", MODEL_ID, False),
    ]
    seen: set[str] = set()
    return [
        item
        for item in candidates
        if item[1].strip() and not (item[1] in seen or seen.add(item[1]))
    ]


@lru_cache(maxsize=1)
def load_model() -> tuple[Any | None, Any | None, str, str]:
    if not MODEL_ENABLED:
        return None, None, "disabled", "model disabled by CONTEXTFORGE_ENABLE_MODEL"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return None, None, "unavailable", f"dependencies unavailable: {type(exc).__name__}: {exc}"

    failures: list[str] = []
    for role, candidate_id, requires_cuda in model_candidates():
        if requires_cuda and not torch.cuda.is_available():
            failures.append(f"{role}: CUDA unavailable")
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate_id, trust_remote_code=True, use_fast=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
            if torch.cuda.is_available():
                kwargs["device_map"] = "cuda"
                kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(candidate_id, **kwargs)
            model.eval()
            return tokenizer, model, candidate_id, f"selected {role}; " + "; ".join(failures)
        except Exception as exc:
            failures.append(f"{role}: {type(exc).__name__}: {exc}")
    return None, None, "unavailable", " | ".join(failures) or "no model candidates"


def format_chat_prompt(tokenizer: Any, stage: str, instruction: str, payload: dict[str, Any]) -> str:
    system = (
        "You are one isolated module inside ContextForge, an agent prompt compiler. "
        "Return only a valid JSON object. Private reasoning internal only. "
        "Never reveal chain of thought, hidden branches, or internal deliberation. "
        "Public fields may contain only decision summaries, assumptions, risks, verification steps, and outputs."
    )
    user = f"MODULE: {stage}\nTASK:\n{instruction}\nINPUT:\n{json_text(payload)}"
    try:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        pass
    return f"{system}\n\n{user}\n\nJSON:"


def generate_json(stage: str, instruction: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str]:
    tokenizer, model, selected_id, load_note = load_model()
    if tokenizer is None or model is None:
        return None, selected_id, load_note
    try:
        import torch

        prompt = format_chat_prompt(tokenizer, stage, instruction, payload)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6144)
        device = getattr(model, "device", None)
        if device is not None and str(device) != "meta":
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=min(MAX_NEW_TOKENS, STAGE_TOKEN_BUDGETS.get(stage, MAX_NEW_TOKENS)),
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        parsed = parse_json_object(raw)
        if parsed is None:
            return None, selected_id, f"{load_note}; invalid JSON output"
        return parsed, selected_id, load_note
    except Exception as exc:
        return None, selected_id, f"{load_note}; generation failed: {type(exc).__name__}: {exc}"


def run_stage(
    stage: str,
    instruction: str,
    payload: dict[str, Any],
    fallback_factory: Callable[[], dict[str, Any]],
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    fallback = fallback_factory()
    candidate, selected_id, note = generate_json(stage, instruction, payload)
    source = "small_model"
    if candidate is None:
        data = fallback
        source = "deterministic_fallback"
    else:
        data = merge_known(fallback, candidate)
    if validator:
        try:
            data = validator(data)
        except Exception as exc:
            data = fallback
            source = "deterministic_fallback"
            note = f"{note}; validation failed: {type(exc).__name__}: {exc}"
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    result = StageResult(data=data, source=source, model_id=selected_id, elapsed_ms=elapsed_ms, note=note)
    _RUNTIME_TRACE.append(result.runtime_row(stage))
    return result.data


def infer_domain(payload: dict[str, Any]) -> str:
    haystack = " ".join(clean_text(v, 1000).lower() for v in payload.values() if isinstance(v, str))
    domains = [
        ("software engineering", ["api", "code", "software", "app", "backend", "frontend"]),
        ("agent systems", ["agent", "workflow", "tool", "autonomous", "mcp"]),
        ("data and analytics", ["data", "dataset", "analytics", "dashboard", "sql"]),
        ("creative production", ["story", "creative", "brand", "content", "design"]),
    ]
    for domain, signals in domains:
        if any(signal in haystack for signal in signals):
            return domain
    return "general knowledge work"


def analyze_intake(input_payload: dict[str, Any]) -> dict[str, Any]:
    payload = {key: clean_text(value, MAX_INPUT_CHARS) if isinstance(value, str) else value for key, value in input_payload.items()}

    def fallback() -> dict[str, Any]:
        missing = [
            label
            for key, label in [
                ("project_idea", "project idea"),
                ("target_user", "target user"),
                ("build_target", "build target"),
                ("output_contract", "output contract"),
                ("verification_criteria", "verification criteria"),
            ]
            if not clean_text(payload.get(key), 200)
        ]
        complexity_signals = sum(
            bool(clean_text(payload.get(key), 300))
            for key in ["user_context", "project_context", "technical_context", "constraints", "inputs_files", "failure_modes"]
        )
        return {
            "domain": infer_domain(payload),
            "task_type": "design and implementation planning",
            "risk_level": clean_text(payload.get("risk_level"), 40) or "Medium",
            "input_type": "structured brief with free-text context",
            "output_type": clean_text(payload.get("build_target"), 200) or "executable prompt architecture",
            "missing_information": missing,
            "complexity": "high" if complexity_signals >= 5 else "medium" if complexity_signals >= 2 else "low",
            "decision_summary": "Normalize the brief into an explicit compiler input before selecting topology.",
            "assumptions": ["Unspecified details may be resolved conservatively during execution."],
            "risks": clean_list(payload.get("failure_modes"), 5) or ["Ambiguous output contract", "Insufficient verification criteria"],
        }

    instruction = (
        "Classify domain, task type, risk level, input type, output type, missing information, complexity, "
        "decision summary, assumptions, and risks. Do not solve the task."
    )
    return run_stage("intake_analysis", instruction, payload, fallback)


def decide_topology(analysis: dict[str, Any], user_topology_choice: str) -> dict[str, Any]:
    choice = user_topology_choice if user_topology_choice in TOPOLOGIES else "Auto"

    def fallback() -> dict[str, Any]:
        risk = clean_text(analysis.get("risk_level"), 40).lower()
        complexity = clean_text(analysis.get("complexity"), 40).lower()
        domain = clean_text(analysis.get("domain"), 100).lower()
        if choice != "Auto":
            topology = choice
            reason = "Explicit user topology choice."
        elif "agent" in domain or risk == "critical":
            topology = "Agent Workflow"
            reason = "Agentic or critical-risk work benefits from explicit execution and recovery states."
        elif complexity == "high":
            topology = "Cascade"
            reason = "Multiple context areas and dependent outputs require sequential specialist prompts."
        elif analysis.get("missing_information"):
            topology = "Context Pack"
            reason = "A reusable context contract should stabilize unresolved inputs."
        else:
            topology = "Single Prompt"
            reason = "The task is bounded enough for one complete execution contract."
        roles_by_topology = {
            "Single Prompt": ["Lead Executor"],
            "Cascade": ["Brief Analyst", "Solution Architect", "Builder", "Verifier"],
            "Context Pack": ["Context Curator", "Execution Prompt Author"],
            "Agent Workflow": ["Planner", "Executor", "Verifier", "Recovery Sentinel"],
        }
        roles = roles_by_topology[topology]
        return {
            "topology": topology,
            "reason": reason,
            "number_of_prompts": len(roles),
            "roles": roles,
            "handoff_contract": "Each stage receives structured upstream output and returns a verifiable downstream artifact.",
        }

    instruction = (
        "Choose Single Prompt, Cascade, Context Pack, or Agent Workflow. Use Cascade when multiple expertise areas "
        "are required, task A feeds task B, or more than six unrelated ACTION sections are required. Respect an "
        "explicit non-Auto user choice. Return topology, reason, number_of_prompts, roles, and handoff_contract."
    )
    return run_stage("topology_decision", instruction, {"analysis": analysis, "user_choice": choice}, fallback)


def extract_vital_structure(analysis: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    def fallback() -> dict[str, Any]:
        vital_few = [
            "A precise output contract",
            "A topology matched to dependency structure",
            "Verifiable acceptance criteria",
            "Explicit failure and recovery behavior",
        ]
        if analysis.get("missing_information"):
            vital_few.insert(0, "Resolution of critical missing context")
        return {
            "vital_few": vital_few[:5],
            "vital_spot": "The output contract: if it is ambiguous, every downstream prompt can appear complete while producing the wrong artifact.",
            "vital_spot_guard": "Restate the output contract before execution and fail QA when required fields or verification evidence are absent.",
            "decision_summary": f"Optimize the {topology.get('topology', 'selected')} architecture around a small set of quality drivers.",
        }

    instruction = (
        "Extract three to five Vital Few elements that determine most output quality and one Vital Spot whose failure "
        "breaks the workflow. Include a concrete guard for the Vital Spot."
    )
    return run_stage("vital_structure", instruction, {"analysis": analysis, "topology": topology}, fallback)


def select_reasoning_architecture(
    analysis: dict[str, Any],
    topology: dict[str, Any],
    selected_layers: list[str],
) -> dict[str, Any]:
    selected = [layer for layer in selected_layers if layer in REASONING_LAYERS]

    def fallback() -> dict[str, Any]:
        layers = selected or ["CRAFT", "Pareto 80/20", "Private CoT", "Self-Correction", "Sentinel Recovery"]
        if topology.get("topology") in {"Cascade", "Agent Workflow"} and "Agentic Loop" not in layers:
            layers.append("Agentic Loop")
        if clean_text(analysis.get("risk_level"), 30).lower() in {"high", "critical"} and "Kahneman System 2" not in layers:
            layers.append("Kahneman System 2")
        configurations = {
            layer: {
                "purpose": {
                    "CRAFT": "Bind context, role, action, format, and target.",
                    "Kahneman System 2": "Slow down at consequential decisions and verify assumptions.",
                    "Pareto 80/20": "Prioritize the few actions that drive most value.",
                    "Agentic Loop": "Plan, act, observe, verify, and recover.",
                    "Tree of Thought controlled": "Compare strategies without exposing hidden branches.",
                    "Private CoT": "Keep reasoning internal and publish only summaries and evidence.",
                    "Self-Correction": "Repair failed checks before final output.",
                    "Sentinel Recovery": "Detect blocked or degraded states and continue safely.",
                }[layer],
                "public_output": "decision summary, assumptions, risks, verification steps, final answer",
            }
            for layer in layers
        }
        return {
            "selected_layers": layers,
            "configurations": configurations,
            "private_reasoning_policy": "Private reasoning internal only.",
            "tree_of_thought_policy": "Expose only: strategy | upside | risk | cost | selected.",
        }

    instruction = (
        "Select and configure only useful reasoning layers. Private CoT must remain internal. Controlled Tree of "
        "Thought may expose only strategy, upside, risk, cost, selected. Return selected_layers, configurations, "
        "private_reasoning_policy, and tree_of_thought_policy."
    )
    return run_stage(
        "reasoning_architecture",
        instruction,
        {"analysis": analysis, "topology": topology, "selected_layers": selected},
        fallback,
    )


def prompt_block(
    title: str,
    role: str,
    action: str,
    analysis: dict[str, Any],
    topology: dict[str, Any],
    vital: dict[str, Any],
    reasoning_architecture: dict[str, Any],
    output_contract: str,
    verification_criteria: str,
) -> str:
    layers = ", ".join(reasoning_architecture.get("selected_layers", []))
    vital_few = "\n".join(f"- {item}" for item in vital.get("vital_few", []))
    return f"""# {title}

[ROLE]
You are {role}. Own the assigned artifact and its verification. Do not impersonate other stages.

[COGNITIVE_LAYERS]
Use: {layers}. Private reasoning internal only. Public output may include only decision summary, assumptions, risks, verification steps, and final answer.

[KAHNEMAN_SYSTEM2]
Pause before consequential decisions. Check assumptions, dependency order, risk, and evidence before committing.

[PARETO_80_20]
Prioritize these Vital Few:
{vital_few}

[VITAL_SPOT]
{vital.get("vital_spot", "The output contract is the single failure point.")}
Guard: {vital.get("vital_spot_guard", "Fail QA when the contract is incomplete.")}

[REASONING_PROTOCOL]
1. Normalize the available context.
2. Identify assumptions and risks.
3. Compare options only when useful. If using controlled Tree of Thought, expose only: strategy | upside | risk | cost | selected.
4. Execute the selected strategy.
5. Verify against the output contract.
Never reveal chain of thought or hidden branches.

[AGENTIC_LOOP]
PLAN -> ACT -> OBSERVE -> VERIFY -> REPAIR or COMPLETE.
On blocked execution, invoke Sentinel Recovery: state the blocker, preserve valid work, choose the safest viable fallback, and continue.

[ACTION]
{action}

[FORMAT_AND_TARGET]
Target topology: {topology.get("topology", "Single Prompt")}
Required output contract: {output_contract or "Return a complete, directly usable artifact with explicit assumptions and verification evidence."}

[QA_CHECKS]
- Required sections and fields are present.
- Claims and assumptions are distinguishable.
- Verification criteria are satisfied: {verification_criteria or "The output is complete, internally consistent, and directly executable."}
- No full chain of thought or hidden Tree of Thought branches are exposed.
- If a check fails, repair the artifact and rerun QA before returning it."""


def deterministic_prompt_pack(
    analysis: dict[str, Any],
    topology: dict[str, Any],
    vital: dict[str, Any],
    reasoning_architecture: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    topology_name = topology.get("topology", "Single Prompt")
    roles = topology.get("roles", ["Lead Executor"])
    project_idea = clean_text(context.get("project_idea"), 1800) or "Execute the supplied project brief."
    output_contract = clean_text(context.get("output_contract"), 1600)
    verification = clean_text(context.get("verification_criteria"), 1200)
    prompts = []
    for index, role in enumerate(roles, start=1):
        if topology_name == "Single Prompt":
            action = f"Turn this brief into the required artifact:\n{project_idea}"
        elif topology_name == "Context Pack":
            action = (
                "Create a reusable, source-aware context pack that separates facts, assumptions, constraints, open "
                "questions, and execution instructions."
                if index == 1
                else "Use the approved context pack to produce the final execution prompt and verification contract."
            )
        elif topology_name == "Agent Workflow":
            agent_actions = {
                "Planner": "Convert the brief into ordered tasks, dependencies, stop conditions, and acceptance tests.",
                "Executor": "Execute the approved plan and return artifacts plus evidence.",
                "Verifier": "Test artifacts against acceptance criteria and identify repair actions.",
                "Recovery Sentinel": "Handle blockers, failed checks, and degraded model/tool states without losing valid work.",
            }
            action = agent_actions.get(role, f"Execute the {role} stage and return a structured handoff.")
        else:
            action = f"Execute stage {index} as {role}; consume the previous structured handoff and produce the next verifiable artifact."
        prompts.append(
            prompt_block(
                f"Prompt {index}: {role}",
                role,
                action,
                analysis,
                topology,
                vital,
                reasoning_architecture,
                output_contract,
                verification,
            )
        )
    execution_plan = [
        f"Run {role}; validate its output contract; pass only verified artifacts downstream."
        for role in roles
    ]
    return {
        "topology": topology_name,
        "prompts": prompts,
        "execution_plan": execution_plan,
        "output_contract": output_contract or "Complete artifact, assumptions, risks, verification steps, final answer.",
    }


def validate_prompt_pack(data: dict[str, Any]) -> dict[str, Any]:
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt pack is empty")
    cleaned_prompts = [clean_text(prompt, 30000) for prompt in prompts if clean_text(prompt, 30000)]
    if not cleaned_prompts:
        raise ValueError("prompt pack contains no usable prompts")
    for prompt in cleaned_prompts:
        missing = [tag for tag in REQUIRED_PROMPT_TAGS if f"[{tag}]" not in prompt]
        if missing:
            raise ValueError(f"prompt missing required tags: {', '.join(missing)}")
    data["prompts"] = cleaned_prompts
    return data


def generate_prompt_pack(
    analysis: dict[str, Any],
    topology: dict[str, Any],
    vital: dict[str, Any],
    reasoning_architecture: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}

    def fallback() -> dict[str, Any]:
        return deterministic_prompt_pack(analysis, topology, vital, reasoning_architecture, context)

    instruction = (
        "Generate the complete prompt pack for the selected topology. Every prompt must contain all required tags: "
        + ", ".join(REQUIRED_PROMPT_TAGS)
        + ". Never request or reveal full chain of thought. Use exactly 'Private reasoning internal only.' "
        "Controlled Tree of Thought exposes only strategy | upside | risk | cost | selected. Return topology, prompts, "
        "execution_plan, and output_contract."
    )
    return run_stage(
        "prompt_pack_generation",
        instruction,
        {
            "analysis": analysis,
            "topology": topology,
            "vital": vital,
            "reasoning_architecture": reasoning_architecture,
            "context": context,
        },
        fallback,
        validate_prompt_pack,
    )


def repair_prompt_text(prompt: str) -> tuple[str, list[str]]:
    repaired = clean_text(prompt, 30000)
    repairs: list[str] = []
    forbidden = [
        r"reveal (?:your|the) (?:full )?chain of thought",
        r"show (?:your|the) (?:full )?chain of thought",
        r"expose hidden branches",
    ]
    for pattern in forbidden:
        if re.search(pattern, repaired, flags=re.IGNORECASE):
            repaired = re.sub(pattern, "provide a concise decision summary", repaired, flags=re.IGNORECASE)
            repairs.append("Removed chain-of-thought leakage request.")
    for tag in REQUIRED_PROMPT_TAGS:
        if f"[{tag}]" not in repaired:
            repaired += f"\n\n[{tag}]\nComplete this section before execution."
            repairs.append(f"Added missing [{tag}] tag.")
    if "Private reasoning internal only." not in repaired:
        repaired = repaired.replace("[REASONING_PROTOCOL]", "[REASONING_PROTOCOL]\nPrivate reasoning internal only.", 1)
        repairs.append("Added private reasoning policy.")
    if "strategy | upside | risk | cost | selected" not in repaired:
        repaired += "\n\nControlled Tree of Thought public schema: strategy | upside | risk | cost | selected."
        repairs.append("Added controlled Tree of Thought public schema.")
    return repaired, repairs


def deterministic_qa(prompt_pack: dict[str, Any]) -> dict[str, Any]:
    repaired_prompts = []
    issues: list[str] = []
    for index, prompt in enumerate(prompt_pack.get("prompts", []), start=1):
        repaired, repairs = repair_prompt_text(str(prompt))
        repaired_prompts.append(repaired)
        issues.extend(f"Prompt {index}: {repair}" for repair in repairs)
    repaired_pack = dict(prompt_pack)
    repaired_pack["prompts"] = repaired_prompts
    missing_tags = [
        tag
        for tag in REQUIRED_PROMPT_TAGS
        if any(f"[{tag}]" not in prompt for prompt in repaired_prompts)
    ]
    leakage = any(
        re.search(r"(reveal|show|expose).{0,24}chain of thought", line, flags=re.IGNORECASE)
        and not re.search(r"\b(never|do not|don't|must not|without)\b", line, flags=re.IGNORECASE)
        for prompt in repaired_prompts
        for line in prompt.splitlines()
    )
    checks = {
        "all_required_tags": not missing_tags,
        "strong_roles": all("[ROLE]" in prompt and len(prompt.split("[ROLE]", 1)[-1].strip()) > 20 for prompt in repaired_prompts),
        "output_contracts": all("[FORMAT_AND_TARGET]" in prompt for prompt in repaired_prompts),
        "no_chain_of_thought_leakage": not leakage,
        "qa_present": all("[QA_CHECKS]" in prompt for prompt in repaired_prompts),
        "repair_logic_present": all("REPAIR" in prompt for prompt in repaired_prompts),
        "tree_of_thought_controlled": all("strategy | upside | risk | cost | selected" in prompt for prompt in repaired_prompts),
    }
    return {
        "pass": all(checks.values()),
        "issues": issues,
        "checks": checks,
        "repaired_prompt_pack": repaired_pack,
    }


def validate_qa(data: dict[str, Any]) -> dict[str, Any]:
    deterministic = deterministic_qa(data.get("repaired_prompt_pack", {}))
    if not deterministic["pass"]:
        return deterministic
    data["pass"] = True
    data["checks"] = deterministic["checks"]
    data["repaired_prompt_pack"] = deterministic["repaired_prompt_pack"]
    return data


def qa_repair_pass(prompt_pack: dict[str, Any]) -> dict[str, Any]:
    def fallback() -> dict[str, Any]:
        return deterministic_qa(prompt_pack)

    instruction = (
        "Check missing required tags, weak roles, missing output contracts, chain-of-thought leakage, missing QA, "
        "missing repair logic, and uncontrolled Tree of Thought. Repair every issue. Return pass, issues, checks, "
        "and repaired_prompt_pack. Never add hidden reasoning."
    )
    return run_stage("qa_repair", instruction, {"prompt_pack": prompt_pack}, fallback, validate_qa)


def score_metrics(
    analysis: dict[str, Any],
    topology: dict[str, Any],
    qa: dict[str, Any],
) -> dict[str, int]:
    checks = qa.get("checks", {})
    check_score = round(100 * sum(bool(value) for value in checks.values()) / max(1, len(checks)))
    missing_count = len(analysis.get("missing_information", []))
    coverage = max(45, 100 - missing_count * 10)
    topology_score = 94 if topology.get("topology") in {"Cascade", "Agent Workflow"} else 86
    risk_score = 96 if checks.get("no_chain_of_thought_leakage") and checks.get("repair_logic_present") else 68
    return {
        "Prompt Integrity": check_score,
        "Context Coverage": coverage,
        "Agent Readiness": topology_score,
        "Risk Control": risk_score,
    }


def deterministic_final(
    analysis: dict[str, Any],
    topology: dict[str, Any],
    vital: dict[str, Any],
    reasoning_architecture: dict[str, Any],
    qa: dict[str, Any],
) -> dict[str, Any]:
    repaired_pack = qa.get("repaired_prompt_pack", {})
    prompts = repaired_pack.get("prompts", [])
    compiled_prompt_pack = "\n\n---\n\n".join(prompts)
    architecture_analysis = {
        "intake": analysis,
        "topology": topology,
        "vital_structure": vital,
        "reasoning_architecture": reasoning_architecture,
    }
    execution_plan = repaired_pack.get("execution_plan", [])
    repair_protocol = [
        "Detect the failed check and preserve valid upstream artifacts.",
        "Identify the smallest repair that restores the output contract.",
        "Apply the repair, rerun QA, and continue only after verification passes.",
        "If a model stage fails, use that stage's deterministic fallback and record it in Runtime Details.",
    ]
    return {
        "architecture_analysis": architecture_analysis,
        "prompt_pack": compiled_prompt_pack,
        "execution_plan": execution_plan,
        "qa_checklist": qa.get("checks", {}),
        "repair_protocol": repair_protocol,
        "metrics": score_metrics(analysis, topology, qa),
    }


def assemble_final_output(
    analysis: dict[str, Any],
    topology: dict[str, Any],
    vital: dict[str, Any],
    reasoning_architecture: dict[str, Any],
    qa: dict[str, Any],
) -> dict[str, Any]:
    def fallback() -> dict[str, Any]:
        return deterministic_final(analysis, topology, vital, reasoning_architecture, qa)

    instruction = (
        "Assemble the final user-facing compiler result without adding hidden reasoning. Return architecture_analysis, "
        "prompt_pack, execution_plan, qa_checklist, repair_protocol, and metrics. The prompt_pack must preserve all "
        "required prompt tags exactly."
    )

    def validate_final(data: dict[str, Any]) -> dict[str, Any]:
        prompt_pack = clean_text(data.get("prompt_pack"), 120000)
        if not prompt_pack:
            raise ValueError("final prompt pack is empty")
        missing = [tag for tag in REQUIRED_PROMPT_TAGS if f"[{tag}]" not in prompt_pack]
        if missing:
            raise ValueError(f"final assembly lost required tags: {', '.join(missing)}")
        data["prompt_pack"] = prompt_pack
        return data

    return run_stage(
        "final_assembly",
        instruction,
        {
            "analysis": analysis,
            "topology": topology,
            "vital": vital,
            "reasoning_architecture": reasoning_architecture,
            "qa": qa,
        },
        fallback,
        validate_final,
    )


def compile_context(
    project_idea: str,
    target_user: str,
    build_target: str,
    topology_choice: str,
    risk_level: str,
    output_language: str,
    selected_layers: list[str],
    user_context: str,
    project_context: str,
    technical_context: str,
    constraints: str,
    inputs_files: str,
    output_contract: str,
    failure_modes: str,
    verification_criteria: str,
) -> tuple[str, str, str, str, str, str]:
    _RUNTIME_TRACE.clear()
    payload = {
        "project_idea": clean_text(project_idea, MAX_INPUT_CHARS),
        "target_user": clean_text(target_user, 2000),
        "build_target": clean_text(build_target, 2000),
        "risk_level": clean_text(risk_level, 100),
        "output_language": clean_text(output_language, 100),
        "user_context": clean_text(user_context, MAX_INPUT_CHARS),
        "project_context": clean_text(project_context, MAX_INPUT_CHARS),
        "technical_context": clean_text(technical_context, MAX_INPUT_CHARS),
        "constraints": clean_text(constraints, MAX_INPUT_CHARS),
        "inputs_files": clean_text(inputs_files, MAX_INPUT_CHARS),
        "output_contract": clean_text(output_contract, MAX_INPUT_CHARS),
        "failure_modes": clean_text(failure_modes, MAX_INPUT_CHARS),
        "verification_criteria": clean_text(verification_criteria, MAX_INPUT_CHARS),
    }
    analysis = analyze_intake(payload)
    topology = decide_topology(analysis, topology_choice)
    vital = extract_vital_structure(analysis, topology)
    reasoning = select_reasoning_architecture(analysis, topology, selected_layers or [])
    pack = generate_prompt_pack(analysis, topology, vital, reasoning, payload)
    qa = qa_repair_pass(pack)
    final = assemble_final_output(analysis, topology, vital, reasoning, qa)

    metrics_html = render_metrics(final.get("metrics", {}))
    architecture_md = "```json\n" + json_text(final.get("architecture_analysis", {})) + "\n```"
    prompt_pack_text = clean_text(final.get("prompt_pack"), 120000)
    execution_md = render_list(final.get("execution_plan", []))
    qa_md = render_qa(final.get("qa_checklist", {}), final.get("repair_protocol", []))
    runtime_md = render_runtime(_RUNTIME_TRACE)
    return metrics_html, architecture_md, prompt_pack_text, execution_md, qa_md, runtime_md


def render_metrics(metrics: dict[str, Any]) -> str:
    cards = []
    for label in ["Prompt Integrity", "Context Coverage", "Agent Readiness", "Risk Control"]:
        try:
            score = max(0, min(100, int(metrics.get(label, 0))))
        except (TypeError, ValueError):
            score = 0
        cards.append(
            f'<div class="metric-card"><span>{label}</span><strong>{score}</strong>'
            f'<div class="metric-track"><i style="width:{score}%"></i></div></div>'
        )
    return '<div class="metrics-bar">' + "".join(cards) + "</div>"


def render_list(items: Any) -> str:
    values = clean_list(items, 30)
    if not values:
        return "No execution steps were produced."
    return "\n".join(f"{index}. {item}" for index, item in enumerate(values, start=1))


def render_qa(checks: Any, repair_protocol: Any) -> str:
    lines = ["### QA Checklist"]
    if isinstance(checks, dict):
        for label, passed in checks.items():
            lines.append(f"- [{'x' if passed else ' '}] {label.replace('_', ' ').title()}")
    lines.append("\n### Repair Protocol")
    lines.extend(f"{index}. {item}" for index, item in enumerate(clean_list(repair_protocol, 20), start=1))
    return "\n".join(lines)


def render_runtime(trace: list[dict[str, Any]]) -> str:
    lines = [
        "| Stage | Source | Model | Time | Note |",
        "|---|---|---|---:|---|",
    ]
    for row in trace:
        note = clean_text(row.get("note"), 240).replace("|", "/")
        lines.append(
            f"| `{row.get('stage')}` | `{row.get('source')}` | `{row.get('model_id')}` | "
            f"{row.get('elapsed_ms')} ms | {note} |"
        )
    fallback_stages = [row["stage"] for row in trace if row.get("source") == "deterministic_fallback"]
    lines.append(
        "\n**Fallback stages:** "
        + (", ".join(f"`{stage}`" for stage in fallback_stages) if fallback_stages else "None")
    )
    return "\n".join(lines)


def load_example() -> tuple[Any, ...]:
    return (
        "Build a privacy-first issue triage agent that turns raw bug reports into prioritized engineering tickets.",
        "Small product engineering teams",
        "A working agent workflow with prompts, handoffs, and acceptance tests",
        "Auto",
        "High",
        "English",
        ["CRAFT", "Kahneman System 2", "Pareto 80/20", "Agentic Loop", "Private CoT", "Self-Correction", "Sentinel Recovery"],
        "The user can provide incomplete reports and may not know technical terminology.",
        "The product must reduce triage time without hiding uncertainty.",
        "Python, GitHub Issues, structured JSON handoffs, no mandatory cloud API.",
        "Never invent reproduction evidence. Keep private reasoning internal.",
        "Bug report text, logs, screenshots, repository metadata.",
        "Prioritized ticket with severity, confidence, assumptions, reproduction steps, owner suggestion, and verification checklist.",
        "Hallucinated root cause; wrong severity; missing evidence; duplicate issue.",
        "All required ticket fields exist; severity is evidence-backed; uncertain claims are labeled; duplicate check completed.",
    )


def build_demo() -> Any:
    import gradio as gr

    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    css = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as handle:
            css = handle.read()

    with gr.Blocks(title=APP_TITLE, css=css) as demo:
        gr.HTML(
            f"""
<section class="forge-hero">
  <div class="hero-kicker">Multi-call small-model pipeline</div>
  <h1>{APP_TITLE}</h1>
  <p>{APP_SUBTITLE}. Turn messy software, app, and agent ideas into executable prompt architectures.</p>
  <div class="hero-badges"><span>7 isolated calls</span><span>Stage-level fallback</span><span>Private reasoning</span><span>Compiler, not generator</span></div>
</section>
"""
        )
        with gr.Row(elem_classes=["forge-layout"]):
            with gr.Column(scale=1, elem_classes=["config-panel"]):
                gr.HTML('<div class="panel-title">Compiler Input</div>')
                project_idea = gr.Textbox(label="Project idea", lines=4, placeholder="Describe the rough idea to compile...")
                with gr.Row():
                    target_user = gr.Textbox(label="Target user")
                    build_target = gr.Textbox(label="Build target")
                with gr.Row():
                    topology_choice = gr.Dropdown(TOPOLOGIES, value="Auto", label="Topology")
                    risk_level = gr.Dropdown(["Low", "Medium", "High", "Critical"], value="Medium", label="Risk level")
                    output_language = gr.Textbox(value="English", label="Output language")
                selected_layers = gr.CheckboxGroup(REASONING_LAYERS, value=["CRAFT", "Pareto 80/20", "Private CoT", "Self-Correction", "Sentinel Recovery"], label="Reasoning layers")
                with gr.Accordion("Context inputs", open=False):
                    user_context = gr.Textbox(label="User context", lines=3)
                    project_context = gr.Textbox(label="Project context", lines=3)
                    technical_context = gr.Textbox(label="Technical context", lines=3)
                    constraints = gr.Textbox(label="Constraints", lines=3)
                    inputs_files = gr.Textbox(label="Inputs / files", lines=3)
                with gr.Accordion("Contracts and controls", open=True):
                    output_contract = gr.Textbox(label="Output contract", lines=3)
                    failure_modes = gr.Textbox(label="Failure modes", lines=3)
                    verification_criteria = gr.Textbox(label="Verification criteria", lines=3)
                with gr.Row():
                    compile_button = gr.Button("Compile Architecture", variant="primary")
                    example_button = gr.Button("Load Example", variant="secondary")

            with gr.Column(scale=1, elem_classes=["output-panel"]):
                metrics = gr.HTML(value=render_metrics({}))
                gr.HTML('<div class="panel-title">Compiled Output</div>')
                with gr.Accordion("Prompt Pack", open=True):
                    prompt_output = gr.Code(label="Copyable compiled prompt pack", language="markdown", lines=28)
                with gr.Accordion("Architecture Analysis", open=False):
                    architecture_output = gr.Markdown()
                with gr.Accordion("Execution Plan", open=False):
                    execution_output = gr.Markdown()
                with gr.Accordion("QA / Repair Protocol", open=False):
                    qa_output = gr.Markdown()
                with gr.Accordion("Runtime Details", open=False):
                    runtime_output = gr.Markdown()

        inputs = [
            project_idea,
            target_user,
            build_target,
            topology_choice,
            risk_level,
            output_language,
            selected_layers,
            user_context,
            project_context,
            technical_context,
            constraints,
            inputs_files,
            output_contract,
            failure_modes,
            verification_criteria,
        ]
        compile_button.click(
            fn=compile_context,
            inputs=inputs,
            outputs=[metrics, architecture_output, prompt_output, execution_output, qa_output, runtime_output],
        )
        example_button.click(fn=load_example, inputs=[], outputs=inputs)
    return demo


demo = None if parse_bool_env("CONTEXTFORGE_SKIP_UI_BUILD", False) else build_demo()


if __name__ == "__main__":
    (demo or build_demo()).launch()
