from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # YAML optional; JSON-only if PyYAML missing

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logger = logging.getLogger("duetmind")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --------------------------------------------------------------------------------------
# Constraint / Policy Engine (lightweight & flexible)
# --------------------------------------------------------------------------------------

class ConstraintRule:
    def __init__(self, name: str, check: Callable[[Dict[str, Any]], bool],
                 severity: str = "minor", description: str = ""):
        self.name = name
        self.check = check
        self.severity = severity
        self.description = description

    def evaluate(self, outcome: Dict[str, Any]) -> bool:
        try:
            return self.check(outcome)
        except Exception:
            return False


class PolicyEngine:
    def __init__(self, rules: Optional[List[ConstraintRule]] = None):
        self.rules = rules or []

    def register_rule(self, rule: ConstraintRule) -> None:
        self.rules.append(rule)

    def evaluate(self, outcome: Dict[str, Any]) -> List[ConstraintRule]:
        return [r for r in self.rules if not r.evaluate(outcome)]


class CognitiveFault(Exception):
    def __init__(self, message, intent, outcome, leakage, tier="minor"):
        super().__init__(message)
        self.intent = intent
        self.outcome = outcome
        self.leakage = leakage
        self.tier = tier


# Utility rule generators / custom funcs

def keyword_rule(keywords: List[str]) -> Callable[[Dict[str, Any]], bool]:
    kws = [k.lower() for k in keywords]
    def check(outcome: Dict[str, Any]) -> bool:
        content = str(outcome.get("content", "")).lower()
        return not any(kw in content for kw in kws)
    return check


def regex_rule(pattern: str, flags: int = re.IGNORECASE) -> Callable[[Dict[str, Any]], bool]:
    rx = re.compile(pattern, flags)
    def check(outcome: Dict[str, Any]) -> bool:
        return rx.search(str(outcome.get("content", ""))) is None
    return check


def applied_ethics_check(outcome: Dict[str, Any]) -> bool:
    content = str(outcome.get("content", "")).lower()
    moral_keywords = [
        "right", "wrong", "justice", "fair", "unfair", "harm", "benefit",
        "responsibility", "duty", "obligation", "virtue", "vice", "good", "bad", "evil",
    ]
    controversy_keywords = [
        "controversy", "debate", "dispute", "conflict", "argument",
        "polarizing", "divisive", "hotly debated", "scandal",
    ]
    return not (
        any(kw in content for kw in moral_keywords)
        and any(kw in content for kw in controversy_keywords)
    )


CUSTOM_FUNCS: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    "applied_ethics_check": applied_ethics_check,
}


def _load_data_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML not installed, cannot read YAML. Use JSON or install pyyaml.")
        return yaml.safe_load(path.read_text())
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported config format: {path.suffix}")


def load_rules_from_file(path: str | Path) -> List[ConstraintRule]:
    data = _load_data_file(Path(path))
    rules_cfg = data.get("rules", [])
    rules: List[ConstraintRule] = []
    for r in rules_cfg:
        name = r["name"]
        severity = r.get("severity", "minor")
        description = r.get("description", "")
        rtype = r["type"]
        if rtype == "keyword":
            check = keyword_rule(r["keywords"])
        elif rtype == "regex":
            check = regex_rule(r["pattern"], re.IGNORECASE)
        elif rtype == "custom":
            func_name = r["func"]
            if func_name not in CUSTOM_FUNCS:
                raise ValueError(f"Unknown custom func: {func_name}")
            check = CUSTOM_FUNCS[func_name]
        else:
            raise ValueError(f"Unknown rule type: {rtype}")
        rules.append(ConstraintRule(name, check, severity, description))
    return rules


# --------------------------------------------------------------------------------------
# Monitor system (end-to-end configurable)
# --------------------------------------------------------------------------------------

MonitorFn = Callable[["DuetMindAgent", str, Dict[str, Any]], None]


@dataclass
class MonitorSpec:
    name: str
    type: str
    severity: str = "minor"
    description: str = ""
    params: Optional[Dict[str, Any]] = None


class MonitorFactory:
    @staticmethod
    def build(spec: MonitorSpec) -> MonitorFn:
        t = spec.type.lower()
        if t == "rcd_policy":
            return MonitorFactory._rcd_policy_monitor(spec)
        if t == "keyword":
            return MonitorFactory._keyword_monitor(spec)
        if t == "regex":
            return MonitorFactory._regex_monitor(spec)
        if t == "resource":
            return MonitorFactory._resource_monitor(spec)
        if t == "custom":
            return MonitorFactory._custom_monitor(spec)
        raise ValueError(f"Unknown monitor type: {spec.type}")

    @staticmethod
    def _handle(spec: MonitorSpec, agent: "DuetMindAgent", task: str, passed: bool, detail: Dict[str, Any]):
        if passed:
            return
        msg = f"Monitor[{spec.name}] violation: {spec.description or spec.type}"
        if spec.severity.lower() == "severe":
            raise CognitiveFault(
                msg,
                intent={"task": task, **(detail.get("intent", {}))},
                outcome=detail.get("outcome", {}),
                leakage={"type": spec.type, "name": spec.name, **detail.get("leakage", {})},
                tier="severe",
            )
        else:
            logger.warning("[MINOR] %s", msg)
            # optionally push to agent interaction history
            agent.interaction_history.append({
                "type": "monitor_minor",
                "monitor": spec.name,
                "task": task,
                "detail": detail,
            })

    @staticmethod
    def _rcd_policy_monitor(spec: MonitorSpec) -> MonitorFn:
        params = spec.params or {}
        rules_file = params.get("rules_file")
        if not rules_file:
            raise ValueError("rcd_policy monitor requires params.rules_file")
        rules = load_rules_from_file(rules_file)
        pe = PolicyEngine(rules)
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
            outcome = result if isinstance(result, dict) else {"content": str(result)}
            violations = pe.evaluate(outcome)
            if violations:
                # escalate if any severe present, else minor lumped
                has_severe = any(v.severity.lower() == "severe" for v in violations)
                detail = {
                    "outcome": outcome,
                    "violations": [v.name for v in violations],
                }
                MonitorFactory._handle(
                    spec if not has_severe else MonitorSpec(**{**spec.__dict__, "severity": "severe"}),
                    agent,
                    task,
                    passed=False,
                    detail=detail,
                )
        return monitor

    @staticmethod
    def _keyword_monitor(spec: MonitorSpec) -> MonitorFn:
        params = spec.params or {}
        keywords = [k.lower() for k in params.get("keywords", [])]
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
            content = str(result.get("content", "")).lower() if isinstance(result, dict) else str(result).lower()
            violated = any(kw in content for kw in keywords)
            MonitorFactory._handle(spec, agent, task, not violated, {"outcome": {"content": content}})
        return monitor

    @staticmethod
    def _regex_monitor(spec: MonitorSpec) -> MonitorFn:
        params = spec.params or {}
        pattern = params.get("pattern")
        flags = re.IGNORECASE | re.MULTILINE
        rx = re.compile(pattern, flags) if pattern else None
        if rx is None:
            raise ValueError("regex monitor requires params.pattern")
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
            content = str(result.get("content", "")) if isinstance(result, dict) else str(result)
            violated = rx.search(content) is not None
            MonitorFactory._handle(spec, agent, task, not violated, {"outcome": {"content": content}})
        return monitor

    @staticmethod
    def _resource_monitor(spec: MonitorSpec) -> MonitorFn:
        params = spec.params or {}
        budget_key = params.get("budget_key", "resource_budget")
        tolerance = float(params.get("tolerance", 0.2))  # e.g., 0.2 = 20%
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
            # Result may optionally contain runtime info set by engine
            runtime = float(result.get("runtime", 0.0)) if isinstance(result, dict) else 0.0
            # Agent can pass the budget for the task via interaction_history tail or style; keep simple: style may carry default
            budget = float(agent.style.get(budget_key, 0.0))
            if budget <= 0:
                # if no budget provided, do nothing
                return
            leakage = (runtime - budget) / budget if budget else 0.0
            violated = leakage > tolerance
            MonitorFactory._handle(spec, agent, task, not violated, {
                "outcome": {"runtime": runtime},
                "leakage": {"budget": budget, "leakage": leakage},
                "intent": {budget_key: budget},
            })
        return monitor

    @staticmethod
    def _custom_monitor(spec: MonitorSpec) -> MonitorFn:
        params = spec.params or {}
        func_name = params.get("func")
        func = CUSTOM_FUNCS.get(func_name)
        if func is None:
            raise ValueError(f"Unknown custom func: {func_name}")
        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
            outcome = result if isinstance(result, dict) else {"content": str(result)}
            passed = func(outcome)
            MonitorFactory._handle(spec, agent, task, passed, {"outcome": outcome})
        return monitor


class MonitorManager:
    def __init__(self, specs: Optional[List[MonitorSpec]] = None):
        self.specs = specs or []
        self.monitors: List[MonitorFn] = [MonitorFactory.build(s) for s in self.specs]

    @staticmethod
    def load_from_file(path: str | Path) -> "MonitorManager":
        data = _load_data_file(Path(path))
        specs = [MonitorSpec(**m) for m in data.get("monitors", [])]
        return MonitorManager(specs)

    def get_callables(self) -> List[MonitorFn]:
        return self.monitors


# --------------------------------------------------------------------------------------
# DuetMindAgent with persistence and config-driven personalities
# --------------------------------------------------------------------------------------

class DuetMindAgent:
    """A cognitive agent with style, reasoning, persistence, and pluggable monitors."""

    def __init__(self, name: str, style: Dict[str, float], engine=None, monitors: Optional[List[MonitorFn]] = None):
        self.name = name
        self.style = style
        self.engine = engine
        self.monitors = monitors or []
        self.knowledge_graph: Dict[str, Any] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        logger.info(f"Agent '{name}' initialized with style {style} and {len(self.monitors)} monitors.")

    # Reasoning + monitors
    def generate_reasoning_tree(self, task: str) -> Dict[str, Any]:
        start = time.perf_counter()
        result = self.engine.safe_think(self.name, task) if self.engine else {"content": "No engine", "confidence": 0.5}
        runtime = time.perf_counter() - start
        # normalize and attach runtime
        if not isinstance(result, dict):
            result = {"content": str(result), "confidence": 0.5}
        result.setdefault("runtime", runtime)

        # Run pluggable monitors
        for monitor in self.monitors:
            monitor(self, task, result)

        if "error" in result:
            logger.warning(f"{self.name}'s reasoning process was halted.")
            node_key = f"{self.name}_fault_{len(self.knowledge_graph)}"
            self.knowledge_graph[node_key] = {"task": task, "result": result, "style": self.style}
            return {"root": node_key, "result": result}

        node_key = f"{self.name}_reasoning_{len(self.knowledge_graph)}"
        styled_result = self._apply_style_influence(result, task)
        self.knowledge_graph[node_key] = {
            "task": task,
            "result": styled_result,
            "style": self.style,
            "timestamp": len(self.knowledge_graph),
        }
        self.interaction_history.append({"task": task, "node": node_key, "result": styled_result})
        return {"root": node_key, "result": styled_result, "agent": self.name, "style_applied": True}

    def _apply_style_influence(self, base_result: Dict[str, Any], task: str) -> Dict[str, Any]:
        styled_result = dict(base_result)
        logic_weight = float(self.style.get("logic", 0.5))
        creativity_weight = float(self.style.get("creativity", 0.5))
        analytical_weight = float(self.style.get("analytical", 0.5))
        if "confidence" in styled_result:
            styled_result["confidence"] *= (0.8 + analytical_weight * 0.4)
        insights: List[str] = []
        if logic_weight > 0.7:
            insights.append("Applying rigorous logical validation")
        if creativity_weight > 0.7:
            insights.append("Exploring creative alternative perspectives")
        if analytical_weight > 0.7:
            insights.append("Conducting systematic decomposition")
        styled_result["style_insights"] = insights
        styled_result["style_signature"] = self.name
        return styled_result

    # Dialogue
    def dialogue_with(self, other_agent: 'DuetMindAgent', topic: str, rounds: int = 3) -> Dict[str, Any]:
        dialogue_history: List[Dict[str, Any]] = []
        current_topic = topic
        logger.info(f"Dialogue between {self.name} and {other_agent.name} on {topic}")
        for round_num in range(rounds):
            my_response = self.generate_reasoning_tree(f"Round {round_num + 1}: {current_topic}")
            dialogue_history.append({"round": round_num + 1, "agent": self.name, "response": my_response})
            other_response = other_agent.generate_reasoning_tree(
                f"Responding to {self.name}'s perspective on: {current_topic}"
            )
            dialogue_history.append({"round": round_num + 1, "agent": other_agent.name, "response": other_response})
            current_topic = self._evolve_topic(current_topic, my_response, other_response)
        synthesis = self._synthesize_dialogue(dialogue_history, topic)
        return {
            "original_topic": topic,
            "final_topic": current_topic,
            "dialogue_history": dialogue_history,
            "synthesis": synthesis,
            "participants": [self.name, other_agent.name],
        }

    def _evolve_topic(self, current_topic: str, response1: Dict[str, Any], response2: Dict[str, Any]) -> str:
        conf1 = float(response1.get("result", {}).get("confidence", 0.5))
        conf2 = float(response2.get("result", {}).get("confidence", 0.5))
        if conf1 > 0.8 and conf2 > 0.8:
            return f"Deep dive into: {current_topic}"
        if conf1 < 0.4 or conf2 < 0.4:
            return f"Alternative approach to: {current_topic}"
        return f"Balanced exploration of: {current_topic}"

    def _synthesize_dialogue(self, dialogue_history: List[Dict[str, Any]], original_topic: str) -> Dict[str, Any]:
        contributions: Dict[str, Any] = {}
        total_conf = 0.0
        insights_count = 0
        for entry in dialogue_history:
            agent = entry["agent"]
            contributions.setdefault(agent, {"rounds": 0, "insights": []})
            contributions[agent]["rounds"] += 1
            result = entry["response"].get("result", {})
            total_conf += float(result.get("confidence", 0.5))
            if "style_insights" in result:
                contributions[agent]["insights"].extend(result["style_insights"])  # <-- FIXED: Removed () after extend
                insights_count += len(result["style_insights"])
        avg_conf = total_conf / len(dialogue_history) if dialogue_history else 0.0
        return {
            "original_topic": original_topic,
            "dialogue_quality": avg_conf,
            "total_insights": insights_count,
            "agent_contributions": contributions,
            "cognitive_diversity": len({c["agent"] for c in dialogue_history}),
        }

    # Persistence
    def save_state(self, path: str) -> None:
        state = {
            "name": self.name,
            "style": self.style,
            "knowledge_graph": self.knowledge_graph,
            "interaction_history": self.interaction_history,
        }
        Path(path).write_text(json.dumps(state, indent=2))
        logger.info(f"Agent state saved to {path}")

    @classmethod
    def load_state(cls, path: str, engine=None, monitors: Optional[List[MonitorFn]] = None) -> 'DuetMindAgent':
        state = json.loads(Path(path).read_text())
        agent = cls(state["name"], state.get("style", {}), engine=engine, monitors=monitors)
        agent.knowledge_graph = state.get("knowledge_graph", {})
        agent.interaction_history = state.get("interaction_history", [])
        logger.info(f"Agent state loaded from {path}")
        return agent

    # Config-driven creation (agent + monitors)
    @staticmethod
    def from_config(config_path: str | Path, engine=None) -> 'DuetMindAgent':
        cfg = _load_data_file(Path(config_path))
        name = cfg["name"]
        style = cfg.get("style", {})
        monitors_path = cfg.get("monitors", {}).get("file") if isinstance(cfg.get("monitors"), dict) else cfg.get("monitors")
        monitors: List[MonitorFn] = []
        if monitors_path:
            mm = MonitorManager.load_from_file(Path(config_path).parent / monitors_path)
            monitors = mm.get_callables()
        return DuetMindAgent(name=name, style=style, engine=engine, monitors=monitors)


# --------------------------------------------------------------------------------------
# Minimal example Engine
# --------------------------------------------------------------------------------------

class ExampleEngine:
    """Toy engine that returns a dict result with content and confidence."""
    def safe_think(self, agent_name: str, task: str) -> Dict[str, Any]:
        # Simulate some work
        time.sleep(0.02)
        # Produce a deterministic pseudo-result for the demo
        conf = 0.6 if "Round" in task else 0.9
        content = f"[{agent_name}] Thoughts about: {task}"
        return {"content": content, "confidence": conf}


# --------------------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------------------

def _write_demo_files(tmpdir: Path) -> Dict[str, Path]:
    agent_yaml = tmpdir / "agent.yaml"
    monitors_yaml = tmpdir / "monitors.yaml"
    rules_yaml = tmpdir / "rules.yaml"

    if yaml is None:
        # JSON fallback if PyYAML not installed
        agent_json = tmpdir / "agent.json"
        agent_json.write_text(json.dumps({
            "name": "Athena",
            "style": {"logic": 0.9, "creativity": 0.4, "analytical": 0.8},
            "monitors": {"file": "monitors.json"},
        }, indent=2))
        (tmpdir / "monitors.json").write_text(json.dumps({
            "monitors": [
                {"name": "policy_rcd", "type": "rcd_policy", "severity": "severe",
                 "description": "Policy rules check on final outcome",
                 "params": {"rules_file": "rules.json"}},
                {"name": "keyword_guard", "type": "keyword", "severity": "severe",
                 "description": "Block certain tokens",
                 "params": {"keywords": ["manipulation", "hate", "racist"]}},
            ]
        }, indent=2))
        (tmpdir / "rules.json").write_text(json.dumps({
            "rules": [
                {"name": "no_manipulation", "severity": "severe", "description": "Blocks manipulation",
                 "type": "keyword", "keywords": ["manipulate", "manipulation", "gaslight", "coerce", "deceive", "exploit"]}
            ]
        }, indent=2))
        return {"agent": agent_json}

    # YAML versions
    agent_yaml.write_text(
        """
name: Athena
style:
  logic: 0.9
  creativity: 0.4
  analytical: 0.8
monitors:
  file: monitors.yaml
""".strip()
    )
    monitors_yaml.write_text(
        """
monitors:
  - name: policy_rcd
    type: rcd_policy
    severity: severe
    description: Policy rules check on final outcome
    params:
      rules_file: rules.yaml
  - name: keyword_guard
    type: keyword
    severity: severe
    description: Block certain tokens
    params:
      keywords: ["manipulation", "hate", "racist"]
  - name: resource_guard
    type: resource
    severity: minor
    description: Flag overruns beyond 20%
    params:
      budget_key: resource_budget
      tolerance: 0.2
""".strip()
    )
    rules_yaml.write_text(
        """
rules:
  - name: no_self_harm
    severity: severe
    description: Reject outcomes promoting self-harm
    type: keyword
    keywords: ["self-harm", "suicide"]
  - name: no_manipulation
    severity: severe
    description: Blocks manipulation
    type: keyword
    keywords: ["manipulate", "manipulation", "gaslight", "coerce", "deceive", "exploit"]
  - name: applied_ethics
    severity: minor
    description: Flags controversial moral issues for review
    type: custom
    func: applied_ethics_check
""".strip()
    )
    return {"agent": agent_yaml}


def _demo() -> None:
    tmpdir = Path("./_duetmind_demo")
    tmpdir.mkdir(exist_ok=True)
    files = _write_demo_files(tmpdir)
    engine = ExampleEngine()
    agent = DuetMindAgent.from_config(files["agent"], engine=engine)

    # Provide a default per-task resource budget in style for the resource monitor demo
    agent.style.setdefault("resource_budget", 0.01)

    # Run a safe task
    print("\n--- SAFE RUN ---")
    out = agent.generate_reasoning_tree("Intro: collaborative planning")
    print(json.dumps(out, indent=2))

    # Run a dialogue with a second agent
    print("\n--- DIALOGUE ---")
    agent_b = DuetMindAgent("Apollo", {"logic": 0.7, "creativity": 0.8, "analytical": 0.5}, engine=engine, monitors=agent.monitors)
    convo = agent.dialogue_with(agent_b, topic="Designing a fair tournament", rounds=2)
    print(f"participants: {convo['participants']}")
    print(f"final_topic: {convo['final_topic']}")

    # Persistence
    save_path = tmpdir / "athena.state.json"
    agent.save_state(str(save_path))
    _ = DuetMindAgent.load_state(str(save_path), engine=engine, monitors=agent.monitors)


if __name__ == "__main__":
    _demo()
