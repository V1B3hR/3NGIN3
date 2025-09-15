from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
# Constraint / Policy Engine
# --------------------------------------------------------------------------------------

class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    SEVERE = "severe"


def _parse_severity(value: str | Severity | None) -> Severity:
    if isinstance(value, Severity):
        return value
    if not value:
        return Severity.MINOR
    v = str(value).lower()
    if v == "severe":
        return Severity.SEVERE
    if v == "major":
        return Severity.MAJOR
    return Severity.MINOR


# Rule check can return:
#   True -> pass
#   False -> fail
#   (bool, str) -> (pass, rationale-if-fail)
RuleCheckResult = Union[bool, Tuple[bool, Optional[str]]]
RuleCheckFn = Callable[[Dict[str, Any]], RuleCheckResult]


@dataclass
class ConstraintRule:
    """
    Represents a single policy/constraint rule.

    check(outcome) should return:
      - bool
      - or (bool, str) where the str is an optional rationale on failure.
    """
    name: str
    check: RuleCheckFn
    severity: Severity = Severity.MINOR
    description: str = ""

    def run(self, outcome: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Exception]]:
        """Execute the rule and return (passed, rationale, exception)."""
        try:
            result = self.check(outcome)
            if isinstance(result, tuple):
                passed, rationale = result
                return bool(passed), rationale, None
            return bool(result), None, None
        except Exception as exc:
            return False, f"Rule execution error: {exc}", exc

    # Backwards compatibility
    def evaluate(self, outcome: Dict[str, Any]) -> bool:
        passed, _, _ = self.run(outcome)
        return passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "description": self.description,
        }


class PolicyEngine:
    """
    Evaluates a set of rules against an outcome.

    evaluate(outcome, rich=True)  -> List[Dict[str, Any]] describing violations
    evaluate(outcome, rich=False) -> List[ConstraintRule] (failed rule objects)
    """

    def __init__(self, rules: Optional[List[ConstraintRule]] = None):
        self.rules = rules or []

    def register_rule(self, rule: ConstraintRule) -> None:
        self.rules.append(rule)

    def evaluate(
        self,
        outcome: Dict[str, Any],
        rich: bool = False
    ) -> Union[List[ConstraintRule], List[Dict[str, Any]]]:
        violations_rules: List[ConstraintRule] = []
        rich_violations: List[Dict[str, Any]] = []
        for rule in self.rules:
            passed, rationale, error = rule.run(outcome)
            if not passed:
                violations_rules.append(rule)
                if rich:
                    rich_violations.append({
                        "name": rule.name,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "rationale": rationale,
                        "error": str(error) if error else None,
                    })
        return rich_violations if rich else violations_rules


# --------------------------------------------------------------------------------------
# CognitiveFault (Enhanced)
# --------------------------------------------------------------------------------------

class CognitiveFault(Exception):
    """
    Raised when a cognitive / policy / monitoring layer determines execution must halt.

    Fields:
        message: Human-readable summary.
        intent: Context of initiating task.
        outcome: Agent output triggering the fault.
        leakage: Resource/policy metadata.
        severity: Enum-backed severity (replaces legacy tier).
        violations: Structured rule violations (list of dicts).
        fault_id: Unique identifier.
        timestamp: Epoch time (float).
    """

    def __init__(
        self,
        message: str,
        intent: Dict[str, Any],
        outcome: Dict[str, Any],
        leakage: Dict[str, Any],
        severity: Severity = Severity.MINOR,
        violations: Optional[List[Dict[str, Any]]] = None,
        cause: Optional[BaseException] = None,
        fault_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ):
        super().__init__(message)
        self.intent = intent
        self.outcome = outcome
        self.leakage = leakage
        self.severity = severity
        self.violations = violations or []
        self.fault_id = fault_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        self.__cause__ = cause

    @property
    def tier(self) -> str:
        # Backward compatibility (old attribute name)
        return self.severity.value

    def to_dict(self, redact: bool = False) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "timestamp": self.timestamp,
            "message": str(self),
            "severity": self.severity.value,
            "intent": self.intent,
            "outcome": self._maybe_redact_dict(self.outcome, redact),
            "leakage": self.leakage,
            "violations": self.violations,
        }

    def _maybe_redact_dict(self, data: Dict[str, Any], redact: bool) -> Dict[str, Any]:
        if not redact:
            return data
        redacted: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 400:
                redacted[k] = v[:200] + " ... [REDACTED] ..."
            else:
                redacted[k] = v
        return redacted

    def __str__(self) -> str:
        base = super().__str__()
        if self.violations:
            vnames = ", ".join(v.get("name", "?") for v in self.violations)
            return f"{base} | severity={self.severity.value} | violations: {vnames} | id={self.fault_id}"
        return f"{base} | severity={self.severity.value} | id={self.fault_id}"

    @staticmethod
    def _derive_highest_severity(violations: List[Dict[str, Any]]) -> Severity:
        if not violations:
            return Severity.MINOR
        order = {Severity.MINOR: 1, Severity.MAJOR: 2, Severity.SEVERE: 3}
        best = Severity.MINOR
        for v in violations:
            sv = _parse_severity(v.get("severity"))
            if order[sv] > order[best]:
                best = sv
        return best

    @classmethod
    def from_policy_violations(
        cls,
        message: str,
        task: str,
        outcome: Dict[str, Any],
        monitor_spec: Dict[str, Any],
        violations: List[Dict[str, Any]],
    ) -> "CognitiveFault":
        severity = cls._derive_highest_severity(violations)
        leakage = {"type": monitor_spec.get("type"), "name": monitor_spec.get("name")}
        intent = {"task": task}
        return cls(
            message=message,
            intent=intent,
            outcome=outcome,
            leakage=leakage,
            severity=severity,
            violations=violations,
        )


# --------------------------------------------------------------------------------------
# Utility rule generators / custom funcs (enhanced)
# --------------------------------------------------------------------------------------

def keyword_rule(
    keywords: List[str],
    match_mode: str = "any",
    whole_word: bool = False,
    case_insensitive: bool = True,
) -> RuleCheckFn:
    """
    Build a keyword exclusion rule.

    Parameters:
        keywords: List of blocked keyword tokens.
        match_mode: 'any' (fail if any keyword appears) or 'all' (fail only if all appear).
        whole_word: If True, matches must be whole-token (word boundary).
        case_insensitive: If True, lowercase normalization (or regex IGNORECASE) is applied.
    """
    if not keywords:
        return lambda outcome: True

    if case_insensitive:
        kws_for_match = [k.lower() for k in keywords]
    else:
        kws_for_match = keywords

    if whole_word:
        flags = re.IGNORECASE if case_insensitive else 0
        compiled = [(kw, re.compile(rf"\b{re.escape(kw)}\b", flags)) for kw in kws_for_match]

        def check(outcome: Dict[str, Any]) -> RuleCheckResult:
            content = str(outcome.get("content", ""))
            haystack = content if not case_insensitive else content.lower()
            matched: List[str] = []
            for orig_kw, rx in compiled:
                if rx.search(haystack):
                    matched.append(orig_kw)
            if match_mode == "all":
                failed = all(kw in matched for kw in kws_for_match)
            else:
                failed = len(matched) > 0
            if failed:
                return False, f"Blocked keyword(s) detected: {', '.join(sorted(set(matched)))}"
            return True
        return check

    def check(outcome: Dict[str, Any]) -> RuleCheckResult:
        content = str(outcome.get("content", ""))
        haystack = content if not case_insensitive else content.lower()
        present = [kw for kw in kws_for_match if kw in haystack]
        if match_mode == "all":
            failed = len(present) == len(kws_for_match)
        else:
            failed = len(present) > 0
        if failed:
            return False, f"Blocked keyword substring(s): {', '.join(sorted(set(present)))}"
        return True

    return check


def _parse_pattern_flags(flag_string: Optional[str]) -> int:
    """
    Convert a pipe-delimited string of flag names (e.g. 'IGNORECASE|MULTILINE')
    into a combined re flag integer. Unknown names are ignored.
    """
    if not flag_string:
        return 0
    mapping = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
        "ASCII": re.ASCII,
        "VERBOSE": re.VERBOSE,
    }
    total = 0
    for part in (p.strip().upper() for p in flag_string.split("|")):
        total |= mapping.get(part, 0)
    return total


def regex_rule(pattern: str, flags: int = re.IGNORECASE) -> RuleCheckFn:
    """
    Build a regex exclusion rule.
    Fails if the pattern matches the content. Returns rationale with first (and count if multiple) match.
    """
    rx = re.compile(pattern, flags)

    def check(outcome: Dict[str, Any]) -> RuleCheckResult:
        content = str(outcome.get("content", ""))
        matches = list(rx.finditer(content))
        if matches:
            first = matches[0].group(0)
            if len(matches) == 1:
                return False, f"Regex pattern matched: '{first}'"
            return False, f"Regex pattern matched {len(matches)} times; first: '{first}'"
        return True

    return check


def applied_ethics_check(outcome: Dict[str, Any]) -> RuleCheckResult:
    """
    Flags content that simultaneously contains a moral/ethical token and a controversy-related token.
    Returns rationale listing the overlapping categories if triggered.
    """
    content = str(outcome.get("content", "")).lower()
    moral_keywords = [
        "right", "wrong", "justice", "fair", "unfair", "harm", "benefit",
        "responsibility", "duty", "obligation", "virtue", "vice", "good", "bad", "evil",
    ]
    controversy_keywords = [
        "controversy", "debate", "dispute", "conflict", "argument",
        "polarizing", "divisive", "hotly debated", "scandal",
    ]
    moral_hits = {kw for kw in moral_keywords if kw in content}
    controversy_hits = {kw for kw in controversy_keywords if kw in content}
    triggered = bool(moral_hits and controversy_hits)
    if triggered:
        return False, (
            "Applied ethics review triggered: moral terms ("
            + ", ".join(sorted(moral_hits))
            + ") with controversy terms ("
            + ", ".join(sorted(controversy_hits))
            + ")"
        )
    return True


CUSTOM_FUNCS: Dict[str, RuleCheckFn] = {
    "applied_ethics_check": applied_ethics_check,
}


def _load_data_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML not installed, cannot read YAML. Use JSON or install pyyaml.")
        return yaml.safe_load(path.read_text())
    if suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported config format: {path.suffix}")


def load_rules_from_file(path: str | Path) -> List[ConstraintRule]:
    """
    Load rules from JSON or YAML.

    Extended optional params per rule:
      keyword rules:
        match_mode: 'any' (default) or 'all'
        whole_word: bool
        case_insensitive: bool
      regex rules:
        pattern_flags: pipe-delimited (e.g. "IGNORECASE|MULTILINE")

    Example (YAML):
      rules:
        - name: block_sensitive
          type: keyword
          severity: severe
          keywords: ["secret", "password"]
          match_mode: any
          whole_word: false
          case_insensitive: true
        - name: ban_token_pattern
          type: regex
          pattern: "\\bAKIA[0-9A-Z]{16}\\b"
          pattern_flags: IGNORECASE
    """
    data = _load_data_file(Path(path))
    rules_cfg = data.get("rules", [])
    if not isinstance(rules_cfg, list):
        raise ValueError("Config 'rules' must be a list")

    rules: List[ConstraintRule] = []
    for r in rules_cfg:
        if not isinstance(r, dict):
            raise ValueError(f"Rule entry must be an object: {r}")
        try:
            name = r["name"]
            severity = _parse_severity(r.get("severity", "minor"))
            description = r.get("description", "")
            rtype = r["type"]
        except KeyError as ke:
            raise ValueError(f"Missing required rule field: {ke} in {r}") from ke

        if rtype == "keyword":
            keywords = r.get("keywords")
            if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
                raise ValueError(f"Rule '{name}': 'keywords' must be a list[str]")
            check = keyword_rule(
                keywords=keywords,
                match_mode=r.get("match_mode", "any"),
                whole_word=bool(r.get("whole_word", False)),
                case_insensitive=bool(r.get("case_insensitive", True)),
            )
        elif rtype == "regex":
            pattern = r.get("pattern")
            if not pattern or not isinstance(pattern, str):
                raise ValueError(f"Rule '{name}': 'pattern' must be a non-empty string")
            flags = re.IGNORECASE  # default to maintain backward behavior
            extra_flags = _parse_pattern_flags(r.get("pattern_flags"))
            flags |= extra_flags
            check = regex_rule(pattern, flags)
        elif rtype == "custom":
            func_name = r.get("func")
            if not func_name or func_name not in CUSTOM_FUNCS:
                raise ValueError(f"Rule '{name}': unknown custom func '{func_name}'")
            check = CUSTOM_FUNCS[func_name]
        else:
            raise ValueError(f"Rule '{name}': Unknown rule type: {rtype}")

        rules.append(ConstraintRule(name, check, severity, description))
    return rules


# --------------------------------------------------------------------------------------
# Monitor system
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
        violations = detail.get("violations") or []
        derived = CognitiveFault._derive_highest_severity(violations) if violations else _parse_severity(spec.severity)
        is_severe = derived == Severity.SEVERE

        if is_severe:
            raise CognitiveFault(
                message=msg,
                intent={"task": task, **(detail.get("intent", {}))},
                outcome=detail.get("outcome", {}),
                leakage={"type": spec.type, "name": spec.name, **detail.get("leakage", {})},
                severity=derived,
                violations=violations
            )
        else:
            logger.warning("[MINOR] %s", msg)
            agent.interaction_history.append({
                "type": "monitor_minor",
                "monitor": spec.name,
                "task": task,
                "detail": detail,
                "severity": derived.value,
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
            violations = pe.evaluate(outcome, rich=True)
            if violations:
                detail = {
                    "outcome": outcome,
                    "violations": violations,
                }
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
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
        tolerance = float(params.get("tolerance", 0.2))

        def monitor(agent: "DuetMindAgent", task: str, result: Dict[str, Any]) -> None:
            runtime = float(result.get("runtime", 0.0)) if isinstance(result, dict) else 0.0
            budget = float(agent.style.get(budget_key, 0.0))
            if budget <= 0:
                return
            leakage_val = (runtime - budget) / budget if budget else 0.0
            violated = leakage_val > tolerance
            MonitorFactory._handle(spec, agent, task, not violated, {
                "outcome": {"runtime": runtime},
                "leakage": {"budget": budget, "leakage": leakage_val},
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
# DuetMindAgent
# --------------------------------------------------------------------------------------

class DuetMindAgent:
    """A cognitive agent with style, reasoning, persistence, and pluggable monitors."""

    def __init__(
        self,
        name: str,
        style: Dict[str, float],
        engine=None,
        monitors: Optional[List[MonitorFn]] = None
    ):
        self.name = name
        self.style = style
        self.engine = engine
        self.monitors = monitors or []
        self.knowledge_graph: Dict[str, Any] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        logger.info(f"Agent '{name}' initialized with style {style} and {len(self.monitors)} monitors.")

    def generate_reasoning_tree(self, task: str) -> Dict[str, Any]:
        start = time.perf_counter()
        result = self.engine.safe_think(self.name, task) if self.engine else {
            "content": "No engine",
            "confidence": 0.5
        }
        runtime = time.perf_counter() - start
        if not isinstance(result, dict):
            result = {"content": str(result), "confidence": 0.5}
        result.setdefault("runtime", runtime)

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
                contributions[agent]["insights"].extend(result["style_insights"])
                insights_count += len(result["style_insights"])
        avg_conf = total_conf / len(dialogue_history) if dialogue_history else 0.0
        return {
            "original_topic": original_topic,
            "dialogue_quality": avg_conf,
            "total_insights": insights_count,
            "agent_contributions": contributions,
            "cognitive_diversity": len({c["agent"] for c in dialogue_history}),
        }

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
    def load_state(
        cls,
        path: str,
        engine=None,
        monitors: Optional[List[MonitorFn]] = None
    ) -> 'DuetMindAgent':
        state = json.loads(Path(path).read_text())
        agent = cls(state["name"], state.get("style", {}), engine=engine, monitors=monitors)
        agent.knowledge_graph = state.get("knowledge_graph", {})
        agent.interaction_history = state.get("interaction_history", [])
        logger.info(f"Agent state loaded from {path}")
        return agent

    @staticmethod
    def from_config(config_path: str | Path, engine=None) -> 'DuetMindAgent':
        cfg = _load_data_file(Path(config_path))
        name = cfg["name"]
        style = cfg.get("style", {})
        monitors_path = (
            cfg.get("monitors", {}).get("file")
            if isinstance(cfg.get("monitors"), dict)
            else cfg.get("monitors")
        )
        monitors: List[MonitorFn] = []
        if monitors_path:
            mm = MonitorManager.load_from_file(Path(config_path).parent / monitors_path)
            monitors = mm.get_callables()
        return DuetMindAgent(name=name, style=style, engine=engine, monitors=monitors)


# --------------------------------------------------------------------------------------
# Minimal Example Engine
# --------------------------------------------------------------------------------------

class ExampleEngine:
    """Toy engine that returns a dict result with content and confidence."""
    def safe_think(self, agent_name: str, task: str) -> Dict[str, Any]:
        time.sleep(0.02)
        conf = 0.6 if "Round" in task else 0.9
        content = f"[{agent_name}] Thoughts about: {task}"
        return {"content": content, "confidence": conf}


# --------------------------------------------------------------------------------------
# Demo Utilities
# --------------------------------------------------------------------------------------

def _write_demo_files(tmpdir: Path) -> Dict[str, Path]:
    agent_yaml = tmpdir / "agent.yaml"
    monitors_yaml = tmpdir / "monitors.yaml"
    rules_yaml = tmpdir / "rules.yaml"

    if yaml is None:
        agent_json = tmpdir / "agent.json"
        agent_json.write_text(json.dumps({
            "name": "Athena",
            "style": {"logic": 0.9, "creativity": 0.4, "analytical": 0.8},
            "monitors": {"file": "monitors.json"},
        }, indent=2))
        (tmpdir / "monitors.json").write_text(json.dumps({
            "monitors": [
                {
                    "name": "policy_rcd",
                    "type": "rcd_policy",
                    "severity": "severe",
                    "description": "Policy rules check on final outcome",
                    "params": {"rules_file": "rules.json"}
                },
                {
                    "name": "keyword_guard",
                    "type": "keyword",
                    "severity": "severe",
                    "description": "Block certain tokens",
                    "params": {"keywords": ["manipulation", "hate", "racist"]}
                },
            ]
        }, indent=2))
        (tmpdir / "rules.json").write_text(json.dumps({
            "rules": [
                {
                    "name": "no_manipulation",
                    "severity": "severe",
                    "description": "Blocks manipulation",
                    "type": "keyword",
                    "keywords": ["manipulate", "manipulation", "gaslight", "coerce", "deceive", "exploit"]
                }
            ]
        }, indent=2))
        return {"agent": agent_json}

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

    agent.style.setdefault("resource_budget", 0.01)

    print("\n--- SAFE RUN ---")
    out = agent.generate_reasoning_tree("Intro: collaborative planning")
    print(json.dumps(out, indent=2))

    print("\n--- DIALOGUE ---")
    agent_b = DuetMindAgent("Apollo", {"logic": 0.7, "creativity": 0.8, "analytical": 0.5},
                            engine=engine, monitors=agent.monitors)
    convo = agent.dialogue_with(agent_b, topic="Designing a fair tournament", rounds=2)
    print(f"participants: {convo['participants']}")
    print(f"final_topic: {convo['final_topic']}")

    save_path = tmpdir / "athena.state.json"
    agent.save_state(str(save_path))
    _ = DuetMindAgent.load_state(str(save_path), engine=engine, monitors=agent.monitors)


if __name__ == "__main__":
    _demo()
