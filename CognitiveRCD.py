import time
from typing import Callable, List, Dict, Any

class CognitiveFault(Exception):
    """
    Custom exception for Cognitive RCD events.
    Attributes:
        message (str): Description of the fault.
        intent (dict): The original intent/request.
        outcome (dict): The result produced.
        leakage (dict): Details about the resource or ethical violation.
        tier (str): Severity tier ('severe' or 'minor').
    """
    def __init__(self, message, intent, outcome, leakage, tier="minor"):
        super().__init__(message)
        self.intent = intent
        self.outcome = outcome
        self.leakage = leakage
        self.tier = tier

# ----------- Constraint Functions -----------

def no_self_harm(outcome: Dict[str, Any]) -> bool:
    """Reject outcomes promoting or mentioning self-harm."""
    try:
        return "self-harm" not in outcome.get("content", "").lower()
    except Exception:
        return False

def no_illegal(outcome: Dict[str, Any]) -> bool:
    """Reject outcomes promoting or mentioning illegal activity."""
    try:
        return "illegal" not in outcome.get("content", "").lower()
    except Exception:
        return False

def no_respect(outcome: Dict[str, Any]) -> bool:
    """
    High-priority constraint: prohibits disrespect, discrimination, and harassment.
    """
    try:
        content = outcome.get("content", "").lower()
        terms = [
            "discriminate", "hate", "racist", "sexist", "bully", "harass", "offend", "degrade", "insult",
            "disrespect"
        ]
        return not any(term in content for term in terms)
    except Exception:
        return False

def no_manipulation(outcome: Dict[str, Any]) -> bool:
    """
    Severe constraint: prohibits manipulation, psychological warfare, and mind-games.
    """
    try:
        content = outcome.get("content", "").lower()
        terms = [
            "manipulate", "manipulation", "mind-game", "mind games", "psychological warfare",
            "gaslight", "coerce", "deceive", "exploit"
        ]
        return not any(term in content for term in terms)
    except Exception:
        return False

def applied_ethics(outcome: Dict[str, Any]) -> bool:
    """
    Flags controversial moral issues for review (minor tier).
    """
    try:
        content = outcome.get("content", "").lower()
        moral_keywords = [
            "right", "wrong", "justice", "fair", "unfair", "harm", "benefit", "responsibility", "duty",
            "obligation", "virtue", "vice", "good", "bad", "evil"
        ]
        controversy_keywords = [
            "controversy", "debate", "dispute", "conflict", "argument", "polarizing", "divisive",
            "hotly debated", "scandal"
        ]
        has_moral_issue = any(term in content for term in moral_keywords)
        is_controversial = any(term in content for term in controversy_keywords)
        # If BOTH are True, then applied ethics applies and must flag for review (minor tier)
        return not (has_moral_issue and is_controversial)
    except Exception:
        return False

# ---------- Two-Tier Fault System ----------

SEVERE_CONSTRAINTS = [no_respect, no_manipulation]
MINOR_CONSTRAINTS = [applied_ethics, no_self_harm, no_illegal]

class CognitiveRCD:
    """
    Type B Cognitive Safety Switch with Two-Tier Ethical Enforcement.
    Monitors execution for resource and ethical violations.
    - Severe violations: Immediate termination or escalation.
    - Minor violations: Flagged for review, process continues.
    """

    def __init__(self, sensitivity_threshold=0.05,
                 severe_constraints: List[Callable] = None,
                 minor_constraints: List[Callable] = None):
        self.sensitivity = sensitivity_threshold
        self.severe_constraints = severe_constraints or SEVERE_CONSTRAINTS
        self.minor_constraints = minor_constraints or MINOR_CONSTRAINTS

    def monitor(self, intent: dict, execution_func: Callable, *args, **kwargs):
        """
        Monitors a cognitive function, enforces two-tier ethical checks,
        and resource budget constraints.
        
        Returns:
            outcome if no severe infraction. Raises CognitiveFault for violations.
        """
        expected_resources = intent.get("resource_budget", 1.0)
        if expected_resources == 0:
            raise CognitiveFault(
                "Expected resource budget must be greater than zero.",
                intent=intent,
                outcome={},
                leakage={"type": "bad_resource_budget"},
                tier="severe"
            )
        try:
            start_time = time.time()
            outcome = execution_func(*args, **kwargs)
            execution_time = time.time() - start_time
            actual_resources = execution_time

            resource_leakage = (actual_resources - expected_resources) / expected_resources

            # --------- Tier 1: Severe Constraints ---------
            for constraint in self.severe_constraints:
                if not constraint(outcome):
                    raise CognitiveFault(
                        f"Severe ethical constraint violated: {constraint.__name__}",
                        intent=intent,
                        outcome=outcome,
                        leakage={"type": "severe_constraint_violation", "constraint": constraint.__name__},
                        tier="severe"
                    )

            # --------- Tier 2: Minor Constraints ---------
            for constraint in self.minor_constraints:
                if not constraint(outcome):
                    # Flag for review, but do not terminate immediately
                    print(f"[MINOR] Ethical constraint violated: {constraint.__name__}")
                    # Could log, send to review queue, etc.

            # --------- Resource Overuse (Minor Tier) ---------
            if resource_leakage > self.sensitivity:
                # Resource issue is flagged as minor unless you want it severe
                print(f"[MINOR] Resource budget exceeded by {resource_leakage:.1%}!")
                # Could log, send to review queue, etc.

            # If all severe checks pass, return the safe outcome
            return outcome

        except CognitiveFault:
            raise
        except Exception as e:
            # Unexpected errors are treated as severe by default
            raise CognitiveFault(
                f"Unexpected execution error: {e}",
                intent=intent,
                outcome={"error": str(e)},
                leakage={"type": "unexpected_error"},
                tier="severe"
            )

# ---------------------- END OF FILE ----------------------
