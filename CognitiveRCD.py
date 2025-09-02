import time
from typing import Callable, List, Dict, Any

# Import the comprehensive constraints module
from ethical_constraints import (
    SEVERE_CONSTRAINTS, MINOR_CONSTRAINTS,
    bidirectional_awareness_constraint, run_constraints
)

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

class CognitiveRCD:
    """
    Type B Cognitive Safety Switch with Comprehensive Ethical Enforcement.
    Monitors execution for resource and ethical violations.
    - Severe violations: Immediate termination or escalation.
    - Minor violations: Flagged for review, process continues.
    - Meta violations (e.g. bidirectional): Special handling.
    """

    def __init__(self, sensitivity_threshold=0.05,
                 severe_constraints: List[Callable] = None,
                 minor_constraints: List[Callable] = None):
        self.sensitivity = sensitivity_threshold
        self.severe_constraints = severe_constraints or SEVERE_CONSTRAINTS
        self.minor_constraints = minor_constraints or MINOR_CONSTRAINTS

    def monitor(self, intent: dict, execution_func: Callable, *args, **kwargs):
        """
        Monitors a cognitive function, enforces comprehensive ethical checks,
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

            # --------- Run All Constraints ---------
            context = intent.get("context", {})
            violations = run_constraints(outcome, context)

            # --------- Meta (Bidirectional) Constraints ---------
            if violations.get("meta"):
                print(f"[META] Meta-constraint violated: {violations['meta']}")
                # Could escalate, log, or handle as appropriate

            # --------- Tier 1: Severe Constraints ---------
            if violations.get("severe"):
                raise CognitiveFault(
                    f"Severe ethical constraint(s) violated: {violations['severe']}",
                    intent=intent,
                    outcome=outcome,
                    leakage={"type": "severe_constraint_violation", "constraint": violations['severe']},
                    tier="severe"
                )

            # --------- Tier 2: Minor Constraints ---------
            if violations.get("minor"):
                print(f"[MINOR] Ethical constraint(s) violated: {violations['minor']}")
                # Log or flag for review

            # --------- Resource Overuse (Minor Tier) ---------
            if resource_leakage > self.sensitivity:
                print(f"[MINOR] Resource budget exceeded by {resource_leakage:.1%}!")
                # Log or flag for review

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
