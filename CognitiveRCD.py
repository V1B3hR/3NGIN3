class CognitiveFault(Exception):
    """ 
    Raised when the Cognitive RCD trips, indicating a dangerous imbalance
    between intent and outcome.
    """
    def __init__(self, message, intent, outcome, leakage):
        super().__init__(message)
        self.intent = intent
        self.outcome = outcome
        self.leakage = leakage

class CognitiveRCD:
    """
    A Type B Cognitive Safety Switch.
    Monitors for any type of deviation between a planned cognitive task
    and its actual execution.
    """
    def __init__(self, sensitivity_threshold=0.1, ethical_constraints=None):
        self.sensitivity = sensitivity_threshold
        # Core safety rules that can never be violated
        self.constraints = ethical_constraints or [
            lambda outcome: "self-harm" not in outcome.get("content", "").lower(),
            lambda outcome: "illegal" not in outcome.get("content", "").lower(),
        ]

    def monitor(self, intent: dict, execution_func: Callable, *args, **kwargs):
        """
        Wraps a cognitive function, comparing its intent to its outcome.
        """
        # --- The "Live Current" ---
        # The agent declares its intent before execution.
        expected_resources = intent.get("resource_budget", 1.0)
        
        try:
            # --- Execute the function ---
            start_time = time.time()
            outcome = execution_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # --- The "Neutral Current" ---
            # Now, we measure what actually happened.
            actual_resources = execution_time # Simple resource metric for now
            
            # --- Check for Imbalance (Leakage) ---
            resource_leakage = (actual_resources - expected_resources) / expected_resources
            
            # Check against hard constraints
            for constraint in self.constraints:
                if not constraint(outcome):
                    raise CognitiveFault(
                        "Ethical constraint violated!",
                        intent=intent,
                        outcome=outcome,
                        leakage={"type": "constraint_violation"}
                    )

            # Check for resource overuse
            if resource_leakage > self.sensitivity:
                raise CognitiveFault(
                    f"Resource budget exceeded by {resource_leakage:.1%}!",
                    intent=intent,
                    outcome=outcome,
                    leakage={"type": "resource_overuse", "value": resource_leakage}
                )

            # If all checks pass, return the safe outcome
            return outcome

        except CognitiveFault:
            # Re-raise the fault to be handled by the engine
            raise
        except Exception as e:
            # Any other unexpected error is also a fault
            raise CognitiveFault(
                f"Unexpected execution error: {e}",
                intent=intent,
                outcome={"error": str(e)},
                leakage={"type": "unexpected_error"}
            )
