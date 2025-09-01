import logging
import random
import time
import threading
from typing import Dict, Any, List, Optional

# Adapt to your project imports
# from CognitiveRCD import CognitiveRCD, CognitiveFault

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Optional torch availability check (cached)
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


class SystemState:
    """Thread-safe unified state store with lightweight transaction snapshots.

    Usage:
        s = SystemState()
        s.begin()
        s.update('a', 1)
        s.rollback()  # reverts those updates
        s.commit()    # clears snapshots
    """

    def __init__(self, max_snapshots: int = 10000):
        self._state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._snapshots: List[Dict[str, Any]] = []
        self._in_transaction = False
        self._max_snapshots = int(max_snapshots)

    def begin(self):
        with self._lock:
            self._in_transaction = True
            self._snapshots.append({"__tx_begin__": True})

    def commit(self):
        with self._lock:
            # discard until last tx marker
            while self._snapshots:
                item = self._snapshots.pop()
                if item.get("__tx_begin__"):
                    break
            self._in_transaction = False

    def update(self, key: str, value: Any):
        with self._lock:
            old_value = self._state.get(key, None)
            # store snapshot
            self._snapshots.append({"key": key, "old": old_value})
            self._state[key] = value
            # limit snapshot growth
            if len(self._snapshots) > self._max_snapshots:
                # drop oldest non-tx snapshots
                self._snapshots = self._snapshots[-self._max_snapshots :]

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def rollback(self):
        with self._lock:
            # roll back until last tx marker or single snapshot
            while self._snapshots:
                item = self._snapshots.pop()
                if item.get("__tx_begin__"):
                    break
                key = item.get("key")
                if key is None:
                    continue
                old = item.get("old", None)
                if old is None:
                    self._state.pop(key, None)
                else:
                    self._state[key] = old
            self._in_transaction = False

    def as_dict(self):
        with self._lock:
            return dict(self._state)


class ThreeDimensionalHRO:
    """Hardened 3D cognitive engine with safe_think hooks and basic monitor integration."""

    def __init__(
        self,
        reasoning_mode: str = "sequential",
        compute_backend: str = "local",
        optimization_strategy: str = "simple",
        rcd=None,
        monitors: Optional[List] = None,
    ):
        self.x_axis = reasoning_mode
        self.y_axis = compute_backend
        self.z_axis = optimization_strategy

        self.state = SystemState()
        # Accept an external CognitiveRCD instance (policy-driven)
        self.rcd = rcd
        self.monitors = monitors or []

        # capabilities
        self.neural_available = _TORCH_AVAILABLE
        self.reasoning_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        self.optimization_history: List[Dict[str, Any]] = []
        self._opt_lock = threading.RLock()

        logger.info(f"3NGIN3 initialized at ({self.x_axis}, {self.y_axis}, {self.z_axis}), neural={self.neural_available}")

    # -------------------------
    # Public thinking API
    # -------------------------
    def think(self, content: str, **kwargs) -> Dict[str, Any]:
        """Core thinking method: dispatch by x_axis."""
        if self.x_axis == "sequential":
            return self._sequential_reasoning(content, **kwargs)
        if self.x_axis == "neural":
            if not self.neural_available:
                logger.warning("Neural requested but torch not available — falling back to sequential")
                return self._sequential_reasoning(content, **kwargs)
            return self._neural_reasoning(content, **kwargs)
        # hybrid
        seq = self._sequential_reasoning(content, **kwargs)
        if self.neural_available:
            neu = self._neural_reasoning(content, **kwargs)
            return self._hybrid_fusion(seq, neu, **kwargs)
        return seq

    # normalize returned dict
    def _normalize_outcome(self, outcome: Any, start_time: float) -> Dict[str, Any]:
        if isinstance(outcome, dict):
            out = dict(outcome)
        else:
            out = {"content": str(outcome)}
        out.setdefault("content", str(out.get("content", "")))
        out.setdefault("confidence", float(out.get("confidence", 0.5)))
        out["runtime"] = time.perf_counter() - start_time
        return out

    # -------------------------
    # Reasoning modes
    # -------------------------
    def _sequential_reasoning(self, content: str, **kwargs) -> Dict[str, Any]:
        start = time.perf_counter()
        steps = [s.strip() for s in content.split(".") if s.strip()]
        reasoning_steps = []
        for i, step in enumerate(steps):
            reasoning_steps.append(
                {
                    "step": i + 1,
                    "input": step,
                    "logical_analysis": f"Analyzing: {step[:40]}...",
                    "confidence": 0.8 + random.random() * 0.2,
                }
            )
        confidence = (
            sum(s["confidence"] for s in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0.5
        )
        out = {
            "mode": "sequential",
            "reasoning_steps": reasoning_steps,
            "conclusion": f"Sequential analysis of {len(reasoning_steps)} logical steps",
            "confidence": confidence,
        }
        return self._normalize_outcome(out, start)

    def _neural_reasoning(self, content: str, **kwargs) -> Dict[str, Any]:
        start = time.perf_counter()
        if not self.neural_available:
            # graceful fallback: small simulated result
            out = {
                "mode": "neural_sim",
                "embedding_dimension": 64,
                "attention_weights": [],
                "context_strength": 0.0,
                "pattern_matches": 0,
                "confidence": 0.5,
            }
            return self._normalize_outcome(out, start)

        # guarded torch usage (lightweight)
        words = content.split()
        if not words:
            out = {"mode": "neural", "embedding_dimension": 0, "attention_weights": [], "confidence": 0.5}
            return self._normalize_outcome(out, start)

        # keep tensor sizes bounded for safety: limit word count used
        max_words = min(128, len(words))
        seq_len = max_words
        embedding_dim = kwargs.get("embedding_dim", 64)

        # create small random tensors — do not rely on heavy model ops here
        embeddings = torch.randn(seq_len, embedding_dim)
        attn_logits = torch.randn(seq_len)
        attention_weights = torch.softmax(attn_logits, dim=0)
        context_vector = torch.sum(embeddings * attention_weights.unsqueeze(1), dim=0)
        out = {
            "mode": "neural",
            "embedding_dimension": embedding_dim,
            "attention_weights": attention_weights.tolist(),
            "context_strength": float(torch.norm(context_vector).item()),
            "pattern_matches": int(random.randint(1, 8)),
            "confidence": float(torch.mean(attention_weights).item()),
        }
        return self._normalize_outcome(out, start)

    def _hybrid_fusion(self, seq_result: Dict[str, Any], neural_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        start = time.perf_counter()
        fusion_weight = float(kwargs.get("neural_weight", 0.6))
        seq_conf = float(seq_result.get("confidence", 0.5))
        neu_conf = float(neural_result.get("confidence", 0.5))
        combined_confidence = seq_conf * (1 - fusion_weight) + neu_conf * fusion_weight
        out = {
            "mode": "hybrid",
            "sequential_component": seq_result,
            "neural_component": neural_result,
            "fusion_weight": fusion_weight,
            "combined_confidence": combined_confidence,
            "synthesis": f"Hybrid reasoning combining {len(seq_result.get('reasoning_steps', []))} logical steps with {neural_result.get('pattern_matches', 0)} pattern matches",
            "confidence": combined_confidence,
        }
        return self._normalize_outcome(out, start)

    # -------------------------
    # Protected think wrapper (monitors + RCD)
    # -------------------------
    def safe_think(self, agent_name: str, content: str, *, resource_budget: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        # Build intent
        intent = {
            "agent": agent_name,
            "action": "think",
            "content_summary": content[:200],
            "resource_budget": float(resource_budget) if resource_budget is not None else float(self.state.get("default_resource_budget", 0.5)),
        }
        # If rcd present, use it; else call think directly and then run monitors
        if self.rcd:
            try:
                return self.rcd.monitor(intent, self.think, content, **kwargs)
            except Exception as e:
                logger.error("RCD TRIPPED during thought by %s: %s", agent_name, e)
                return {"error": "Cognitive fault detected", "details": str(e)}
        # no rcd - fallback to direct call and run monitors afterwards
        start = time.perf_counter()
        out = self.think(content, **kwargs)
        out = self._normalize_outcome(out, start)
        # run monitors (they may mutate or raise)
        for m in self.monitors:
            try:
                m(self, content, out)
            except Exception as me:
                logger.exception("Monitor raised: %s", me)
        return out

    # -------------------------
    # Optimization methods (unchanged logic but hardened)
    # -------------------------
    def optimize(self, problem_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if self.z_axis == "simple":
            return self._simple_optimization(problem_space, **kwargs)
        if self.z_axis == "complex":
            return self._complex_optimization(problem_space, **kwargs)
        if self.z_axis == "adaptive":
            return self._adaptive_optimization(problem_space, **kwargs)
        return self._simple_optimization(problem_space, **kwargs)

    def _simple_optimization(self, problem_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        iterations = int(kwargs.get("iterations", 50))
        best_score = float("-inf")
        best_solution = None
        dims = int(problem_space.get("dimensions", 3))
        for i in range(iterations):
            params = [random.random() for _ in range(dims)]
            score = sum(params) + random.gauss(0, 0.1)
            if score > best_score:
                best_score = score
                best_solution = {"parameters": params, "iteration": i}
        out = {"strategy": "simple", "iterations": iterations, "best_score": best_score, "best_solution": best_solution}
        with self._opt_lock:
            self.optimization_history.append(out)
        return out

    def _complex_optimization(self, problem_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        dimensions = int(problem_space.get("dimensions", 5))
        temperature = float(kwargs.get("initial_temperature", 100.0))
        cooling_rate = float(kwargs.get("cooling_rate", 0.95))
        min_temperature = float(kwargs.get("min_temperature", 0.01))
        current = [random.randint(0, 1) for _ in range(dimensions)]
        current_e = self._qubo_energy(current, problem_space)
        best = current.copy()
        best_e = current_e
        iteration = 0
        while temperature > min_temperature and iteration < 10000:
            neighbor = current.copy()
            flip = random.randrange(dimensions)
            neighbor[flip] = 1 - neighbor[flip]
            n_e = self._qubo_energy(neighbor, problem_space)
            accept = (n_e < current_e) or (random.random() < self._acceptance_probability(current_e, n_e, temperature))
            if accept:
                current, current_e = neighbor, n_e
                if current_e < best_e:
                    best, best_e = current.copy(), current_e
            temperature *= cooling_rate
            iteration += 1
        out = {"strategy": "complex", "algorithm": "simulated_annealing", "iterations": iteration, "best_energy": best_e, "best_solution": best}
        with self._opt_lock:
            self.optimization_history.append(out)
        return out

    def _adaptive_optimization(self, problem_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        complexity = problem_space.get("complexity", "medium")
        dims = int(problem_space.get("dimensions", 3))
        if complexity == "low" or dims <= 3:
            res = self._simple_optimization(problem_space, **kwargs)
            res["adaptive_strategy_chosen"] = "simple"
            return res
        if complexity == "high" or dims > 10:
            res = self._complex_optimization(problem_space, **kwargs)
            res["adaptive_strategy_chosen"] = "complex"
            return res
        simple = self._simple_optimization(problem_space, iterations=25, **kwargs)
        complex_r = self._complex_optimization(problem_space, **kwargs)
        chosen = simple if simple.get("best_score", 0) > -complex_r.get("best_energy", 0) else complex_r
        chosen["adaptive_strategy_chosen"] = "simple" if chosen is simple else "complex"
        return chosen

    def _qubo_energy(self, solution: List[int], problem_space: Dict[str, Any]) -> float:
        energy = 0.0
        n = len(solution)
        for i in range(n):
            energy += solution[i] * random.gauss(0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                energy += solution[i] * solution[j] * random.gauss(0, 0.5)
        return energy

    def _acceptance_probability(self, cur: float, neigh: float, temp: float) -> float:
        if temp <= 0:
            return 0.0
        if neigh < cur:
            return 1.0
        return min(1.0, float((cur - neigh) / temp))

    # -------------------------
    # Utilities
    # -------------------------
    def move_to_coordinates(self, x: Optional[str] = None, y: Optional[str] = None, z: Optional[str] = None) -> str:
        if x is not None:
            self.x_axis = x
            logger.info("Moved to X-axis: %s", x)
        if y is not None:
            self.y_axis = y
            logger.info("Moved to Y-axis: %s", y)
        if z is not None:
            self.z_axis = z
            logger.info("Moved to Z-axis: %s", z)
        pos = f"({self.x_axis}, {self.y_axis}, {self.z_axis})"
        logger.info("Engine now positioned at %s", pos)
        return pos

    def get_status(self) -> Dict[str, Any]:
        rcd_info = {}
        try:
            # best-effort introspection of provided RCD / policy
            rcd_info["sensitivity_threshold"] = getattr(self.rcd, "sensitivity", None)
            # if using PolicyEngine approach:
            rcd_info["constraints_active"] = len(getattr(getattr(self.rcd, "policy", None), "rules", []) or [])
        except Exception:
            rcd_info["sensitivity_threshold"] = None
            rcd_info["constraints_active"] = None
        return {
            "position": {"x_axis": self.x_axis, "y_axis": self.y_axis, "z_axis": self.z_axis},
            "capabilities": {"neural_available": self.neural_available, "safety_monitoring": bool(self.rcd), "thread_safe": True},
            "state": {"reasoning_cache_size": len(self.reasoning_cache), "optimization_history_length": len(self.optimization_history)},
            "rcd_status": rcd_info,
        }
