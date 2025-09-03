import logging
import random
import time
import threading
from typing import Dict, Any, List, Optional

# --- Logging setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --- Torch availability check ---
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

# --- Thread-safe system state for cognitive engine ---
class SystemState:
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
            while self._snapshots:
                item = self._snapshots.pop()
                if item.get("__tx_begin__"):
                    break
            self._in_transaction = False

    def update(self, key: str, value: Any):
        with self._lock:
            old_value = self._state.get(key, None)
            self._snapshots.append({"key": key, "old": old_value})
            self._state[key] = value
            if len(self._snapshots) > self._max_snapshots:
                self._snapshots = self._snapshots[-self._max_snapshots :]

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def rollback(self):
        with self._lock:
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

# --- Gut-brain system state ---
class MicrobiomeSystemState:
    def __init__(self):
        self.anxiety = 0
        self.overload = 0
        self.memory = 0
        self.cyber_defense = 5
        self.health_score = 100

    def absorb_overload(self, amount):
        self.overload = max(0, self.overload - amount)

    def boost_memory(self, amount):
        self.memory += amount

    def trigger_anxiety(self, amount):
        self.anxiety += amount

    def defend(self, attack_strength):
        if self.cyber_defense >= attack_strength:
            print("Defended against attack!")
        else:
            print("Attack penetrated defenses!")

    def update_health(self, delta):
        self.health_score = max(0, min(100, self.health_score + delta))

# --- Microbiome species ---
class MicrobiomeSpecies:
    def __init__(self, name, role, effect, is_bad=False, abundance=1):
        self.name = name
        self.role = role
        self.effect = effect
        self.is_bad = is_bad
        self.abundance = abundance

    def act(self, system):
        for _ in range(self.abundance):
            self.effect(system)

# --- Neuron loop (simulated) ---
class NeuronLoop:
    def __init__(self):
        self.activity = 0
        self.memory = 0

    def stimulate(self, amount):
        self.activity += amount

    def rest(self):
        self.activity = max(0, self.activity - 1)

# --- Vagus nerve: gut-brain bridge ---
class VagusNerve:
    def __init__(self, gut_state: MicrobiomeSystemState, cognitive_state: SystemState):
        self.gut_state = gut_state
        self.cognitive_state = cognitive_state

    def transmit_signals(self):
        try:
            # Relay gut state to cognitive state
            self.cognitive_state.update("anxiety_level", self.gut_state.anxiety)
            self.cognitive_state.update("overload_level", self.gut_state.overload)
            self.cognitive_state.update("memory_score", self.gut_state.memory)
            self.cognitive_state.update("cyber_defense", self.gut_state.cyber_defense)
            self.cognitive_state.update("microbiome_health", self.gut_state.health_score)
            logging.info("[VagusNerve] Signals transmitted from gut to brain.")
        except Exception as e:
            logging.error(f"[VagusNerve] Transmission error: {e}")

# --- ThreeDimensionalHRO cognitive engine ---
class ThreeDimensionalHRO:
    def __init__(
        self,
        reasoning_mode: str = "sequential",
        compute_backend: str = "local",
        optimization_strategy: str = "simple",
        rcd=None,
        monitors: Optional[List] = None,
        species_capacity: int = 10,
    ):
        self.x_axis = reasoning_mode
        self.y_axis = compute_backend
        self.z_axis = optimization_strategy
        self.state = SystemState()
        self.rcd = rcd
        self.monitors = monitors or []
        self.neural_available = _TORCH_AVAILABLE
        self.reasoning_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        self.optimization_history: List[Dict[str, Any]] = []
        self._opt_lock = threading.RLock()

        # Gut-brain system
        self.microbiome_state = MicrobiomeSystemState()
        self.neurons = [NeuronLoop() for _ in range(10)]
        self.species_capacity = species_capacity
        self.microbiome = self._init_microbiome_species()
        self.vagus_nerve = VagusNerve(self.microbiome_state, self.state)

        logger.info(f"3NGIN3 initialized at ({self.x_axis}, {self.y_axis}, {self.z_axis}), neural={self.neural_available}")

    # --- Microbiome species & population dynamics ---
    def _init_microbiome_species(self):
        helpers = [
            MicrobiomeSpecies(
                name="Lactobacillus", role="anxiety_reducer",
                effect=lambda sys: sys.absorb_overload(2), abundance=2
            ),
            MicrobiomeSpecies(
                name="Bifidobacterium", role="memory_helper",
                effect=lambda sys: sys.boost_memory(1), abundance=2
            ),
            MicrobiomeSpecies(
                name="DopamineActivator", role="activity_boost",
                effect=lambda sys: [n.stimulate(2) for n in self.neurons], abundance=1
            ),
            MicrobiomeSpecies(
                name="Akkermansia", role="barrier_strengthener",
                effect=lambda sys: setattr(sys, 'cyber_defense', sys.cyber_defense + 1), abundance=1
            ),
            MicrobiomeSpecies(
                name="Faecalibacterium", role="anti_inflammatory",
                effect=lambda sys: sys.absorb_overload(1), abundance=1
            ),
        ]
        attackers = [
            MicrobiomeSpecies(
                name="Pathogenus", role="anxiety_trigger",
                effect=lambda sys: sys.trigger_anxiety(3), is_bad=True, abundance=1
            ),
            MicrobiomeSpecies(
                name="Clostridium_difficile", role="memory_disruptor",
                effect=lambda sys: setattr(sys, 'memory', max(0, sys.memory - 2)), is_bad=True, abundance=1
            ),
            MicrobiomeSpecies(
                name="E_coli_pathogenic", role="defense_weakener",
                effect=lambda sys: setattr(sys, 'cyber_defense', max(1, sys.cyber_defense - 1)), is_bad=True, abundance=1
            ),
        ]
        species = helpers + attackers
        if len(species) > self.species_capacity:
            species = random.sample(species, self.species_capacity)
        return species

    def update_population_dynamics(self):
        beneficial = [s for s in self.microbiome if not s.is_bad]
        pathogenic = [s for s in self.microbiome if s.is_bad]
        # Competition
        if len(pathogenic) > len(beneficial):
            for b in beneficial:
                b.abundance = max(1, b.abundance - 1)
            self.microbiome_state.update_health(-10)
        if len(beneficial) > len(pathogenic):
            for p in pathogenic:
                p.abundance = max(1, p.abundance - 1)
            self.microbiome_state.update_health(+5)
        # Symbiosis
        names = [s.name for s in beneficial]
        if "Akkermansia" in names and "Faecalibacterium" in names:
            for s in beneficial:
                if s.name in ("Akkermansia", "Faecalibacterium"):
                    s.abundance += 1

    def integrate_training_data(self, dataset):
        # Placeholder: dataset should be a dict {species_name: abundance}
        for s in self.microbiome:
            if s.name in dataset:
                s.abundance = dataset[s.name]

    def check_microbiome_safety(self):
        health = self.microbiome_state.health_score
        if health < 50:
            print("Health score low! Triggering probiotic intervention.")
            for s in self.microbiome:
                if not s.is_bad:
                    s.abundance += 1
            self.microbiome_state.update_health(+20)
        pathogenic_load = sum(s.abundance for s in self.microbiome if s.is_bad)
        if pathogenic_load > 5:
            print("High pathogenic load! Activating immune response.")
            for s in self.microbiome:
                if s.is_bad:
                    s.abundance = max(1, s.abundance - 1)
            self.microbiome_state.update_health(+10)

    # --- Gut-brain simulation cycle ---
    def simulate_microbiome_phase(self, phase: str, dataset=None):
        print(f"\n--- Gut-Brain Phase: {phase} ---")
        try:
            if dataset:
                self.integrate_training_data(dataset)
            self.update_population_dynamics()
            self.check_microbiome_safety()
            for species in self.microbiome:
                species.act(self.microbiome_state)
            for n in self.neurons:
                if phase == "busy":
                    n.stimulate(random.randint(1, 3))
                elif phase == "rest":
                    n.rest()
                self.microbiome_state.memory += n.memory
            # Gut-brain signal relay
            self.vagus_nerve.transmit_signals()
            print(f"Anxiety (gut): {self.microbiome_state.anxiety}, Overload (gut): {self.microbiome_state.overload}, Memory (gut): {self.microbiome_state.memory}, Cyber Defense (gut): {self.microbiome_state.cyber_defense}, Health: {self.microbiome_state.health_score}")
            print(f"Cognitive State | anxiety_level: {self.state.get('anxiety_level')}, overload_level: {self.state.get('overload_level')}, memory_score: {self.state.get('memory_score')}, cyber_defense: {self.state.get('cyber_defense')}, microbiome_health: {self.state.get('microbiome_health')}")
        except Exception as e:
            logging.error(f"[MicrobiomePhase] Error during phase '{phase}': {e}")

    # --- Diagnostics and test methods ---
    def run_diagnostics(self):
        try:
            print("\n[Diagnostics]")
            print("Gut anxiety:", self.microbiome_state.anxiety)
            print("Cognitive anxiety:", self.state.get("anxiety_level"))
            print("Gut overload:", self.microbiome_state.overload)
            print("Cognitive overload:", self.state.get("overload_level"))
            print("Gut memory:", self.microbiome_state.memory)
            print("Cognitive memory:", self.state.get("memory_score"))
            print("Gut cyber defense:", self.microbiome_state.cyber_defense)
            print("Cognitive cyber defense:", self.state.get("cyber_defense"))
            print("Microbiome health score:", self.microbiome_state.health_score)
            print("Cognitive microbiome health:", self.state.get("microbiome_health"))
        except Exception as e:
            logging.error(f"[Diagnostics] Error: {e}")

    # --- Cognitive engine logic ---
    def think(self, content: str, **kwargs) -> Dict[str, Any]:
        if self.x_axis == "sequential":
            return self._sequential_reasoning(content, **kwargs)
        if self.x_axis == "neural":
            if not self.neural_available:
                logger.warning("Neural requested but torch not available — falling back to sequential")
                return self._sequential_reasoning(content, **kwargs)
            return self._neural_reasoning(content, **kwargs)
        seq = self._sequential_reasoning(content, **kwargs)
        if self.neural_available:
            neu = self._neural_reasoning(content, **kwargs)
            return self._hybrid_fusion(seq, neu, **kwargs)
        return seq

    def _normalize_outcome(self, outcome: Any, start_time: float) -> Dict[str, Any]:
        if isinstance(outcome, dict):
            out = dict(outcome)
        else:
            out = {"content": str(outcome)}
        out.setdefault("content", str(out.get("content", "")))
        out.setdefault("confidence", float(out.get("confidence", 0.5)))
        out["runtime"] = time.perf_counter() - start_time
        return out

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
            out = {
                "mode": "neural_sim",
                "embedding_dimension": 64,
                "attention_weights": [],
                "context_strength": 0.0,
                "pattern_matches": 0,
                "confidence": 0.5,
            }
            return self._normalize_outcome(out, start)
        words = content.split()
        if not words:
            out = {"mode": "neural", "embedding_dimension": 0, "attention_weights": [], "confidence": 0.5}
            return self._normalize_outcome(out, start)
        max_words = min(128, len(words))
        seq_len = max_words
        embedding_dim = kwargs.get("embedding_dim", 64)
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

    def safe_think(self, agent_name: str, content: str, *, resource_budget: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        intent = {
            "agent": agent_name,
            "action": "think",
            "content_summary": content[:200],
            "resource_budget": float(resource_budget) if resource_budget is not None else float(self.state.get("default_resource_budget", 0.5)),
        }
        if self.rcd:
            try:
                return self.rcd.monitor(intent, self.think, content, **kwargs)
            except Exception as e:
                logger.error("RCD TRIPPED during thought by %s: %s", agent_name, e)
                return {"error": "Cognitive fault detected", "details": str(e)}
        start = time.perf_counter()
        out = self.think(content, **kwargs)
        out = self._normalize_outcome(out, start)
        for m in self.monitors:
            try:
                m(self, content, out)
            except Exception as me:
                logger.exception("Monitor raised: %s", me)
        return out

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
        out = {"strategy": "complex", "algorithm": "simulated_annealing", "iterations": iteration, "best
