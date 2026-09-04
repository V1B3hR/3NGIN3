from __future__ import annotations

import logging
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, TypeVar

# --- Logging setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --- Optional Torch support ---
try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


T = TypeVar("T")
ImageEncoder = Callable[..., Any]
CognitiveEngine = Callable[..., Any]
Monitor = Callable[["ThreeDimensionalHRO", str, Dict[str, Any]], None]
SpeciesEffect = Callable[["MicrobiomeSystemState"], None]

# --- Component registries ---
image_encoders: Dict[str, ImageEncoder] = {}
cognitive_engines: Dict[str, CognitiveEngine] = {}
microbiome_species_registry: Dict[str, Callable[..., "MicrobiomeSpecies"]] = {}


def register_image_encoder(
    fn: Optional[ImageEncoder] = None,
    *,
    name: Optional[str] = None,
) -> Callable[[ImageEncoder], ImageEncoder] | ImageEncoder:
    """Register an image encoder by an explicit name or function name."""

    def decorator(encoder: ImageEncoder) -> ImageEncoder:
        encoder_name = (name or encoder.__name__).lower()
        image_encoders[encoder_name] = encoder
        return encoder

    return decorator(fn) if fn is not None else decorator


def register_cognitive_engine(
    name: Optional[str] = None,
) -> Callable[[CognitiveEngine], CognitiveEngine]:
    """Register a cognitive-engine factory or class."""

    def decorator(engine: CognitiveEngine) -> CognitiveEngine:
        engine_name = (name or engine.__name__).lower()
        cognitive_engines[engine_name] = engine
        return engine

    return decorator


def register_microbiome_species(
    species_class: Callable[..., "MicrobiomeSpecies"],
) -> Callable[..., "MicrobiomeSpecies"]:
    """Register a microbiome species class using its lowercase class name."""
    microbiome_species_registry[species_class.__name__.lower()] = species_class
    return species_class


def get_image_encoder(model_name: str) -> ImageEncoder:
    normalized_name = model_name.lower()
    try:
        return image_encoders[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unknown image encoder: {model_name}") from error


def get_cognitive_engine(engine_name: str) -> CognitiveEngine:
    normalized_name = engine_name.lower()
    try:
        return cognitive_engines[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unknown cognitive engine: {engine_name}") from error


def get_microbiome_species(
    species_name: str,
) -> Callable[..., "MicrobiomeSpecies"]:
    normalized_name = species_name.lower()
    try:
        return microbiome_species_registry[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unknown microbiome species: {species_name}") from error


def is_image_encoder(model_name: str) -> bool:
    return model_name.lower() in image_encoders


def is_cognitive_engine(engine_name: str) -> bool:
    return engine_name.lower() in cognitive_engines


def build_image_encoder(
    config_encoder: Mapping[str, Any],
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Build an image encoder from a configuration mapping."""
    try:
        model_name = str(config_encoder["NAME"])
    except KeyError as error:
        raise ValueError("Image encoder config must contain a 'NAME' key.") from error

    if model_name.lower().startswith("cls_"):
        model_name = model_name[4:]

    return get_image_encoder(model_name)(config_encoder, verbose=verbose, **kwargs)


def build_cognitive_engine(engine_name: str, **kwargs: Any) -> Any:
    """Build a registered cognitive engine."""
    return get_cognitive_engine(engine_name)(**kwargs)


class SystemState:
    """Thread-safe key/value state with basic nested transaction support."""

    _MISSING = object()

    def __init__(self, max_snapshots: int = 10_000):
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be at least 1.")

        self._state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._snapshots: List[Dict[str, Any]] = []
        self._transaction_depth = 0
        self._max_snapshots = max_snapshots

    @property
    def in_transaction(self) -> bool:
        with self._lock:
            return self._transaction_depth > 0

    def begin(self) -> None:
        """Start a transaction. Nested transactions are supported."""
        with self._lock:
            self._snapshots.append({"transaction_marker": True})
            self._transaction_depth += 1

    def commit(self) -> None:
        """Commit the most recently opened transaction."""
        with self._lock:
            if self._transaction_depth == 0:
                raise RuntimeError("No active transaction to commit.")

            while self._snapshots:
                snapshot = self._snapshots.pop()
                if snapshot.get("transaction_marker"):
                    break

            self._transaction_depth -= 1

    def update(self, key: str, value: Any) -> None:
        """Set a value and record the old value when inside a transaction."""
        with self._lock:
            if self._transaction_depth > 0:
                self._snapshots.append(
                    {
                        "key": key,
                        "existed": key in self._state,
                        "old_value": self._state.get(key),
                    }
                )
                self._trim_snapshots()

            self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def remove(self, key: str) -> None:
        """Remove a key, retaining it for rollback when appropriate."""
        with self._lock:
            if key not in self._state:
                return

            if self._transaction_depth > 0:
                self._snapshots.append(
                    {
                        "key": key,
                        "existed": True,
                        "old_value": self._state[key],
                    }
                )
                self._trim_snapshots()

            del self._state[key]

    def rollback(self) -> None:
        """Roll back all changes made in the most recent transaction."""
        with self._lock:
            if self._transaction_depth == 0:
                raise RuntimeError("No active transaction to roll back.")

            while self._snapshots:
                snapshot = self._snapshots.pop()
                if snapshot.get("transaction_marker"):
                    break

                key = snapshot["key"]
                if snapshot["existed"]:
                    self._state[key] = snapshot["old_value"]
                else:
                    self._state.pop(key, None)

            self._transaction_depth -= 1

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _trim_snapshots(self) -> None:
        """Keep snapshot memory bounded without dropping transaction markers."""
        if len(self._snapshots) <= self._max_snapshots:
            return

        markers = [
            index
            for index, snapshot in enumerate(self._snapshots)
            if snapshot.get("transaction_marker")
        ]
        if markers:
            first_marker = markers[0]
            self._snapshots = self._snapshots[first_marker:]
        else:
            self._snapshots = self._snapshots[-self._max_snapshots:]


class MicrobiomeSystemState:
    """Thread-safe mutable state shared by microbiome species and neurons."""

    def __init__(self) -> None:
        self.anxiety = 0.0
        self.overload = 0.0
        self.memory = 0.0
        self.cyber_defense = 5.0
        self.health_score = 100.0
        self._lock = threading.RLock()

    def absorb_overload(self, amount: float) -> None:
        with self._lock:
            self.overload = max(0.0, self.overload - float(amount))

    def add_overload(self, amount: float) -> None:
        with self._lock:
            self.overload = max(0.0, self.overload + float(amount))

    def boost_memory(self, amount: float) -> None:
        with self._lock:
            self.memory = max(0.0, self.memory + float(amount))

    def reduce_memory(self, amount: float) -> None:
        with self._lock:
            self.memory = max(0.0, self.memory - float(amount))

    def trigger_anxiety(self, amount: float) -> None:
        with self._lock:
            self.anxiety = max(0.0, self.anxiety + float(amount))

    def change_cyber_defense(self, amount: float) -> None:
        with self._lock:
            self.cyber_defense = max(0.0, self.cyber_defense + float(amount))

    def defend(self, attack_strength: float) -> bool:
        with self._lock:
            defended = self.cyber_defense >= float(attack_strength)

        if defended:
            logger.info("Defended against attack.")
        else:
            logger.warning("Attack penetrated defenses.")

        return defended

    def update_health(self, delta: float) -> None:
        with self._lock:
            self.health_score = max(0.0, min(100.0, self.health_score + float(delta)))

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "anxiety": self.anxiety,
                "overload": self.overload,
                "memory": self.memory,
                "cyber_defense": self.cyber_defense,
                "health_score": self.health_score,
            }


@dataclass
class MicrobiomeSpecies:
    """A species that changes gut-brain system state during a phase."""

    name: str
    role: str
    effect: SpeciesEffect
    is_bad: bool = False
    abundance: int = 1

    def __post_init__(self) -> None:
        self.abundance = max(1, int(self.abundance))

    def act(self, system: MicrobiomeSystemState) -> None:
        for _ in range(self.abundance):
            self.effect(system)


@register_microbiome_species
class Lactobacillus(MicrobiomeSpecies):
    def __init__(self, abundance: int = 2) -> None:
        super().__init__(
            name="Lactobacillus",
            role="overload_reducer",
            effect=lambda system: system.absorb_overload(2),
            abundance=abundance,
        )


@register_microbiome_species
class Bifidobacterium(MicrobiomeSpecies):
    def __init__(self, abundance: int = 2) -> None:
        super().__init__(
            name="Bifidobacterium",
            role="memory_helper",
            effect=lambda system: system.boost_memory(1),
            abundance=abundance,
        )


@register_microbiome_species
class Pathogenus(MicrobiomeSpecies):
    def __init__(self, abundance: int = 1) -> None:
        super().__init__(
            name="Pathogenus",
            role="anxiety_trigger",
            effect=lambda system: system.trigger_anxiety(3),
            is_bad=True,
            abundance=abundance,
        )


class NeuronLoop:
    """Thread-safe simulated neuron with activity and memory metrics."""

    def __init__(self) -> None:
        self.activity = 0.0
        self.memory = 0.0
        self._lock = threading.RLock()

    def stimulate(self, amount: float) -> None:
        with self._lock:
            amount = float(amount)
            self.activity += amount
            self.memory += amount * 0.1

    def rest(self) -> None:
        with self._lock:
            self.activity = max(0.0, self.activity - 1.0)
            self.memory = max(0.0, self.memory - 0.05)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {"activity": self.activity, "memory": self.memory}


class VagusNerve:
    """Copies a consistent microbiome snapshot to cognitive system state."""

    def __init__(
        self,
        gut_state: MicrobiomeSystemState,
        cognitive_state: SystemState,
    ) -> None:
        self.gut_state = gut_state
        self.cognitive_state = cognitive_state

    def transmit_signals(self) -> Dict[str, float]:
        signals = self.gut_state.snapshot()

        self.cognitive_state.update("anxiety_level", signals["anxiety"])
        self.cognitive_state.update("overload_level", signals["overload"])
        self.cognitive_state.update("memory_score", signals["memory"])
        self.cognitive_state.update("cyber_defense", signals["cyber_defense"])
        self.cognitive_state.update("microbiome_health", signals["health_score"])

        logger.info("[VagusNerve] Signals transmitted from gut to brain.")
        return signals


@register_cognitive_engine("threedimensionalhro")
class ThreeDimensionalHRO:
    """A configurable reasoning, optimization, and gut-brain simulation engine."""

    VALID_REASONING_MODES = {"sequential", "neural", "hybrid"}
    VALID_OPTIMIZATION_STRATEGIES = {"simple", "complex", "adaptive"}

    def __init__(
        self,
        reasoning_mode: str = "sequential",
        compute_backend: str = "local",
        optimization_strategy: str = "simple",
        rcd: Optional[Any] = None,
        monitors: Optional[List[Monitor]] = None,
        species_capacity: int = 10,
        random_seed: Optional[int] = None,
        neuron_count: int = 10,
    ) -> None:
        if species_capacity < 1:
            raise ValueError("species_capacity must be at least 1.")
        if neuron_count < 1:
            raise ValueError("neuron_count must be at least 1.")

        self.x_axis = reasoning_mode.lower()
        self.y_axis = compute_backend
        self.z_axis = optimization_strategy.lower()

        if self.x_axis not in self.VALID_REASONING_MODES:
            raise ValueError(
                f"Unsupported reasoning mode '{reasoning_mode}'. "
                f"Expected one of {sorted(self.VALID_REASONING_MODES)}."
            )
        if self.z_axis not in self.VALID_OPTIMIZATION_STRATEGIES:
            raise ValueError(
                f"Unsupported optimization strategy '{optimization_strategy}'. "
                f"Expected one of {sorted(self.VALID_OPTIMIZATION_STRATEGIES)}."
            )

        self.state = SystemState()
        self.rcd = rcd
        self.monitors = list(monitors or [])
        self.neural_available = _TORCH_AVAILABLE
        self.reasoning_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self.optimization_history: List[Dict[str, Any]] = []
        self._opt_lock = threading.RLock()
        self._rng = random.Random(random_seed)

        self.microbiome_state = MicrobiomeSystemState()
        self.neurons = [NeuronLoop() for _ in range(neuron_count)]
        self._last_total_neuron_memory = 0.0
        self.species_capacity = species_capacity
        self.microbiome = self._init_microbiome_species()
        self.vagus_nerve = VagusNerve(self.microbiome_state, self.state)

        logger.info(
            "3NGIN3 initialized at (%s, %s, %s), neural=%s",
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.neural_available,
        )

    def _init_microbiome_species(self) -> List[MicrobiomeSpecies]:
        species_list: List[MicrobiomeSpecies] = [
            get_microbiome_species("lactobacillus")(abundance=2),
            get_microbiome_species("bifidobacterium")(abundance=2),
            get_microbiome_species("pathogenus")(abundance=1),
            MicrobiomeSpecies(
                name="DopamineActivator",
                role="activity_boost",
                effect=lambda _system: [
                    neuron.stimulate(2) for neuron in self.neurons
                ],
            ),
            MicrobiomeSpecies(
                name="Akkermansia",
                role="barrier_strengthener",
                effect=lambda system: system.change_cyber_defense(1),
            ),
            MicrobiomeSpecies(
                name="Faecalibacterium",
                role="anti_inflammatory",
                effect=lambda system: system.absorb_overload(1),
            ),
            MicrobiomeSpecies(
                name="Clostridium_difficile",
                role="memory_disruptor",
                effect=lambda system: system.reduce_memory(2),
                is_bad=True,
            ),
            MicrobiomeSpecies(
                name="E_coli_pathogenic",
                role="defense_weakener",
                effect=lambda system: system.change_cyber_defense(-1),
                is_bad=True,
            ),
        ]

        if len(species_list) > self.species_capacity:
            species_list = self._rng.sample(species_list, self.species_capacity)

        return species_list

    def update_population_dynamics(self) -> None:
        beneficial = [species for species in self.microbiome if not species.is_bad]
        pathogenic = [species for species in self.microbiome if species.is_bad]

        if len(pathogenic) > len(beneficial):
            for species in beneficial:
                species.abundance = max(1, species.abundance - 1)
            self.microbiome_state.update_health(-10)
        elif len(beneficial) > len(pathogenic):
            for species in pathogenic:
                species.abundance = max(1, species.abundance - 1)
            self.microbiome_state.update_health(5)

        beneficial_names = {species.name for species in beneficial}
        if {"Akkermansia", "Faecalibacterium"} <= beneficial_names:
            for species in beneficial:
                if species.name in {"Akkermansia", "Faecalibacterium"}:
                    species.abundance += 1

    def integrate_training_data(self, dataset: Mapping[str, int]) -> None:
        """Update species abundance using validated external values."""
        for species in self.microbiome:
            if species.name not in dataset:
                continue

            abundance = int(dataset[species.name])
            species.abundance = max(1, abundance)

    def check_microbiome_safety(self) -> List[str]:
        """Apply simple interventions and return events that occurred."""
        events: List[str] = []
        health = self.microbiome_state.snapshot()["health_score"]

        if health < 50:
            for species in self.microbiome:
                if not species.is_bad:
                    species.abundance += 1
            self.microbiome_state.update_health(20)
            events.append("probiotic_intervention")

        pathogenic_load = sum(
            species.abundance for species in self.microbiome if species.is_bad
        )
        if pathogenic_load > 5:
            for species in self.microbiome:
                if species.is_bad:
                    species.abundance = max(1, species.abundance - 1)
            self.microbiome_state.update_health(10)
            events.append("immune_response")

        return events

    def simulate_microbiome_phase(
        self,
        phase: str,
        dataset: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """Run one busy, rest, or idle gut-brain simulation phase."""
        normalized_phase = phase.lower()
        if normalized_phase not in {"busy", "rest", "idle"}:
            raise ValueError("phase must be one of: 'busy', 'rest', or 'idle'.")

        if dataset is not None:
            self.integrate_training_data(dataset)

        self.update_population_dynamics()
        safety_events = self.check_microbiome_safety()

        for species in self.microbiome:
            species.act(self.microbiome_state)

        for neuron in self.neurons:
            if normalized_phase == "busy":
                neuron.stimulate(self._rng.randint(1, 3))
            elif normalized_phase == "rest":
                neuron.rest()

        neuron_snapshots = [neuron.snapshot() for neuron in self.neurons]
        total_neuron_memory = sum(item["memory"] for item in neuron_snapshots)
        memory_delta = total_neuron_memory - self._last_total_neuron_memory
        self.microbiome_state.boost_memory(memory_delta)
        self._last_total_neuron_memory = total_neuron_memory

        gut_state = self.vagus_nerve.transmit_signals()
        result = {
            "phase": normalized_phase,
            "safety_events": safety_events,
            "gut_state": gut_state,
            "cognitive_state": self.state.as_dict(),
            "microbiome": [
                {
                    "name": species.name,
                    "role": species.role,
                    "is_bad": species.is_bad,
                    "abundance": species.abundance,
                }
                for species in self.microbiome
            ],
            "neurons": {
                "count": len(self.neurons),
                "total_activity": sum(item["activity"] for item in neuron_snapshots),
                "total_memory": total_neuron_memory,
            },
        }

        logger.info(
            "Completed microbiome phase '%s': health=%.1f, anxiety=%.1f",
            normalized_phase,
            gut_state["health_score"],
            gut_state["anxiety"],
        )
        return result

    def run_diagnostics(self) -> Dict[str, Any]:
        """Return a serializable snapshot of engine health and neuron metrics."""
        neuron_snapshots = [neuron.snapshot() for neuron in self.neurons]
        diagnostics = {
            "gut_state": self.microbiome_state.snapshot(),
            "cognitive_state": self.state.as_dict(),
            "neurons": {
                "count": len(neuron_snapshots),
                "total_activity": sum(item["activity"] for item in neuron_snapshots),
                "total_memory": sum(item["memory"] for item in neuron_snapshots),
            },
            "optimization_runs": len(self.optimization_history),
        }
        logger.info("Diagnostics generated.")
        return diagnostics

    def think(self, content: str, **kwargs: Any) -> Dict[str, Any]:
        """Perform sequential, neural, or hybrid reasoning."""
        if not isinstance(content, str):
            raise TypeError("content must be a string.")

        if self.x_axis == "sequential":
            return self._sequential_reasoning(content, **kwargs)

        if self.x_axis == "neural":
            if not self.neural_available:
                logger.warning(
                    "Neural mode requested but torch is unavailable; "
                    "falling back to sequential reasoning."
                )
                return self._sequential_reasoning(content, **kwargs)
            return self._neural_reasoning(content, **kwargs)

        sequential_result = self._sequential_reasoning(content, **kwargs)
        if not self.neural_available:
            return sequential_result

        neural_result = self._neural_reasoning(content, **kwargs)
        return self._hybrid_fusion(sequential_result, neural_result, **kwargs)

    def _normalize_outcome(
        self,
        outcome: Any,
        start_time: float,
    ) -> Dict[str, Any]:
        result = dict(outcome) if isinstance(outcome, dict) else {"content": str(outcome)}
        result.setdefault("content", "")
        result["content"] = str(result["content"])
        result["confidence"] = max(
            0.0,
            min(1.0, float(result.get("confidence", 0.5))),
        )
        result["runtime"] = time.perf_counter() - start_time
        return result

    def _sequential_reasoning(
        self,
        content: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        cache_key = f"sequential:{content}"

        with self._cache_lock:
            cached = self.reasoning_cache.get(cache_key)
        if cached is not None:
            cached_result = dict(cached)
            cached_result["cached"] = True
            return self._normalize_outcome(cached_result, start_time)

        steps = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", content)
            if sentence.strip()
        ]

        reasoning_steps: List[Dict[str, Any]] = []
        for index, step in enumerate(steps, start=1):
            confidence = 0.80 + self._rng.random() * 0.20
            reasoning_steps.append(
                {
                    "step": index,
                    "input": step,
                    "logical_analysis": f"Analyzing: {step[:80]}",
                    "confidence": confidence,
                }
            )

        confidence = (
            sum(step["confidence"] for step in reasoning_steps) / len(reasoning_steps)
            if reasoning_steps
            else 0.5
        )
        result = {
            "mode": "sequential",
            "reasoning_steps": reasoning_steps,
            "conclusion": f"Sequential analysis of {len(reasoning_steps)} logical steps.",
            "confidence": confidence,
            "cached": False,
        }

        with self._cache_lock:
            self.reasoning_cache[cache_key] = dict(result)

        return self._normalize_outcome(result, start_time)

    def _neural_reasoning(
        self,
        content: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if not self.neural_available:
            return self._normalize_outcome(
                {
                    "mode": "neural_sim",
                    "embedding_dimension": 0,
                    "attention_weights": [],
                    "context_strength": 0.0,
                    "pattern_matches": 0,
                    "confidence": 0.5,
                },
                start_time,
            )

        words = content.split()
        if not words:
            return self._normalize_outcome(
                {
                    "mode": "neural",
                    "embedding_dimension": 0,
                    "attention_weights": [],
                    "context_strength": 0.0,
                    "pattern_matches": 0,
                    "confidence": 0.5,
                },
                start_time,
            )

        embedding_dim = int(kwargs.get("embedding_dim", 64))
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")

        sequence_length = min(128, len(words))
        seed = kwargs.get("seed")
        generator = None

        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))

        embeddings = torch.randn(
            sequence_length,
            embedding_dim,
            generator=generator,
        )
        attention_logits = torch.randn(sequence_length, generator=generator)
        attention_weights = torch.softmax(attention_logits, dim=0)
        context_vector = torch.sum(
            embeddings * attention_weights.unsqueeze(1),
            dim=0,
        )

        max_attention = float(torch.max(attention_weights).item())
        pattern_matches = int(
            torch.sum(attention_weights > (1.0 / sequence_length)).item()
        )
        result = {
            "mode": "neural",
            "embedding_dimension": embedding_dim,
            "attention_weights": attention_weights.tolist(),
            "context_strength": float(torch.norm(context_vector).item()),
            "pattern_matches": pattern_matches,
            "confidence": max_attention,
            "seed": seed,
        }
        return self._normalize_outcome(result, start_time)

    def _hybrid_fusion(
        self,
        sequential_result: Dict[str, Any],
        neural_result: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        neural_weight = float(kwargs.get("neural_weight", 0.6))

        if not 0.0 <= neural_weight <= 1.0:
            raise ValueError("neural_weight must be between 0.0 and 1.0.")

        sequential_confidence = float(sequential_result.get("confidence", 0.5))
        neural_confidence = float(neural_result.get("confidence", 0.5))
        combined_confidence = (
            sequential_confidence * (1.0 - neural_weight)
            + neural_confidence * neural_weight
        )

        return self._normalize_outcome(
            {
                "mode": "hybrid",
                "sequential_component": sequential_result,
                "neural_component": neural_result,
                "fusion_weight": neural_weight,
                "combined_confidence": combined_confidence,
                "synthesis": (
                    "Hybrid reasoning combining "
                    f"{len(sequential_result.get('reasoning_steps', []))} logical steps "
                    f"with {neural_result.get('pattern_matches', 0)} pattern matches."
                ),
                "confidence": combined_confidence,
            },
            start_time,
        )

    def safe_think(
        self,
        agent_name: str,
        content: str,
        *,
        resource_budget: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run thought through an optional safety monitor and local monitors."""
        budget = (
            float(resource_budget)
            if resource_budget is not None
            else float(self.state.get("default_resource_budget", 0.5))
        )
        intent = {
            "agent": agent_name,
            "action": "think",
            "content_summary": content[:200],
            "resource_budget": budget,
        }

        if self.rcd is not None:
            try:
                result = self.rcd.monitor(intent, self.think, content, **kwargs)
                return self._normalize_outcome(result, time.perf_counter())
            except Exception as error:
                logger.exception(
                    "RCD tripped during thought by %s: %s",
                    agent_name,
                    error,
                )
                return {
                    "error": "Cognitive fault detected",
                    "details": str(error),
                    "confidence": 0.0,
                }

        result = self.think(content, **kwargs)
        for monitor in self.monitors:
            try:
                monitor(self, content, result)
            except Exception:
                logger.exception("Monitor raised an exception.")

        return result

    def optimize(
        self,
        problem_space: Mapping[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Optimize according to the configured strategy."""
        if self.z_axis == "simple":
            return self._simple_optimization(problem_space, **kwargs)
        if self.z_axis == "complex":
            return self._complex_optimization(problem_space, **kwargs)
        return self._adaptive_optimization(problem_space, **kwargs)

    def _simple_optimization(
        self,
        problem_space: Mapping[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        iterations = max(1, int(kwargs.get("iterations", 50)))
        dimensions = max(1, int(problem_space.get("dimensions", 3)))
        objective = problem_space.get("objective")

        if objective is not None and not callable(objective):
            raise TypeError("problem_space['objective'] must be callable.")

        best_score = float("-inf")
        best_solution: Optional[Dict[str, Any]] = None

        for iteration in range(iterations):
            parameters = [self._rng.random() for _ in range(dimensions)]
            score = (
                float(objective(parameters))
                if objective is not None
                else sum(parameters)
            )

            if score > best_score:
                best_score = score
                best_solution = {
                    "parameters": parameters,
                    "iteration": iteration,
                }

        result = {
            "strategy": "simple",
            "iterations": iterations,
            "best_score": best_score,
            "best_solution": best_solution,
        }
        self._record_optimization(result)
        return result

    def _complex_optimization(
        self,
        problem_space: Mapping[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        dimensions = max(1, int(problem_space.get("dimensions", 5)))
        temperature = float(kwargs.get("initial_temperature", 100.0))
        cooling_rate = float(kwargs.get("cooling_rate", 0.95))
        min_temperature = float(kwargs.get("min_temperature", 0.01))
        max_iterations = max(1, int(kwargs.get("max_iterations", 10_000)))

        if temperature <= 0:
            raise ValueError("initial_temperature must be greater than 0.")
        if not 0.0 < cooling_rate < 1.0:
            raise ValueError("cooling_rate must be greater than 0 and less than 1.")
        if min_temperature <= 0:
            raise ValueError("min_temperature must be greater than 0.")

        current = [self._rng.randint(0, 1) for _ in range(dimensions)]
        current_energy = self._qubo_energy(current, problem_space)
        best = current.copy()
        best_energy = current_energy

        iteration = 0
        while temperature > min_temperature and iteration < max_iterations:
            neighbor = current.copy()
            flip_index = self._rng.randrange(dimensions)
            neighbor[flip_index] = 1 - neighbor[flip_index]

            neighbor_energy = self._qubo_energy(neighbor, problem_space)
            accepted = (
                neighbor_energy < current_energy
                or self._rng.random()
                < self._acceptance_probability(
                    current_energy,
                    neighbor_energy,
                    temperature,
                )
            )

            if accepted:
                current = neighbor
                current_energy = neighbor_energy

                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy

            temperature *= cooling_rate
            iteration += 1

        result = {
            "strategy": "complex",
            "algorithm": "simulated_annealing",
            "iterations": iteration,
            "best_solution": {
                "parameters": best,
                "energy": best_energy,
            },
            "final_temperature": temperature,
        }
        self._record_optimization(result)
        return result

    def _adaptive_optimization(
        self,
        problem_space: Mapping[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        complexity = str(problem_space.get("complexity", "medium")).lower()
        dimensions = int(problem_space.get("dimensions", 3))

        if complexity == "low" or dimensions < 5:
            return self._simple_optimization(problem_space, **kwargs)

        return self._complex_optimization(problem_space, **kwargs)

    @staticmethod
    def _qubo_energy(
        solution: List[int],
        problem_space: Mapping[str, Any],
    ) -> float:
        """Calculate deterministic QUBO energy from linear and pairwise weights."""
        weights = problem_space.get("weights", {})
        linear_weights = problem_space.get("linear_weights", {})

        if not isinstance(weights, Mapping):
            raise TypeError("problem_space['weights'] must be a mapping.")
        if not isinstance(linear_weights, Mapping):
            raise TypeError("problem_space['linear_weights'] must be a mapping.")

        energy = 0.0

        for index, value in enumerate(solution):
            energy += float(linear_weights.get(str(index), 0.0)) * value

        for left_index in range(len(solution)):
            for right_index in range(left_index + 1, len(solution)):
                weight = float(weights.get(f"{left_index}_{right_index}", 0.0))
                energy += weight * solution[left_index] * solution[right_index]

        return energy

    @staticmethod
    def _acceptance_probability(
        current_energy: float,
        new_energy: float,
        temperature: float,
    ) -> float:
        """Return the simulated-annealing probability of accepting a solution."""
        if new_energy <= current_energy:
            return 1.0
        if temperature <= 0:
            return 0.0

        exponent = (current_energy - new_energy) / temperature
        try:
            return min(1.0, math.exp(exponent))
        except OverflowError:
            return 0.0

    def _record_optimization(self, result: Dict[str, Any]) -> None:
        with self._opt_lock:
            self.optimization_history.append(dict(result))

    def get_status(self) -> Dict[str, Any]:
        """Get a serializable summary of current engine state."""
        return {
            "position": {
                "reasoning_mode": self.x_axis,
                "compute_backend": self.y_axis,
                "optimization_strategy": self.z_axis,
            },
            "capabilities": {
                "neural_available": self.neural_available,
                "thread_safe": True,
                "safety_monitoring": self.rcd is not None,
            },
            "state": self.state.as_dict(),
            "microbiome_health": self.microbiome_state.snapshot()["health_score"],
            "microbiome_state": self.microbiome_state.snapshot(),
            "optimization_runs": len(self.optimization_history),
            "rcd_status": (
                {
                    "constraints_active": True,
                    "configured": True,
                }
                if self.rcd is not None
                else None
            ),
        }

    def move_to_coordinates(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
    ) -> Dict[str, str]:
        """Move the engine to new reasoning, backend, and optimization coordinates."""
        if x is not None:
            normalized_x = x.lower()
            if normalized_x not in self.VALID_REASONING_MODES:
                raise ValueError(
                    f"Unsupported reasoning mode '{x}'. "
                    f"Expected one of {sorted(self.VALID_REASONING_MODES)}."
                )
            self.x_axis = normalized_x

        if y is not None:
            self.y_axis = y

        if z is not None:
            normalized_z = z.lower()
            if normalized_z not in self.VALID_OPTIMIZATION_STRATEGIES:
                raise ValueError(
                    f"Unsupported optimization strategy '{z}'. "
                    f"Expected one of {sorted(self.VALID_OPTIMIZATION_STRATEGIES)}."
                )
            self.z_axis = normalized_z

        logger.info(
            "Engine moved to (%s, %s, %s)",
            self.x_axis,
            self.y_axis,
            self.z_axis,
        )
        return {
            "reasoning_mode": self.x_axis,
            "compute_backend": self.y_axis,
            "optimization_strategy": self.z_axis,
        }


def create_default_cognitive_system(**kwargs: Any) -> ThreeDimensionalHRO:
    """Create the default registered 3NGIN3 cognitive engine."""
    return build_cognitive_engine("threedimensionalhro", **kwargs)


def list_registered_components() -> Dict[str, List[str]]:
    """List all currently registered component names."""
    return {
        "image_encoders": sorted(image_encoders.keys()),
        "cognitive_engines": sorted(cognitive_engines.keys()),
        "microbiome_species": sorted(microbiome_species_registry.keys()),
    }


@register_image_encoder(name="custom_vision")
def custom_vision_encoder(
    config_encoder: Mapping[str, Any],
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a placeholder custom-vision encoder configuration."""
    if verbose:
        logger.info("Creating custom vision encoder with config: %s", config_encoder)

    return {
        "type": "custom_vision",
        "config": dict(config_encoder),
        "options": kwargs,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Available components:", list_registered_components())

    engine = create_default_cognitive_system(
        reasoning_mode="hybrid",
        optimization_strategy="adaptive",
        random_seed=42,
    )

    microbiome_result = engine.simulate_microbiome_phase("busy")
    print("Microbiome result:", microbiome_result)

    thinking_result = engine.safe_think(
        "test_agent",
        "This is a test of the cognitive system. It has multiple steps.",
        seed=42,
    )
    print("Thinking result:", thinking_result)

    optimization_result = engine.optimize(
        {
            "dimensions": 5,
            "complexity": "high",
            "weights": {
                "0_1": -1.0,
                "1_2": 0.5,
                "2_3": -0.75,
                "3_4": 0.25,
            },
        },
        initial_temperature=20.0,
    )
    print("Optimization result:", optimization_result)

    print("Diagnostics:", engine.run_diagnostics())
