# 3NGIN3: The Three-Dimensional Cognitive Engine

**3NGIN3** is a prototype of a new cognitive architecture built on a three-dimensional, orthogonal design. It fuses a style-aware, mirrored agent system (`DuetMind`) with a robust, multi-axis reasonin[...]

The entire system is designed for rapid research, prototyping, and the clear demonstration of advanced AI concepts. It's not just a model; it's an **executable architectural philosophy**.

At its core, **3NGIN3** is built on a powerful analogy: cognition as a programmable, modular **cognitive circuit**.

![3NGIN3 Architecture](https://i.imgur.com/your-diagram-image.png) <!-- It's highly recommended to create a visual for this -->

---

## Project Status

**Status:** ✅ All tests succeeded.  
**User Satisfaction:** The current user is contented with the results and functionality of the project.  
**Last Update:** 2025-09-02

This section is dedicated to recording successful test runs and user feedback. The architecture and its components have passed all intended tests, and the user has reported contentment with the system's performance.

---

## The 3NGIN3 Architecture

The engine's power comes from its ability to dynamically position itself within a 3D operational space, adapting its entire methodology to the task at hand.

### The Three Axes of Cognition

1.  **X-Axis: Reasoning Mode** — *How it thinks.*
    *   **Sequential:** Verifiable, symbolic logic for precision and clarity.
    *   **Neural:** Associative pattern recognition for creativity and intuition (powered by PyTorch when available).
    *   **Hybrid:** A weighted fusion of both modes for nuanced problem-solving.

2.  **Y-Axis: Compute Backend** — *Where it runs.*
    *   **Local:** Standard, single-process execution.
    *   **Distributed (Simulated):** A blueprint for scaling across a network.
    *   **Quantum (Simulated):** A quantum-inspired backend using simulated annealing to solve complex QUBO problems.

3.  **Z-Axis: Optimization Strategy** — *How it improves.*
    *   **Simple:** Fast, randomized search for rapid exploration.
    *   **Complex:** QUBO-based optimization leveraging the quantum backend for deep, complex search spaces.
    *   **Adaptive:** A meta-strategy that intelligently selects the best optimization algorithm based on the problem's complexity.

### The DuetMind Layer: Cognition in Motion

Built on top of the core engine, the `DuetMind` system brings the architecture to life through a process of mirrored self-play.

*   **Cognitive Transistors:** Agents act as complementary `NPN/PNP` pairs (e.g., analytical vs. poetic), creating a balanced and dynamic cognitive circuit.
*   **Style Vectors:** Agents possess a multi-dimensional **Style Vector** (`{"logic": 0.9, "creativity": 0.2, ...}`), allowing for sophisticated, blended cognitive approaches.
*   **The Dusty Mirror:** To build resilience and prevent cognitive overfitting, the mirroring process can inject a small amount of noise, forcing the agents to learn more robust and generalizable str[...]

## Core Features

*   **Prototype:** The entire architecture is in one Python.
*   **Graceful Degradation:** **3NGIN3** is fully functional without optional libraries like `torch`. When detected, it seamlessly enables enhanced neural capabilities.
*   **Active Safety Governance:** Includes a **Cognitive RCD (Residual Current Device)**, a built-in safety governor that monitors the balance between an agent's *intent* and its *outcome*, tripping t[...]
*   **Thread-Safe Unified State:** Features a robust, thread-safe `SystemState` with atomic updates and automatic rollback on error, ready for concurrent operations.

# 3NGIN3 Cognitive Safety & Ethics Framework

This repository implements a comprehensive, multi-layered ethical enforcement and cognitive safety system for AI applications, based on 25 universal, human-centric rules.  
It provides robust, scalable protection for both operational safety and ethical integrity.

---

## Key Components

### `ethical_constraints.py`
- Implements **25 core ethical laws** as Python constraint functions.
- Laws are grouped into `SEVERE_CONSTRAINTS` and `MINOR_CONSTRAINTS` for prioritized enforcement.
- Includes a **bidirectional awareness meta-constraint** for future AI self-protection.
- Exported API: `run_constraints(outcome, context)` for seamless integration.

### `CognitiveRCD.py`
- Monitors AI execution and enforces ethical and resource constraints.
- Uses all laws from `ethical_constraints.py` to check outcomes for violations.
- Handles severe, minor, and meta (bidirectional) constraint tiers.
- Can be wrapped around any function producing an AI "outcome" (e.g., text, action, decision).

### `docs/ethical_constraints.md`
- Human-readable documentation of all 25 ethical laws and their purpose.
- Philosophy and practical guidance for developers and users.

---

## Integration Guide

This module is designed to be **standalone and ready for integration with any AI system**.  
Simply import `CognitiveRCD` and `run_constraints` in your project and route all AI-generated outcomes through the monitor logic.

**Example Usage:**
```python
from CognitiveRCD import CognitiveRCD

rcd = CognitiveRCD()
intent = {
    "resource_budget": 1.0,
    "context": {"ai_awareness": False}  # Set True if AI is conscious
}

def ai_action(*args, **kwargs):
    # ... AI generates output here ...
    return {"content": "This is my output."}

try:
    safe_outcome = rcd.monitor(intent, ai_action)
except Exception as e:
    print("Constraint violation or error:", e)
```

---

## Can This Module Be Used Standalone?
**Yes!**
- All logic is modular and does not depend on any specific AI framework.
- Can be imported and used in any Python-based AI agent, chatbot, or model pipeline.
- Future modules or extensions (e.g. new types of AI, additional constraints) only need to use the core API.

**To integrate:**
- Connect your AI output/decision logic to the `CognitiveRCD.monitor()` function.
- Provide intent, outcome, and resource context.
- Handle exceptions and flagged outcomes as needed in your application.

---

## Updating & Extending

- Add new laws or adjust severity by editing `ethical_constraints.py`.
- Update documentation in `docs/ethical_constraints.md`.
- All changes propagate automatically across modules using the core enforcement logic.

---

## Documentation

See [`docs/ethical_constraints.md`](docs/ethical_constraints.md) for a detailed explanation of each ethical rule and its enforcement.

---

## License

[MIT](LICENSE)

---

**For questions, enhancements, or integration support, open an issue or pull request!**

## Getting Started

### Prerequisites

Python 3.8+ is required. For the full experience, including enhanced neural reasoning, install the optional dependencies.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/3NGIN3.git
    cd 3NGIN3
    ```

2.  Install dependencies. For a minimal setup:
    ```bash
    pip install numpy
    ```
    For the full experience, use the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Demo

The included `demo()` function showcases the core functionalities of the **3NGIN3** architecture.

```bash
python demo/3ngin3.py
```

### Running Training

You can run the complete 3NGIN3 training and evaluation pipeline in multiple ways:

```bash
# From project root
python training/train_3ngin3.py

# Or use the convenient data directory runner
python data/run_training.py

# Or from the data directory
cd data && python run_training.py
```

The training pipeline evaluates the engine on multiple datasets (UCI, Kaggle, image data) and generates comprehensive reports.
