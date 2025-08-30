# 3NGIN3: The Three-Dimensional Cognitive Engine

**3NGIN3** is a prototype of a new cognitive architecture built on a three-dimensional, orthogonal design. It fuses a style-aware, mirrored agent system (`DuetMind`) with a robust, multi-axis reasoning and optimization engine.

The entire system is a self-contained, "single-cell" script designed for rapid research, prototyping, and the clear demonstration of advanced AI concepts. It's not just a model; it's an **executable architectural philosophy**.

At its core, **3NGIN3** is built on a powerful analogy: cognition as a programmable, modular **cognitive circuit**.

![3NGIN3 Architecture](https://i.imgur.com/your-diagram-image.png) <!-- It's highly recommended to create a visual for this -->

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
*   **The Dusty Mirror:** To build resilience and prevent cognitive overfitting, the mirroring process can inject a small amount of noise, forcing the agents to learn more robust and generalizable strategies.

## Core Features

*   **Single-File Prototype:** The entire architecture is in one Python script for maximum clarity, portability, and auditability.
*   **Graceful Degradation:** **3NGIN3** is fully functional without optional libraries like `torch`. When detected, it seamlessly enables enhanced neural capabilities.
*   **Active Safety Governance:** Includes a **Cognitive RCD (Residual Current Device)**, a built-in safety governor that monitors the balance between an agent's *intent* and its *outcome*, tripping the circuit to halt dangerous or runaway processes.
*   **Thread-Safe Unified State:** Features a robust, thread-safe `SystemState` with atomic updates and automatic rollback on error, ready for concurrent operations.

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
python 3ngin3.py  # or your script's filename
```

The demonstration will:
1.  Initialize the **3NGIN3** core.
2.  Execute reasoning tasks by moving along the **X-Axis**.
3.  Run a complex optimization task leveraging the **Z-Axis**.
4.  Launch the **DuetMind** cognitive dialogue to solve a creative design problem.
5.  Report on the final state and capabilities of the engine.

## The Future of 3NGIN3

This prototype is the foundation for several critical research directions:

*   **The Meta-Controller:** Implementing a true learning layer (e.g., using Reinforcement Learning) to allow the engine to autonomously learn the optimal `(X, Y, Z)` configuration for any given task.
*   **Advanced Cognitive Modules:** Replacing the current simulation stubs with more sophisticated models, such as Graph Neural Networks (GNNs) or Liquid Neural Networks (LNNs), that align with the "cognitive circuit" philosophy.
*   **Real-World Backend Integration:** Connecting the Y-Axis to real distributed frameworks (like Ray or Dask) and quantum hardware (like D-Wave Leap).

## Contribution

**3NGIN3** is a research project exploring the frontiers of modular, safe, and interpretable AI. Contributions in the form of architectural critiques, feature ideas, and code are highly encouraged. Please open an issue to discuss any proposals.

---
*Architecting the future of adaptable intelligence.*
