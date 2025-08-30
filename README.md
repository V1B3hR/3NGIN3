# 3NGIN3
# DuetMind-3D: A Fused Cognitive Architecture

**DuetMind-3D** is a research prototype of a novel, self-contained cognitive architecture. It fuses a style-aware, mirrored conversational agent system (`DuetMind`) with a robust, multi-dimensional reasoning and optimization engine (`ThreeDimensionalHRO`).

The entire system is designed as a single-file, "single-cell" script to provide a clear, auditable, and easily modifiable platform for experimenting with advanced AI concepts. It's an executable architectural diagram.

![Architecture Diagram](https://i.imgur.com/your-diagram-image.png)  <!-- You can create a simple diagram for this -->

---

## The Core Philosophy: A Cognitive Circuit

This project moves beyond biological metaphors ("neural networks") and adopts a **computational metaphor**—thinking of cognition as a programmable, modular electronic circuit.

*   **Agents as Transistors:** The core `DuetMind` agents act as complementary `NPN/PNP` transistors, one processing information analytically, the other poetically.
*   **Dialogue as Amplification:** The interaction between agents is designed to create **emergent phenomena**, where the resulting insight is greater than the sum of its parts, like constructive interference in a signal.
*   **Safety as a Circuit Breaker:** The system includes a `CognitiveRCD` (Residual Current Device), a safety governor that monitors the balance between an agent's *intent* and its *outcome*, tripping the circuit to halt any dangerous or runaway processes.

## Key Architectural Features

The system is organized along three orthogonal axes, allowing for dynamic reconfiguration to tackle a wide range of tasks.

### The Three-Dimensional HRO Engine

1.  **X-Axis: Reasoning Modes**
    *   **Sequential:** Verifiable, symbolic logic.
    *   **Neural:** Pattern recognition and associative thinking (using PyTorch when available).
    *   **Hybrid:** A weighted combination of both modes.

2.  **Y-Axis: Compute Backends**
    *   **Local:** Standard, single-process execution.
    *   **Distributed (Simulated):** A stub for scaling tasks across multiple workers.
    *   **Quantum (Simulated):** A quantum-inspired backend using simulated annealing to solve complex optimization problems (QUBO).

3.  **Z-Axis: Optimization Strategies**
    *   **Simple:** Fast, random-search optimization.
    *   **Complex:** QUBO-based optimization leveraging the quantum backend.
    *   **Adaptive:** A meta-strategy that chooses between Simple and Complex based on the problem's nature.

### The DuetMind Agent Layer

*   **Style-Aware Agents:** Agents are not monolithic. They possess a **Style Vector** (`{"logic": 0.9, "creativity": 0.2, ...}`), allowing for nuanced, multi-dimensional cognitive approaches.
*   **Mirrored Self-Play:** Agents reason, then "mirror" and critique each other's outputs. This creates a feedback loop that stabilizes reasoning and promotes synthesis.
*   **The "Dusty Mirror":** To encourage robust, generalizable learning, the mirroring process can be configured with a `distortion_level`, injecting a small amount of noise to prevent cognitive overfitting.

## Core Features

*   **Single-File Prototype:** The entire architecture is contained in a single Python script for maximum clarity and portability.
*   **Graceful Degradation:** The system is fully functional without optional libraries like `torch`. If detected, it seamlessly enables enhanced neural capabilities.
*   **Thread-Safe State:** Features a unified, thread-safe `SystemState` with atomic updates and automatic rollback on error, making it robust for concurrent operations.
*   **Built-in Safety Governance:** The `CognitiveRCD` provides an active safety layer that prevents resource overuse and enforces ethical constraints.

## Getting Started

### Prerequisites

You need Python 3.8+. For the full experience including neural reasoning, install the optional dependencies.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/DuetMind-3D.git
    cd DuetMind-3D
    ```

2.  Install the required dependencies. For a minimal setup:
    ```bash
    pip install numpy
    ```
    For the full experience with PyTorch:
    ```bash
    pip install numpy torch
    ```
    (Or use the `requirements.txt` file)
    ```bash
    pip install -r requirements.txt
    ```

### Running the Demo

The script includes a self-contained demonstration that showcases the core functionalities of the system.

```bash
python main_script_name.py
```

The demo will:
1.  Initialize the 3D-HRO engine.
2.  Perform reasoning tasks using different modes (Sequential, Hybrid).
3.  Run an optimization task using the Adaptive strategy.
4.  Execute the full `DuetMind` cognitive dialogue to solve a complex design problem.
5.  Print a final status report of the system.

## Future Research & Development

This prototype serves as a launchpad for several exciting research directions:

*   **Closing the Learning Loop:** Implementing a true **Meta-Controller** (e.g., using Reinforcement Learning) to allow the system to learn the optimal `(reasoning, backend, strategy)` configuration for different tasks.
*   **Advanced Neural Modules:** Replacing the current stubs with more sophisticated models like Graph Neural Networks (GNNs) or Liquid Neural Networks (LNNs) that align better with the "cognitive circuit" philosophy.
*   **Real Backend Integration:** Connecting the Y-Axis stubs to real distributed computing frameworks (like Ray or Dask) and real quantum hardware (like D-Wave Leap).

## Contribution

This is a research project, and contributions in the form of ideas, architectural critiques, and code are highly welcome. Please open an issue to discuss any major changes or ideas.

---
*This project is dedicated to exploring the future of modular, safe, and interpretable AI architectures.*
