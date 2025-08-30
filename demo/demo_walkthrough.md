# Demo Walkthrough

## Preparation

- Ensure Python 3.8+ is installed.
- (Optional) Install `torch` for neural capabilities.

## Running the Demo

```bash
python 3ngin3.py
```

## Demo Steps

1. **Engine Initialization:**  
   - Logs starting coordinates (X, Y, Z).
2. **X-Axis Reasoning:**  
   - Demonstrates sequential, neural, and hybrid modes.
3. **Z-Axis Optimization:**  
   - Demonstrates simple, complex, and adaptive strategies.
4. **DuetMind Dialogue:**  
   - Agents collaborate on a creative design challenge.
5. **Safety Mechanisms:**  
   - Resource and ethical constraint tests.
6. **Y-Axis Backend (Optional):**  
   - Modify the demo to cycle through local, distributed, quantum for backend versatility.

## Output Interpretation

- See [demo_output.md](../demo_output.md) for example logs and analysis.

## Customizing the Demo

- To explicitly test Y-Axis, update `move_to_coordinates(y='distributed')` and `move_to_coordinates(y='quantum')` in the demo file.
