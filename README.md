
# LLM Inference Load Simulator

This repository contains my solution for the technical assignment of the **Future Start Internship 2026**. The task was to simulate an LLM inference server behavior based on multi-modal request traces and optimize resource allocation.

## Repository Structure
```
├── batcher.py           
├── AzureLMMTrace.csv    
├── batcher/             
│   ├── CMakeLists.txt
│   ├── include/         (Request.h, Batch.h, Accelerator.h, ScheduleModel.h)
|   ├── data/            (AzureLMMTrace.csv)
│   └── src/             (main.cpp)
```

## Development Process
The project was developed in two stages to ensure both architectural correctness and runtime performance:

1.  **Python Prototype:** Initially, a discrete-event simulator was built in Python. This phase focused on defining the core object-oriented architecture (`Request`, `Batch`, `Accelerator`, `ScheduleModel`) and implementing the planning logic with batch scheduling (using `A * sqrt(B)` cost scaling) and state-machine transitions.
2.  **C++ Implementation:** To achieve high throughput and eliminate Python's execution overhead, the logic was migrated to C++. This transition allowed for finer control over memory allocation and significantly accelerated the simulation loop, proving essential for processing large-scale traces efficiently.

## Performance Metrics
The simulation was conducted using the provided Azure LMM trace dataset. 
**Configuration parameters:** 
`N=10`, `M=317000.0`, `K=1000.0`, `A=100`, `B=100`, `C=100`, `X=Y=Z=2`.

### Execution Statistics:

#### TTFT (Time To First Token) [ms]
*   **Median:** 0.00
*   **Average:** 49.13
*   **Min:** 0.00
*   **Max:** 2,314,900.00

#### T (Time per Token) [ms/token]
*   **Median:** 1.88
*   **Average:** 11.88
*   **Min:** 0.00
*   **Max:** 1,873.66

## How to Build and Run
1. Ensure you have `CMake` and a C++ compiler installed.
2. Build the project:
   ```bash
   cd batcher
   mkdir build && cd build
   cmake ..
   ```
3. Execution: 
   - Open the generated `.sln` file in Visual Studio.
   - **Important:** Set the "Working Directory" to `$(SolutionDir)..\` in project properties to ensure the simulator correctly locates `test.csv`.
   - Build and run the `inference_simulator` target.
```
