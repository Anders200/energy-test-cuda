# Energy test benchmark for CUDA

## Build & Run Instructions

### Prerequisites

* **CUDA Toolkit**
  Install from: [https://developer.nvidia.com/cuda/toolkit](https://developer.nvidia.com/cuda/toolkit)
* **CMake** (≥ 3.18 recommended)
* **Python 3.11+**

---

### Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Build the project

```bash
cmake -B build -S .
cmake --build build
```
If you are using a virtual environment then build with
```bash
cmake -B build -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build build
```
---

### Run

```bash
python3 run/run.py
```

