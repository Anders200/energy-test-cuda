
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

---

### Run

```bash
python3 run/run.py
```

