--

# Installation Guide: Clone Detection Research Environment

This repository contains the environment setup and benchmarking tools for evaluating Pre-trained Models (PTMs) on the **BigCloneBench** dataset. This setup is optimized for high-performance deep learning on dual NVIDIA RTX 6000 Ada GPUs.

## 1. System Requirements

* **OS:** Ubuntu 22.04+ (or compatible Linux distribution)
* **Hardware:** 2x NVIDIA RTX 6000 Ada (96GB total VRAM)
* **Driver:** NVIDIA Driver version 550.142+
* **CUDA:** 12.4
* **Language:** Python 3.11

## 2. Environment Setup

We recommend using **Miniconda** to manage the research environment. This ensures that CUDA-specific libraries are correctly linked without interfering with system-level packages.

### Step 1: Create the Conda Environment

```bash
conda create -n bigclone python=3.11 -y
conda activate bigclone

```

### Step 2: Install PyTorch with CUDA 12.4 Support

Install the core deep learning framework directly via the `pytorch` and `nvidia` channels. This ensures that the internal `nvcc` tools and shared libraries are perfectly matched to your hardware.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

```

### Step 3: Resolve Intel MKL Library Conflicts (Critical)

**Note:** A known conflict exists between the latest Intel Math Kernel Library (MKL) 2025.x provided by the default Conda channel and the PyTorch binaries, which often expect MKL 2024.x. This mismatch causes an `undefined symbol: iJIT_NotifyEvent` error when importing torch.

To prevent this, explicitly downgrade the MKL library to the compatible 2024 version:

```bash
conda install "mkl<2025" -y

```

### Step 4: Install Research Dependencies

Use the provided `requirements.txt` to install the Transformers library, database connectors for BigCloneBench, and structural parsing tools.

**Important:** Ensure your `requirements.txt` **does not** contain `torch`, `torchvision`, or `torchaudio`, as installing these via pip will overwrite the optimized Conda binaries and re-introduce library conflicts.

```bash
pip install -r requirements.txt

```

## 3. Verifying the Setup

Run the following Python snippet to verify that both RTX 6000 GPUs are visible, the MKL libraries are linking correctly, and the environment can perform parallel operations:

```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # Test tensor allocation on each GPU
    x = torch.rand(1000, 1000).to(f'cuda:{i}')
    print(f"Successfully allocated tensor on GPU {i}")

```

---

## 4. Scientific and Outreach Context

This setup marks the launch of a **new research project** within our lab focused on the empirical robustness of Large Language Models. A primary goal of this initiative is the **active mentorship of undergraduate and graduate researchers** through the technical and experimental phases of AI systems research.

* **Research Focus:** We are establishing a new benchmarking framework using **BigCloneBench** to evaluate the semantic understanding capabilities of state-of-the-art PTMs (such as GraphCodeBERT and CodeT5).
* **Mentorship:** Students serve as core contributors to this new project, managing the high-performance AI infrastructure and troubleshooting complex system dependencies—such as the Intel MKL binary conflicts documented above.
* **Preparation:** By guiding students from the initial environment design to the execution of large-scale experiments, we are preparing the next generation of engineers to handle the computational and architectural challenges of modern software intelligence.

This infrastructure lays the groundwork for our team's upcoming contributions to the field of automated code analysis and clone detection.