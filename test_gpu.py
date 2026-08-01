import torch
import time
import pandas as pd
import os

print("=" * 60)
print("SSCon Pre-Flight Sanity Check")
print("=" * 60)

# --- 1. PyTorch MPS Backend Check ---
print("\n[1/3] PyTorch MPS Backend Check")
print(f"  PyTorch version: {torch.__version__}")
print(f"  MPS built:       {torch.backends.mps.is_built()}")
print(f"  MPS available:   {torch.backends.mps.is_available()}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"  Selected device: {device}")

# --- 2. GPU Compute Benchmark ---
print("\n[2/3] GPU vs CPU Matmul Benchmark")
size = (5000, 5000)  # Reduced from 20000 for faster sanity check

tensor_cpu = torch.randn(size)

start_time_cpu = time.time()
result_cpu = torch.matmul(tensor_cpu, tensor_cpu)
end_time_cpu = time.time()
elapsed_time_cpu = end_time_cpu - start_time_cpu
print(f"  CPU matmul ({size[0]}x{size[1]}): {elapsed_time_cpu:.4f}s")

if device.type == "mps":
    tensor_gpu = torch.randn(size, device=device)
    # Warm-up pass
    _ = torch.matmul(tensor_gpu, tensor_gpu)
    torch.mps.synchronize()

    start_time_gpu = time.time()
    result_gpu = torch.matmul(tensor_gpu, tensor_gpu)
    torch.mps.synchronize()
    end_time_gpu = time.time()
    elapsed_time_gpu = end_time_gpu - start_time_gpu
    print(f"  MPS matmul ({size[0]}x{size[1]}): {elapsed_time_gpu:.4f}s")
    print(f"  MPS speedup over CPU: {elapsed_time_cpu / elapsed_time_gpu:.2f}x")
else:
    print("  [SKIP] MPS not available, GPU benchmark skipped.")

# --- 3. Data Pickle Integrity Check ---
print("\n[3/3] Data Pickle Integrity Check")
data_files = {
    "labeled_training_data":   "data/labeled_training_data.pkl",
    "unlabeled_training_data": "data/unlabeled_training_data.pkl",
    "validation_data":         "data/validation_data.pkl",
    "test_data":               "data/test_data.pkl",
}

all_ok = True
for name, path in data_files.items():
    if not os.path.exists(path):
        print(f"  [FAIL] {name}: file not found at {path}")
        all_ok = False
        continue
    try:
        df = pd.read_pickle(path)
        cols = list(df.columns)
        print(f"  [OK]   {name}: {df.shape[0]} rows, {df.shape[1]} cols {cols}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        all_ok = False

# --- Summary ---
print("\n" + "=" * 60)
if all_ok and device.type == "mps":
    print("PRE-FLIGHT PASSED: MPS active, all data pickles valid.")
elif all_ok:
    print("PRE-FLIGHT PASSED (CPU-only): all data pickles valid.")
else:
    print("PRE-FLIGHT FAILED: see errors above.")
print("=" * 60)

