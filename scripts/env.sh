# Source this before running any gsplat-backed script.
#   source scripts/env.sh
#
# This box has no system CUDA toolkit and no Python dev headers, so both are
# provided locally (installed during project setup):
#   - CUDA 12.8 toolkit  -> /home/chang/cuda
#   - Python 3.10 headers (extracted from .deb) -> /home/chang/pydev_root
# gsplat JIT-compiles its CUDA extension on first import and needs all three.
export CUDA_HOME=/home/chang/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH=/home/chang/pydev_root/usr/include:/home/chang/pydev_root/usr/include/python3.10
export TORCH_CUDA_ARCH_LIST="12.0"   # RTX 5060 = sm_120
