#!/bin/bash
# 不用 docker; 依赖 uv, python3.12, Intel 驱动 (内核模块 + level-zero), oneAPI

set -euo pipefail

VLLM_DIR=~/workspace/ai/vllm
VENV=$VLLM_DIR/.venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

err() { echo "错误: $*" >&2; exit 1; }

[[ -d "$VLLM_DIR" ]] || err "找不到 vLLM 源码 $VLLM_DIR (git clone https://github.com/vllm-project/vllm.git)"
[[ -d /opt/intel/oneapi ]] || err "找不到 oneAPI (/opt/intel/oneapi)"
command -v uv >/dev/null || err "找不到 uv"

# vllm-xpu-kernels 的 wheel 只针对 python3.12 构建
[[ -x "$VENV/bin/python" ]] || uv venv --python 3.12 "$VENV"
"$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' \
    || err "$VENV 不是 python3.12, 删掉重建"

echo "== 安装 XPU 依赖 (torch 2.14+xpu / triton-xpu / vllm-xpu-kernels) =="
# torch 与 wheels.vllm.ai 两个 extra index 都要, 且必须 unsafe-best-match
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
    -r "$VLLM_DIR/requirements/xpu.txt"

echo "== 固定 oneAPI DLE 包版本 (与 torch 2.14 的 xpu graph 支持配套) =="
uv pip install --python "$VENV/bin/python" \
    "intel-cmplr-lib-rt==2026.1.1" \
    "intel-cmplr-lib-ur==2026.1.1" \
    "intel-cmplr-lic-rt==2026.1.1" \
    "intel-sycl-rt==2026.1.1" \
    "oneccl-devel==2022.1.2; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "oneccl==2022.1.2; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "dpcpp-cpp-rt==2026.1.1" \
    "intel-opencl-rt==2026.1.1" \
    "intel-openmp==2026.1.1" \
    "intel-pti==1.1.0"

echo "== 编译安装 vLLM (XPU 目标) =="
cd "$VLLM_DIR"

git checkout -- .
git pull

VLLM_TARGET_DEVICE=xpu VLLM_WORKER_MULTIPROC_METHOD=spawn \
    uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps -e .

echo "== 安装两个 A770 适配插件 =="
uv pip install --python "$VENV/bin/python" --no-build-isolation -e "$SCRIPT_DIR/vllm-xpu-platform-plugin"
uv pip install --python "$VENV/bin/python" --no-build-isolation -e "$SCRIPT_DIR/vllm-gdn-triton-plugin"

echo "== 校验 =="
CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" - <<'EOF'
import torch
from vllm.model_executor.custom_op import op_registry_oot
from vllm.platforms import current_platform

print("torch", torch.__version__, "| XPU 数量:", torch.xpu.device_count())
print("platform", type(current_platform).__name__)
assert current_platform.device_type == "xpu"
assert "QwenGatedDeltaNetAttention" in op_registry_oot
EOF

echo "完成: $VENV (跑基准: ./bench.sh)"
