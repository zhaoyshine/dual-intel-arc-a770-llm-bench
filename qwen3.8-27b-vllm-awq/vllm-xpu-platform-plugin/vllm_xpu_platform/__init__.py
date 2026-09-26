# vLLM 会激活每个探测成功的内置 platform 插件, 而 CUDA 探测只向 NVML 问设备数,
# 机器上任何 NVIDIA 卡 (这里 GT 1030) 都能让它与 XPU 同时激活, vLLM 随即报
# "Only one platform plugin can be activated" 退出; OOT platform 插件优先于内置探测, 故钉死在 XPU.


def xpu_platform_plugin() -> str | None:
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "vllm.platforms.xpu.XPUPlatform"
    return None
