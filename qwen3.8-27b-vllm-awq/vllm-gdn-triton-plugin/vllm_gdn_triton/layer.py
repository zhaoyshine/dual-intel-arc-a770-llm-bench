# XPU 上 GDN 层硬编码走 forward_xpu, 即 vllm_xpu_kernels 的 SYCL chunk GDN 内核,
# 它在 Arc A770 (DG2) 上让 IGC 的 DPAS 代码生成断言崩溃; 故改走 forward_cuda (FLA triton 实现).
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)


@PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")
class QwenGatedDeltaNetAttentionTriton(QwenGatedDeltaNetAttention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._forward_method = self.forward_cuda
