家人们，用你发财的小手点一点右上角的星星⭐吧

# 双 Intel Arc A770 LLM 基准测试

双 Intel Arc A770 16GB 运行大模型。

每个文件夹对应一个「模型 + 启动器」配置，点击 readme 查看完整报告。

[English version](README.md)

| 模型 | 启动器 | 量化 | 报告 |
|---|---|---|---|
| Qwen3.8 27B | llama.cpp（SYCL 层切分 / MTP 投机解码） | Q4_K_M | [中](qwen3.8-27b-llama-cpp-q4-k-m/README_zh.md) / [en](qwen3.8-27b-llama-cpp-q4-k-m/README.md) |
| Qwen3.8 27B | vLLM（XPU 后端，自编译） | AWQ-INT4 | [中](qwen3.8-27b-vllm-awq/README_zh.md) / [en](qwen3.8-27b-vllm-awq/README.md) |
