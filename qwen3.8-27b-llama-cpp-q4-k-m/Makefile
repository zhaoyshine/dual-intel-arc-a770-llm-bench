SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help build bench start devices driver-status driver-xe driver-i915 \
        tc-start tc-bench coder-start coder-bench \
        all vulkan sycl sycl-tensor sycl-mtp sycl-tensor-mtp layer tensor

help:
	@echo "make build [后端]        编译; 后端: sycl (默认) vulkan"
	@echo "make bench [配置]        跑基准; 不带参数跑全部"
	@echo "                         配置: vulkan sycl sycl-tensor sycl-mtp sycl-tensor-mtp"
	@echo "make start               启动 llama-server; 思考档位写死 low"
	@echo "make tc-start            启动 ThinkingCap-Qwen3.8-27B (内置 MTP); 档位写死 medium"
	@echo "make tc-bench [配置]     跑 ThinkingCap 基准; 配置: layer tensor (默认全部)"
	@echo "make coder-start         启动 Coder390-Qwen3.8-27B (内置 Q4 MTP); 档位写死 high"
	@echo "make coder-bench [配置]  跑 Coder390 基准; 配置: layer tensor (默认全部)"
	@echo "make devices             列出 llama.cpp 可见设备"
	@echo "make driver-status       查看 A770 当前内核驱动"
	@echo "make driver-xe           切到 xe 驱动 (重启生效)"
	@echo "make driver-i915         切回 i915 驱动 (重启生效)"

build:
	./build.sh $(or $(filter-out $@,$(MAKECMDGOALS)),sycl)

bench:
	./bench.sh $(filter-out $@,$(MAKECMDGOALS))

start:
	./start.sh

tc-start:
	./tc-start.sh

tc-bench:
	./tc-bench.sh $(filter-out $@,$(MAKECMDGOALS))

coder-start:
	./coder-start.sh

coder-bench:
	./coder-bench.sh $(filter-out $@,$(MAKECMDGOALS))

devices:
	./bench.sh devices

driver-status:
	./gpu_driver.sh status

driver-xe:
	./gpu_driver.sh xe

driver-i915:
	./gpu_driver.sh i915

# make bench sycl 等的位置参数在此吞掉
all vulkan sycl sycl-tensor sycl-mtp sycl-tensor-mtp layer tensor:
	@:
