SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help build bench start devices driver-status driver-xe driver-i915 \
        all vulkan sycl sycl-tensor sycl-mtp sycl-tensor-mtp low medium high

help:
	@echo "make build [后端]        编译; 后端: sycl (默认) vulkan"
	@echo "make bench [配置]        跑基准; 不带参数跑全部"
	@echo "                         配置: vulkan sycl sycl-tensor sycl-mtp sycl-tensor-mtp"
	@echo "make start [档位]        启动 llama-server; 思考档位: low (默认) medium high"
	@echo "make devices             列出 llama.cpp 可见设备"
	@echo "make driver-status       查看 A770 当前内核驱动"
	@echo "make driver-xe           切到 xe 驱动 (重启生效)"
	@echo "make driver-i915         切回 i915 驱动 (重启生效)"

build:
	./build.sh $(or $(filter-out $@,$(MAKECMDGOALS)),sycl)

bench:
	./bench.sh $(filter-out $@,$(MAKECMDGOALS))

start:
	./start.sh $(filter-out $@,$(MAKECMDGOALS))

devices:
	./bench.sh devices

driver-status:
	./gpu_driver.sh status

driver-xe:
	./gpu_driver.sh xe

driver-i915:
	./gpu_driver.sh i915

# make bench sycl / make start high 的位置参数在此吞掉
all vulkan sycl sycl-tensor sycl-mtp sycl-tensor-mtp low medium high:
	@:
