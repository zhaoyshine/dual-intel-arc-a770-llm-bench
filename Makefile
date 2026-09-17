SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help build bench start devices driver-status driver-xe driver-i915

help:
	@echo "make build               编译 SYCL 后端 (build_sycl.sh)"
	@echo "make bench [配置]        跑基准; 不带参数跑全部"
	@echo "                         配置: vulkan-official vulkan sycl sycl-tensor sycl-mtp"
	@echo "make start [模式]        启动 llama-server; 模式: mtp (默认) base"
	@echo "make devices             列出 llama.cpp 可见设备"
	@echo "make driver-status       查看 A770 当前内核驱动"
	@echo "make driver-xe           切到 xe 驱动 (重启生效)"
	@echo "make driver-i915         切回 i915 驱动 (重启生效)"

build:
	./build_sycl.sh

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

# make bench sycl / make start base 的位置参数在此吞掉
%:
	@:
