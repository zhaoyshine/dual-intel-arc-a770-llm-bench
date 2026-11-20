SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help build bench start 8b sycl sycl-tensor

help:
	@echo "make build               导出 llama.cpp HEAD + 打 patches/*.patch, 构建到 ~/workspace/ai/llama.cpp/build-sycl-opt"
	@echo "make bench [配置]        跑基准; 8b 27b sycl sycl-tensor; 默认全跑"
	@echo "make start               启动 llama-server; 思考档位写死 low"

build:
	./build.sh

bench:
	./bench.sh $(filter-out $@,$(MAKECMDGOALS))

start:
	./start.sh

# make bench sycl 等的位置参数在此吞掉
8b sycl sycl-tensor:
	@:
