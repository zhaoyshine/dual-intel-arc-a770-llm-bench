SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help build bench start sycl sycl-tensor sycl-tensor-mtp low medium high

help:
	@echo "make build               导出 llama.cpp HEAD + 打 patches/*.patch, 构建到 ~/workspace/ai/llama.cpp/build-sycl-opt"
	@echo "make bench [配置]        跑基准; 配置: sycl(layer) sycl-tensor sycl-tensor-mtp; 默认三种都跑"
	@echo "make start [档位]        启动 llama-server; 思考档位: low (默认) medium high"

build:
	./build.sh

bench:
	./bench.sh $(filter-out $@,$(MAKECMDGOALS))

start:
	./start.sh $(filter-out $@,$(MAKECMDGOALS))

# make bench sycl / make start high 的位置参数在此吞掉
sycl sycl-tensor sycl-tensor-mtp low medium high:
	@:
