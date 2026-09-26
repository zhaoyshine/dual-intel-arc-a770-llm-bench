SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help build bench devices

help:
	@echo "make build               编译/安装 vLLM XPU 与两个 A770 适配插件"
	@echo "make bench [配置]        跑基准; 不带参数跑两个配置 (tp / pp)"
	@echo "                         配置: tp pp"
	@echo "make devices             列出 vLLM 可见的 XPU 设备"

build:
	./build.sh

bench:
	./bench.sh $(filter-out $@,$(MAKECMDGOALS))

devices:
	./bench.sh devices

# 吞掉 make bench tp|pp 的位置参数
%:
	@:
