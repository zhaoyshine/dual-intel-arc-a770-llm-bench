SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help build start monitor clean

help:
	@echo "make build          本机编译 c0ven (核心 + kernel + Python 前端)"
	@echo "make start          启动服务 (127.0.0.1:8080)"
	@echo "make monitor        终端监控面板, 每 1s 刷新指标 (地址取 config.sh 的 HOST/PORT)"
	@echo "make clean          删除 build/ 和 engine 表缓存 cache/"

build:
	./build.sh

start:
	./start.sh

monitor:
	@SCRIPT_DIR=$(CURDIR) . ./config.sh >/dev/null && exec "$$VENV/bin/python" monitor.py "http://$$HOST:$$PORT"
clean:
	rm -rf build cache
