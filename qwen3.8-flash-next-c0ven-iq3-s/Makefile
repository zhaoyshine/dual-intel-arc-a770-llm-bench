SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help build start monitor clean
URL ?= http://127.0.0.1:8080

help:
	@echo "make build          本机编译 c0ven (核心 + kernel + Python 前端)"
	@echo "make start          启动服务 (127.0.0.1:8080)"
	@echo "make monitor        终端监控面板, 每 1s 刷新指标 (URL=http://host:port 可改)"
	@echo "make clean          删除 build/ 和依赖缓存"

build:
	./build.sh

start:
	./start.sh

monitor:
	@SCRIPT_DIR=$(CURDIR) . ./config.sh >/dev/null && exec "$$VENV/bin/python" monitor.py "$(URL)"
clean:
	rm -rf build cache
