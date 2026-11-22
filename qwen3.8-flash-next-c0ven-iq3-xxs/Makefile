SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help build start clean

help:
	@echo "make build          本机编译 c0ven (核心 + kernel + Python 前端)"
	@echo "make start          启动服务 (127.0.0.1:8080)"
	@echo "make clean          删除 build/ 和依赖缓存"

build:
	./build.sh

start:
	./start.sh

clean:
	rm -rf build cache
