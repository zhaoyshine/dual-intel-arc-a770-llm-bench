SHELL := /bin/bash
.DEFAULT_GOAL := help

STRATA_DIR := $(HOME)/workspace/ai/Strata
PYTHON := $(STRATA_DIR)/.venv/bin/python
PROFILE := $(CURDIR)/strata-coder-iq1-m.json

.PHONY: help build start bench clean

help:
	@echo "make build          编译 Strata SYCL 引擎 (A770, JIT)"
	@echo "make start          启动 OpenAI 兼容服务 (127.0.0.1:8080)"
	@echo "make bench          Strata 官方标准 benchmark (standard-v1, 自动起停引擎)"
	@echo "make clean          删除 build-sycl"

build:
	./build.sh

start:
	./start.sh

bench:
	. ./sycl_env.sh && $(PYTHON) $(STRATA_DIR)/strata.py bench run --profile $(PROFILE) $(ARGS)

clean:
	rm -rf ~/workspace/ai/Strata/build-sycl
