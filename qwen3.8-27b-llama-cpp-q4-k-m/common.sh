#!/bin/bash
# 共用工具函数, 被各脚本 source

err() { echo "错误: $*" >&2; exit 1; }

usage() { echo "$@" >&2; }
