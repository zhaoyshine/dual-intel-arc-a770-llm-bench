#!/bin/bash
# 切换 Intel Arc A770 内核驱动 i915 / xe (改引导参数 + 重建 initramfs, 重启生效)
# 用法: ./gpu_driver.sh status|i915|xe
# 切 xe 要显式 force_probe: DG2 内核对 i915 是默认, xe 不是。

set -euo pipefail

PCI_ID=8086:56a0        # DG2 [Arc A770]
ARGS_XE="xe.force_probe=56a0 i915.force_probe=!56a0"
ARGS_I915="i915.force_probe=56a0 xe.force_probe=!56a0"

err() { echo "错误: $*" >&2; exit 1; }

usage() {
    cat <<'EOF'
用法: ./gpu_driver.sh status|i915|xe

  status  显示每张 A770 当前绑定的驱动与引导参数
  i915    切回默认的 i915 驱动 (重启生效)
  xe      切到 xe 驱动 (重启生效)
EOF
}

a770_devs() {
    lspci -Dn | awk -v id="$PCI_ID" '$3 == id { print $1 }'
}

current_driver() {
    local link
    link=$(readlink "/sys/bus/pci/devices/$1/driver" 2>/dev/null) || true
    if [[ -n "$link" ]]; then
        basename "$link"
    else
        echo 未绑定
    fi
}

show_status() {
    local dev probe
    while read -r dev; do
        printf '%s  %s\n' "$dev" "$(current_driver "$dev")"
    done < <(a770_devs)
    probe=$(grep -oE '(i915|xe)\.force_probe=[^ ]*' /proc/cmdline | tr '\n' ' ') || true
    echo "引导参数: ${probe:-无}"
}

set_driver() {
    local driver=$1 args
    case "$driver" in
        xe)   args=$ARGS_XE ;;
        i915) args=$ARGS_I915 ;;
    esac
    command -v grubby >/dev/null || err "找不到 grubby"
    command -v dracut >/dev/null || err "找不到 dracut"
    sudo grubby --update-kernel=ALL \
        --remove-args="i915.force_probe xe.force_probe" \
        --args="$args"
    sudo dracut -f
    echo "已切到 $driver, 重启生效: sudo systemctl reboot"
    echo "重启后 ./gpu_driver.sh status 应显示 $driver"
}

[[ -n "$(a770_devs)" ]] || err "没找到 A770 ($PCI_ID)"

case "${1:-}" in
    status)  show_status ;;
    i915|xe) set_driver "$1" ;;
    -h|--help|help) usage ;;
    *)       usage >&2; err "未知参数 '${1:-}'" ;;
esac
