#!/bin/bash
# 实测各 GPU 的 PCIe 带宽: 锁页主机内存 <-> 显存, 256 MiB x 5 轮, 报 min/mean/max GB/s
# 依赖: cc + OpenCL 头文件 (opencl-headers), 缓存编译产物到 ~/.cache/pcie_speed
# 用法: ./pcie_speed.sh

set -euo pipefail

CACHE_DIR=${XDG_CACHE_HOME:-$HOME/.cache}/pcie_speed
BIN=$CACHE_DIR/pcie_bw
SRC=$CACHE_DIR/pcie_bw.c
TMP=$CACHE_DIR/src.tmp

err() { echo "错误: $*" >&2; exit 1; }

command -v cc >/dev/null || err "找不到 cc (需 gcc 或 clang)"

mkdir -p "$CACHE_DIR"

cat > "$TMP" <<'EOF'
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define PASSES 2
#define ROUNDS 5
#define WARMUP_S 5.0

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static cl_int transfer(cl_command_queue q, cl_mem d, char *hp, size_t N, int write)
{
	if (write)
		return clEnqueueWriteBuffer(q, d, CL_TRUE, 0, N, hp, 0, NULL, NULL);
	return clEnqueueReadBuffer(q, d, CL_TRUE, 0, N, hp, 0, NULL, NULL);
}

static cl_int warmup(cl_command_queue q, cl_mem d, char *hp, size_t N)
{
	double t0 = now_s();

	while (now_s() - t0 < WARMUP_S) {
		cl_int err = transfer(q, d, hp, N, 1);

		if (err == CL_SUCCESS)
			err = transfer(q, d, hp, N, 0);
		if (err != CL_SUCCESS)
			return err;
	}
	return CL_SUCCESS;
}

static cl_int bench(cl_command_queue q, cl_mem d, char *hp, size_t N, int write,
		    double *min, double *mean, double *max)
{
	double tmin = 1e30, tmax = 0, sum = 0;

	for (int k = 0; k < ROUNDS; k++) {
		double t0 = now_s();
		cl_int err = transfer(q, d, hp, N, write);
		double dt;

		if (err != CL_SUCCESS)
			return err;
		dt = now_s() - t0;
		tmin = dt < tmin ? dt : tmin;
		tmax = dt > tmax ? dt : tmax;
		sum += dt;
	}
	*min = N / tmax / 1e9;
	*mean = N / (sum / ROUNDS) / 1e9;
	*max = N / tmin / 1e9;
	return CL_SUCCESS;
}

int main(void)
{
	cl_platform_id plats[16];
	cl_uint np = 0;
	const size_t N = 256ul << 20;

	if (clGetPlatformIDs(16, plats, &np) != CL_SUCCESS || np == 0) {
		printf("无 OpenCL 平台\n");
		return 1;
	}
	printf("min/mean/max GB/s, %d 遍取优, 每遍预热 %.0fs + %d 轮 x %zu MiB\n",
	       PASSES, WARMUP_S, ROUNDS, N >> 20);
	for (cl_uint i = 0; i < np; i++) {
		cl_device_id devs[16];
		cl_uint nd = 0;

		if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 16, devs, &nd) != CL_SUCCESS || nd == 0)
			continue;
		for (cl_uint j = 0; j < nd; j++) {
			char name[128] = {0};
			cl_int err = CL_SUCCESS;

			clGetDeviceInfo(devs[j], CL_DEVICE_NAME, sizeof name, name, NULL);
			cl_context ctx = clCreateContext(NULL, 1, &devs[j], NULL, NULL, &err);
			if (!ctx) {
				printf("%-40s 上下文创建失败 (err %d)\n", name, err);
				continue;
			}
			cl_command_queue q = clCreateCommandQueue(ctx, devs[j], 0, &err);
			cl_mem d = q ? clCreateBuffer(ctx, CL_MEM_READ_WRITE, N, NULL, &err) : NULL;
			cl_mem h = q ? clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N, NULL, &err) : NULL;
			char *hp = (d && h) ? clEnqueueMapBuffer(q, h, CL_TRUE, CL_MAP_WRITE, 0, N, 0, NULL, NULL, &err) : NULL;

			if (!hp) {
				printf("%-40s 显存分配失败 (err %d)\n", name, err);
			} else {
				double wmin, wmean, wmax, rmin, rmean, rmax, best = -1;
				double b[6];
				int p;

				memset(hp, 1, N);
				for (p = 0; p < PASSES; p++) {
					if ((err = warmup(q, d, hp, N)) != CL_SUCCESS ||
					    (err = bench(q, d, hp, N, 1, &wmin, &wmean, &wmax)) != CL_SUCCESS ||
					    (err = bench(q, d, hp, N, 0, &rmin, &rmean, &rmax)) != CL_SUCCESS) {
						printf("%-40s 传输失败 (err %d)\n", name, err);
						goto release;
					}
					if (wmean + rmean > best) {
						best = wmean + rmean;
						b[0] = wmin; b[1] = wmean; b[2] = wmax;
						b[3] = rmin; b[4] = rmean; b[5] = rmax;
					}
				}
				printf("%-40s 主机→显存 %5.2f/%5.2f/%5.2f GB/s  显存→主机 %5.2f/%5.2f/%5.2f GB/s\n",
				       name, b[0], b[1], b[2], b[3], b[4], b[5]);
			}
release:
			if (h)
				clReleaseMemObject(h);
			if (d)
				clReleaseMemObject(d);
			if (q)
				clReleaseCommandQueue(q);
			clReleaseContext(ctx);
		}
	}
	return 0;
}
EOF

if [[ ! -x "$BIN" ]] || ! cmp -s "$TMP" "$SRC"; then
    mv "$TMP" "$SRC"
    echo "== 编译 $BIN =="
    cc -O2 -o "$BIN" "$SRC" -l:libOpenCL.so.1 ||
        { rm -f "$BIN"; err "编译失败: 需 gcc 与 OpenCL 头文件 (opencl-headers)"; }
else
    rm -f "$TMP"
fi

exec "$BIN"
