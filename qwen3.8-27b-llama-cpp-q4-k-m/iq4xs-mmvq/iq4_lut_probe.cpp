// Isolated SYCL probe for the IQ4_XS GEMV inner loop used by ggml-sycl's
// vec_dot_iq4_xs_q8_1 (8 lanes per 256-weight block, 4 uint32 loads + 8 dp4a per lane).
// Varies the table lookup and the block layout to separate LUT cost from memory cost.
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

template <typename T1, typename T2, typename T3>
static inline auto dp4a(T1 a, T2 b, T3 c) {
    auto va = sycl::vec<T1, 1>(a).template as<sycl::vec<int8_t, 4>>().template convert<int>();
    auto vb = sycl::vec<T2, 1>(b).template as<sycl::vec<int8_t, 4>>().template convert<int>();
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}

static const int8_t kvalues[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

static const uint16_t table_pair[256] = {
    33153, 33176, 33197, 33215, 33231, 33245, 33258, 33270,
    33025, 33037, 33049, 33062, 33077, 33093, 33113, 33137,
    39041, 39064, 39085, 39103, 39119, 39133, 39146, 39158,
    38913, 38925, 38937, 38950, 38965, 38981, 39001, 39025,
    44417, 44440, 44461, 44479, 44495, 44509, 44522, 44534,
    44289, 44301, 44313, 44326, 44341, 44357, 44377, 44401,
    49025, 49048, 49069, 49087, 49103, 49117, 49130, 49142,
    48897, 48909, 48921, 48934, 48949, 48965, 48985, 49009,
    53121, 53144, 53165, 53183, 53199, 53213, 53226, 53238,
    52993, 53005, 53017, 53030, 53045, 53061, 53081, 53105,
    56705, 56728, 56749, 56767, 56783, 56797, 56810, 56822,
    56577, 56589, 56601, 56614, 56629, 56645, 56665, 56689,
    60033, 60056, 60077, 60095, 60111, 60125, 60138, 60150,
    59905, 59917, 59929, 59942, 59957, 59973, 59993, 60017,
    63105, 63128, 63149, 63167, 63183, 63197, 63210, 63222,
    62977, 62989, 63001, 63014, 63029, 63045, 63065, 63089,
    385, 408, 429, 447, 463, 477, 490, 502,
    257, 269, 281, 294, 309, 325, 345, 369,
    3457, 3480, 3501, 3519, 3535, 3549, 3562, 3574,
    3329, 3341, 3353, 3366, 3381, 3397, 3417, 3441,
    6529, 6552, 6573, 6591, 6607, 6621, 6634, 6646,
    6401, 6413, 6425, 6438, 6453, 6469, 6489, 6513,
    9857, 9880, 9901, 9919, 9935, 9949, 9962, 9974,
    9729, 9741, 9753, 9766, 9781, 9797, 9817, 9841,
    13697, 13720, 13741, 13759, 13775, 13789, 13802, 13814,
    13569, 13581, 13593, 13606, 13621, 13637, 13657, 13681,
    17793, 17816, 17837, 17855, 17871, 17885, 17898, 17910,
    17665, 17677, 17689, 17702, 17717, 17733, 17753, 17777,
    22913, 22936, 22957, 22975, 22991, 23005, 23018, 23030,
    22785, 22797, 22809, 22822, 22837, 22853, 22873, 22897,
    29057, 29080, 29101, 29119, 29135, 29149, 29162, 29174,
    28929, 28941, 28953, 28966, 28981, 28997, 29017, 29041
};

static void init_tables() {}

// 8 indexed byte loads per 4-byte word (upstream implementation)
static inline void lut_scalar(const int8_t * __restrict__ kv, uint32_t w, int & v1, int & v2) {
    uint32_t aux = w & 0x0f0f0f0f;
    const uint8_t * q = (const uint8_t *) &aux;
    v1 = (kv[q[0]] | (kv[q[1]] << 8) | (kv[q[2]] << 16) | (kv[q[3]] << 24));
    aux = (w >> 4) & 0x0f0f0f0f;
    q = (const uint8_t *) &aux;
    v2 = (kv[q[0]] | (kv[q[1]] << 8) | (kv[q[2]] << 16) | (kv[q[3]] << 24));
}

// 4 uint16 pair lookups + 6 mask/shift ops per 4-byte word
static inline void lut_pair(const uint16_t * __restrict__ tbl, uint32_t w, int & v1, int & v2) {
    const uint32_t t0 = tbl[w & 0xff];
    const uint32_t t1 = tbl[(w >> 8) & 0xff];
    const uint32_t t2 = tbl[(w >> 16) & 0xff];
    const uint32_t t3 = tbl[(w >> 24) & 0xff];
    v1 = (int) ((t0 & 0xff) | ((t1 & 0xff) << 8) | ((t2 & 0xff) << 16) | ((t3 & 0xff) << 24));
    v2 = (int) (((t0 >> 8) & 0xff) | (((t1 >> 8) & 0xff) << 8) | (((t2 >> 8) & 0xff) << 16) | (((t3 >> 8) & 0xff) << 24));
}

// 4 uint16 lookups from a kernel-static const table
static inline void lut_pair_const(uint32_t w, int & v1, int & v2) {
    const uint32_t t0 = table_pair[w & 0xff];
    const uint32_t t1 = table_pair[(w >> 8) & 0xff];
    const uint32_t t2 = table_pair[(w >> 16) & 0xff];
    const uint32_t t3 = table_pair[(w >> 24) & 0xff];
    v1 = (int) ((t0 & 0xff) | ((t1 & 0xff) << 8) | ((t2 & 0xff) << 16) | ((t3 & 0xff) << 24));
    v2 = (int) (((t0 >> 8) & 0xff) | (((t1 >> 8) & 0xff) << 8) | (((t2 >> 8) & 0xff) << 16) | (((t3 >> 8) & 0xff) << 24));
}

struct lut_arg_t { uint16_t t[256]; };

// 4 uint16 lookups from a table passed by value (kernel arg -> constant memory)
static inline void lut_pair_slm(const sycl::local_accessor<uint16_t, 1> & lut, uint32_t w, int & v1, int & v2) {
    const uint32_t t0 = lut[w & 0xff];
    const uint32_t t1 = lut[(w >> 8) & 0xff];
    const uint32_t t2 = lut[(w >> 16) & 0xff];
    const uint32_t t3 = lut[(w >> 24) & 0xff];
    v1 = (int) ((t0 & 0xff) | ((t1 & 0xff) << 8) | ((t2 & 0xff) << 16) | ((t3 & 0xff) << 24));
    v2 = (int) (((t0 >> 8) & 0xff) | (((t1 >> 8) & 0xff) << 8) | (((t2 >> 8) & 0xff) << 16) | (((t3 >> 8) & 0xff) << 24));
}

static inline void lut_pair_arg(const lut_arg_t & tbl, uint32_t w, int & v1, int & v2) {
    const uint32_t t0 = tbl.t[w & 0xff];
    const uint32_t t1 = tbl.t[(w >> 8) & 0xff];
    const uint32_t t2 = tbl.t[(w >> 16) & 0xff];
    const uint32_t t3 = tbl.t[(w >> 24) & 0xff];
    v1 = (int) ((t0 & 0xff) | ((t1 & 0xff) << 8) | ((t2 & 0xff) << 16) | ((t3 & 0xff) << 24));
    v2 = (int) (((t0 >> 8) & 0xff) | (((t1 >> 8) & 0xff) << 8) | (((t2 >> 8) & 0xff) << 16) | (((t3 >> 8) & 0xff) << 24));
}

// no LUT (wrong math): lower bound on the memory-only cost
static inline void lut_none(uint32_t w, int & v1, int & v2) {
    v1 = (int) (w & 0x0f0f0f0f);
    v2 = (int) ((w >> 4) & 0x0f0f0f0f);
}

// layout 0: reordered/SoA (qs contiguous, 128B per block, meta in a separate array)
// layout 1: upstream interleaved (136B per block: d, scales_h, scales_l[4], qs[128])
struct layout_t {
    const uint8_t * qs_base;
    const uint8_t * meta_base;
    int             qs_stride;
    int             qs_offset;
    int             meta_stride;
};

#ifndef SG
#define SG 16
#endif
#define NSG 16   // subgroups per workgroup, matching the backend's launch shape

template <int MODE>
static void gemv(sycl::queue & q, const layout_t l, const int8_t * kv, const uint16_t * tbl, const int8_t * act,
                 const float * ds, float * out, int blocks_per_row, int rows, const lut_arg_t luta = {}) {
    sycl::local_accessor<uint16_t, 1> dummy;
    const size_t nsub = NSG;
    const size_t ngrp = ((size_t) rows + nsub - 1) / nsub;
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ngrp * nsub * SG), sycl::range<3>(1, 1, nsub * SG)),
                   [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg   = it.get_sub_group();
        const int  row  = (int) (it.get_group(2) * sg.get_group_linear_range() + sg.get_group_linear_id());
        const int  lane = (int) sg.get_local_linear_id();
        if (row >= rows) return;
        const int ib32 = lane & 7;
        const size_t row_block0 = (size_t) row * blocks_per_row;
        float sum = 0.0f;
        for (int b = lane >> 3; b < blocks_per_row; b += SG / 8) {
            const size_t   blk = row_block0 + b;
            const uint8_t * qs  = l.qs_base + blk * l.qs_stride + l.qs_offset;
            const uint8_t * mt  = l.meta_base + blk * l.meta_stride;
            const uint32_t * q4 = (const uint32_t *) (qs + 16 * ib32);
            const int      * a8 = (const int *) (act + 32 * ib32);
            const int8_t     ls = (int8_t) (((mt[4 + ib32 / 2] >> 4 * (ib32 % 2)) & 0xf) |
                                            (((*(const uint16_t *) (mt + 2) >> 2 * ib32) & 3) << 4));
            const float d = (float) (*(const sycl::half *) mt) * (float) (ls - 32) * ds[ib32];
            int sumi1 = 0, sumi2 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int v1, v2;
                if constexpr (MODE == 0) lut_scalar(kv, q4[j], v1, v2);
                if constexpr (MODE == 1) lut_pair(tbl, q4[j], v1, v2);
                if constexpr (MODE == 2) lut_none(q4[j], v1, v2);
                if constexpr (MODE == 3) lut_pair_const(q4[j], v1, v2);
                if constexpr (MODE == 4) lut_pair_arg(luta, q4[j], v1, v2);
                sumi1 = dp4a(v1, a8[j + 0], sumi1);
                sumi2 = dp4a(v2, a8[j + 4], sumi2);
            }
            sum += d * (float) (sumi1 + sumi2);
        }
        for (int m = SG / 2; m > 0; m >>= 1) {
            sum += sycl::shift_group_left(sg, sum, m);
        }
        if (lane == 0) {
            out[row] = sum;
        }
    });
    q.wait();
}

static lut_arg_t g_lut;

static void run_mode(sycl::queue & q, int mode, const layout_t l, const int8_t * kv, const uint16_t * tbl,
                     const int8_t * act, const float * ds, float * out, int bpr, int rows, int iters,
                     const char * name) {
    for (int i = 0; i < 3; ++i) {
        if (mode == 0) gemv<0>(q, l, kv, tbl, act, ds, out, bpr, rows);
        if (mode == 1) gemv<1>(q, l, kv, tbl, act, ds, out, bpr, rows);
        if (mode == 2) gemv<2>(q, l, kv, tbl, act, ds, out, bpr, rows);
        if (mode == 3) gemv<3>(q, l, kv, tbl, act, ds, out, bpr, rows);
        if (mode == 4) gemv<4>(q, l, kv, tbl, act, ds, out, bpr, rows, g_lut);
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (mode == 0) gemv<0>(q, l, kv, tbl, act, ds, out, bpr, rows);
        if (mode == 1) gemv<1>(q, l, kv, tbl, act, ds, out, bpr, rows);
        if (mode == 2) gemv<2>(q, l, kv, tbl, act, ds, out, bpr, rows);
        if (mode == 3) gemv<3>(q, l, kv, tbl, act, ds, out, bpr, rows);
        if (mode == 4) gemv<4>(q, l, kv, tbl, act, ds, out, bpr, rows, g_lut);
    }
    auto   t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    size_t bytes = (size_t) rows * bpr * 136;
    std::printf("%-34s %10.3f %10.1f\n", name, ms, (double) bytes / 1e9 / (ms / 1000.0));
}

int main(int argc, char ** argv) {
    const int nrows = argc > 1 ? std::atoi(argv[1]) : 17408;
    const int ncols = argc > 2 ? std::atoi(argv[2]) : 5120;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 40;
    const int bpr   = ncols / 256;

    init_tables();
    for (int i = 0; i < 256; ++i) g_lut.t[i] = table_pair[i];
    sycl::queue q{ sycl::gpu_selector_v };
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("%-34s %10s %10s\n", "variant", "ms/iter", "GB/s");

    // interleaved buffer (upstream layout)
    const size_t nb    = (size_t) nrows * bpr;
    const size_t bytes = nb * 136;
    uint8_t * w = sycl::malloc_device<uint8_t>(bytes, q);
    {
        std::vector<uint8_t> h(bytes);
        for (size_t i = 0; i < bytes; ++i) h[i] = (uint8_t) (i * 2654435761u >> 13);
        q.memcpy(w, h.data(), bytes).wait();
    }
    // SoA buffer (reordered layout): qs 128B/block, meta 8B/block
    uint8_t * qs_soa   = sycl::malloc_device<uint8_t>(nb * 128, q);
    uint8_t * meta_soa = sycl::malloc_device<uint8_t>(nb * 8, q);
    q.memset(qs_soa, 0x5a, nb * 128).wait();
    q.memset(meta_soa, 0x3c, nb * 8).wait();

    const size_t nq8 = (size_t) ncols / 32;
    int8_t * act = sycl::malloc_device<int8_t>(nq8 * 32, q);
    float  * ds  = sycl::malloc_device<float>(nq8, q);
    float  * out = sycl::malloc_device<float>(nrows, q);
    {
        std::vector<int8_t> h(nq8 * 32);
        std::vector<float>  hd(nq8);
        for (size_t i = 0; i < h.size(); ++i) h[i] = (int8_t) ((i * 37) % 255 - 127);
        for (size_t i = 0; i < hd.size(); ++i) hd[i] = 0.01f * (float) (i % 17) - 0.08f;
        q.memcpy(act, h.data(), h.size()).wait();
        q.memcpy(ds, hd.data(), hd.size() * sizeof(float)).wait();
    }

    int8_t * kv  = sycl::malloc_device<int8_t>(16, q);
    uint16_t * tbl = sycl::malloc_device<uint16_t>(256, q);
    q.memcpy(kv, kvalues, 16).wait();
    q.memcpy(tbl, table_pair, 512).wait();

    layout_t inter = { w, w, 136, 8, 136 };
    layout_t soa   = { qs_soa, meta_soa, 128, 0, 8 };
    layout_t soa64 = { qs_soa + 64, meta_soa, 128, 0, 8 };
    layout_t soa32 = { qs_soa + 32, meta_soa, 128, 0, 8 };

    run_mode(q, 0, inter, kv, tbl, act, ds, out, bpr, nrows, iters, "interleaved 136B + scalar LUT");
    run_mode(q, 1, inter, kv, tbl, act, ds, out, bpr, nrows, iters, "interleaved 136B + pair LUT");
    run_mode(q, 2, inter, kv, tbl, act, ds, out, bpr, nrows, iters, "interleaved 136B + no LUT");
    run_mode(q, 0, soa, kv, tbl, act, ds, out, bpr, nrows, iters, "SoA 128B + scalar LUT");
    run_mode(q, 1, soa, kv, tbl, act, ds, out, bpr, nrows, iters, "SoA 128B + pair LUT");
    run_mode(q, 2, soa, kv, tbl, act, ds, out, bpr, nrows, iters, "SoA 128B + no LUT");
    run_mode(q, 3, soa, kv, tbl, act, ds, out, bpr, nrows, iters, "SoA 128B + const pair LUT");
    run_mode(q, 3, inter, kv, tbl, act, ds, out, bpr, nrows, iters, "interleaved + const pair LUT");
    run_mode(q, 4, soa, kv, tbl, act, ds, out, bpr, nrows, iters, "SoA 128B + arg pair LUT");
    run_mode(q, 3, soa64, kv, tbl, act, ds, out, bpr, nrows, iters, "SoA +64B misaligned, const LUT");
    run_mode(q, 3, soa32, kv, tbl, act, ds, out, bpr, nrows, iters, "SoA +32B misaligned, const LUT");

    sycl::free(kv, q);
    sycl::free(tbl, q);
    sycl::free(w, q);
    sycl::free(qs_soa, q);
    sycl::free(meta_soa, q);
    sycl::free(act, q);
    sycl::free(ds, q);
    sycl::free(out, q);
    return 0;
}
