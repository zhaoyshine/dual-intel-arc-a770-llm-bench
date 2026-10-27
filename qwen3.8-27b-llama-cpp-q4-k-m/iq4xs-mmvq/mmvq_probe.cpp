// Matvec (1-token decode) probe: measures achieved weight-read bandwidth of the
// SYCL MMVQ kernels for the quant/shape pairs this model actually uses.
// Usage: mmvq_probe <libggml-sycl.so path> <device name> [n-iterations]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct case_t {
    const char * name;
    ggml_type    type;
    int64_t      ncols;  // rows of src0, i.e. output size
    int64_t      nrows;  // K dimension
    int64_t      batch = 1;
};

// host reference: dequantize IQ4_XS and dot with the same f32 activations
static const int8_t kv_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

static void iq4_xs_reference(const uint8_t * blk, const float * act, int64_t nrows, int64_t ncols, int64_t row,
                             double & ref_out) {
    const int64_t nb = ncols / 256;
    double        acc = 0.0;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * x = blk + (row * nb + b) * 136;
        const float     d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) x);
        const uint16_t  sh = *(const uint16_t *) (x + 2);
        const uint8_t * qs = x + 8;
        const float   * a  = act + b * 256;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((x[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
            const float scale = d * (float) (ls - 32);
            for (int j = 0; j < 16; ++j) {
                const uint8_t byte = qs[ib * 16 + j];
                acc += (double) a[ib * 32 + j] * scale * (double) kv_iq4nl[byte & 0xf];
                acc += (double) a[ib * 32 + 16 + j] * scale * (double) kv_iq4nl[byte >> 4];
            }
        }
    }
    ref_out = acc;
}

static size_t rand_init(ggml_tensor * t, int variant) {
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    // deterministic, non-degenerate payload; quantization-friendly magnitudes
    for (size_t i = 0; i < nbytes; ++i) {
        buf[i] = (uint8_t) ((i * 1103515245u + variant * 12345u) >> 16);
    }
    // f32 tensors need sane values
    if (t->type == GGML_TYPE_F32) {
        float * f = (float *) buf.data();
        const int64_t n = nbytes / sizeof(float);
        for (int64_t i = 0; i < n; ++i) {
            f[i] = 0.01f * (float) ((i * 37) % 100 - 50);
        }
    }
    ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    return nbytes;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <libggml-sycl.so> <device> [iters]\n", argv[0]);
        return 1;
    }
    const int iters = argc > 3 ? std::atoi(argv[3]) : 20;
    const char * only = argc > 4 ? argv[4] : nullptr;

    ggml_backend_load(argv[1]);
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(argv[2]);
    if (!dev) {
        std::fprintf(stderr, "no device %s\n", argv[2]);
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    const std::vector<case_t> cases = {
        { "Q4_K  ffn_gate", GGML_TYPE_Q4_K,   17408, 5120 },
        { "Q4_K  ffn_down", GGML_TYPE_Q4_K,    5120, 17408 },
        { "Q4_K  17408/1280", GGML_TYPE_Q4_K, 17408, 1280 },
        { "Q5_K  attn_qkv", GGML_TYPE_Q5_K,   10240, 5120 },
        { "Q5_K  ffn_up",   GGML_TYPE_Q5_K,   17408, 5120 },
        { "Q5_K  ssm_out",  GGML_TYPE_Q5_K,    6144, 5120 },
        { "Q6_K  ffn_down", GGML_TYPE_Q6_K,    5120, 17408 },
        { "IQ4_XS ffn_gate",GGML_TYPE_IQ4_XS, 17408, 5120 },
        { "IQ4_XS ffn_gate n2",GGML_TYPE_IQ4_XS, 17408, 5120, 2 },
        { "IQ4_XS ffn_gate n3",GGML_TYPE_IQ4_XS, 17408, 5120, 3 },
        { "IQ4_XS ffn_gate n4",GGML_TYPE_IQ4_XS, 17408, 5120, 4 },
        { "IQ4_XS ffn_gate n6",GGML_TYPE_IQ4_XS, 17408, 5120, 6 },
        { "IQ4_XS ffn_gate n8",GGML_TYPE_IQ4_XS, 17408, 5120, 8 },
        { "IQ4_XS ffn_down",GGML_TYPE_IQ4_XS,  5120, 17408 },
        { "IQ4_XS ffn_down n2",GGML_TYPE_IQ4_XS, 5120, 17408, 2 },
        { "IQ4_XS ffn_down n4",GGML_TYPE_IQ4_XS, 5120, 17408, 4 },
        { "IQ4_XS 17408/1280", GGML_TYPE_IQ4_XS, 17408, 1280 },
        { "IQ4_XS 17408/1280 n4", GGML_TYPE_IQ4_XS, 17408, 1280, 4 },
        { "IQ4_XS attn_qkv", GGML_TYPE_IQ4_XS, 10240, 5120 },
        { "IQ4_XS attn_qkv n2", GGML_TYPE_IQ4_XS, 10240, 5120, 2 },
        { "IQ4_XS attn_qkv n4", GGML_TYPE_IQ4_XS, 10240, 5120, 4 },
        { "IQ4_XS attn_gate", GGML_TYPE_IQ4_XS, 6144, 5120 },
        { "IQ4_XS attn_gate n2", GGML_TYPE_IQ4_XS, 6144, 5120, 2 },
        { "IQ4_XS attn_gate n4", GGML_TYPE_IQ4_XS, 6144, 5120, 4 },
        { "IQ4_NL ffn_down",GGML_TYPE_IQ4_NL, 17408, 5120 },
        { "IQ4_NL ffn_up",  GGML_TYPE_IQ4_NL, 17408, 5120 },
        { "IQ4_NL 17408/1280", GGML_TYPE_IQ4_NL, 17408, 1280 },
        { "IQ3_S attn_qkv", GGML_TYPE_IQ3_S,  10240, 5120 },
        { "Q3_K  ffn_up",   GGML_TYPE_Q3_K,   17408, 5120 },
        { "Q4_0  17408",    GGML_TYPE_Q4_0,   17408, 5120 },
    };

    std::printf("%-18s %8s %8s %10s %10s %8s\n", "case", "rows", "cols", "ms/iter", "GB/s", "MiB");
    for (const case_t & c : cases) {
        if (only && !std::strstr(c.name, only)) continue;
        std::fprintf(stderr, "[case] %s\n", c.name);
        ggml_init_params ip = { /*mem_size*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead(), /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * w = ggml_new_tensor_2d(ctx, c.type, c.nrows, c.ncols);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.nrows, c.batch);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_set_output(y);

        // deterministic f32 activations and, for quantized types, a validly quantized weight matrix
        std::vector<float> host_act(c.nrows * c.batch);
        for (int64_t i = 0; i < c.nrows * c.batch; ++i) {
            host_act[i] = 0.02f * (float) ((i * 7919) % 101 - 50);
        }
        std::vector<float> host_w;
        for (int64_t i = 0; i < c.ncols * c.nrows; ++i) {
            host_w.push_back(0.02f * (float) ((i * 1031) % 199 - 99));
        }
        bool have_ref = c.type == GGML_TYPE_IQ4_XS;
        std::vector<double> ref(c.ncols * c.batch, 0.0);

        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!wbuf) {
            std::printf("%-18s  alloc failed\n", c.name);
            ggml_free(ctx);
            continue;
        }
        std::vector<uint8_t> w_head;
        {
            std::vector<uint8_t> qbuf(ggml_nbytes(w));
            const size_t written = ggml_quantize_chunk(c.type, host_w.data(), qbuf.data(), 0, c.ncols, c.nrows, nullptr);
            GGML_ASSERT(written == qbuf.size());
            w_head.assign(qbuf.begin(), qbuf.begin() + 136);
            ggml_backend_tensor_set(w, qbuf.data(), 0, qbuf.size());
            ggml_backend_tensor_set(x, host_act.data(), 0, c.nrows * c.batch * sizeof(float));
            if (have_ref) {
                for (int64_t j = 0; j < c.batch; ++j) {
                    for (int64_t r = 0; r < c.ncols; ++r) {
                        iq4_xs_reference(qbuf.data(), host_act.data() + j * c.nrows, c.ncols, c.nrows, r,
                                         ref[j * c.ncols + r]);
                    }
                }
            }
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);

        // time-based warmup: the GPU needs hundreds of ms of load to reach its boost clock, and a
        // short run at the idle clock would otherwise dominate the average
        const auto warm_t0 = std::chrono::steady_clock::now();
        double     warm_ms = 0.0;
        while (warm_ms < 1000.0) {
            ggml_backend_graph_compute(backend, gf);  // also triggers the one-time reorder
            warm_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warm_t0).count();
        }
        ggml_backend_synchronize(backend);

        bool reordered = false;
        if (c.type == GGML_TYPE_IQ4_XS) {
            std::vector<uint8_t> head(w_head.size());
            ggml_backend_tensor_get(w, head.data(), 0, head.size());
            ggml_backend_synchronize(backend);
            reordered = std::memcmp(head.data(), w_head.data(), head.size()) != 0;
        }

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_graph_compute(backend, gf);
        }
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::steady_clock::now();

        const double ms   = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        const double gb   = (double) ggml_nbytes(w) / 1e9;
        std::vector<float> yv(c.ncols * c.batch);
        ggml_backend_tensor_get(y, yv.data(), 0, c.ncols * c.batch * sizeof(float));
        double chk = 0.0;
        for (int64_t i = 0; i < c.ncols; ++i) chk += (double) yv[i] * (double) (i % 7 - 3);
        double relerr = 0.0;
        if (have_ref) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                num += std::fabs((double) yv[i] - ref[i]);
                den += std::fabs(ref[i]);
            }
            relerr = den > 0 ? num / den : num;
        }
        std::printf("%-18s %8lld %8lld %10.3f %10.1f %8.1f  chk=%.6f relerr=%.5f reorder=%d\n", c.name,
                    (long long) c.ncols, (long long) c.nrows, ms, gb / (ms / 1000.0), ggml_nbytes(w) / 1048576.0,
                    chk, relerr, (int) reordered);

        ggml_backend_buffer_free(wbuf);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return 0;
}
