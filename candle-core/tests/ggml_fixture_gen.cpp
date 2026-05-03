#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

struct FixtureCase {
    std::string name;
    std::string op;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> rhs_shape;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<int32_t> ids;
    std::vector<int64_t> output_shape;
    std::vector<float> expected;
    std::vector<uint32_t> expected_u32;
    bool has_clamp = false;
    float clamp_min = 0.0f;
    float clamp_max = 0.0f;
};

static ggml_context * make_ctx() {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*512 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "failed to init ggml context\n";
        std::exit(1);
    }
    return ctx;
}

static std::vector<float> make_values(size_t n, float offset, float scale, bool positive_only) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        const int pattern = static_cast<int>((i * 37u + 13u) % 29u) - 14;
        float v = offset + scale * (static_cast<float>(pattern) / 7.0f);
        if (positive_only) {
            v = std::fabs(v) + 0.25f;
        }
        out[i] = v;
    }
    return out;
}

static void set_tensor_f32_or_f16(ggml_tensor * t, const std::vector<float> & data, ggml_type type) {
    if (static_cast<size_t>(ggml_nelements(t)) != data.size()) {
        std::cerr << "set_tensor_f32_or_f16: data size mismatch\n";
        std::exit(1);
    }
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        return;
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            f16[i] = ggml_fp32_to_fp16(data[i]);
        }
        ggml_backend_tensor_set(t, f16.data(), 0, f16.size() * sizeof(ggml_fp16_t));
        return;
    }
    std::cerr << "unsupported dtype in set_tensor_f32_or_f16\n";
    std::exit(1);
}

static void set_tensor_i32(ggml_tensor * t, const std::vector<int32_t> & data) {
    if (t->type != GGML_TYPE_I32) {
        std::cerr << "set_tensor_i32: tensor is not i32\n";
        std::exit(1);
    }
    if (static_cast<size_t>(ggml_nelements(t)) != data.size()) {
        std::cerr << "set_tensor_i32: data size mismatch\n";
        std::exit(1);
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int32_t));
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

    std::vector<float> out;
    out.reserve(static_cast<size_t>(ggml_nelements(t)));

    const size_t block_size = ggml_blck_size(t->type);
    std::vector<float> block_values(block_size);
    const ggml_type_traits * traits = ggml_get_type_traits(t->type);
    const bool quantized = ggml_is_quantized(t->type);

    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += static_cast<int64_t>(block_size)) {
                    const size_t byte_index = static_cast<size_t>(
                        i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + (i0 / static_cast<int64_t>(block_size)) * t->nb[0]
                    );
                    if (t->type == GGML_TYPE_F32) {
                        out.push_back(*reinterpret_cast<const float *>(&raw[byte_index]));
                    } else if (t->type == GGML_TYPE_F16) {
                        out.push_back(ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(&raw[byte_index])));
                    } else if (t->type == GGML_TYPE_I32) {
                        out.push_back(static_cast<float>(*reinterpret_cast<const int32_t *>(&raw[byte_index])));
                    } else if (t->type == GGML_TYPE_I64) {
                        out.push_back(static_cast<float>(*reinterpret_cast<const int64_t *>(&raw[byte_index])));
                    } else if (quantized) {
                        traits->to_float(&raw[byte_index], block_values.data(), block_size);
                        out.insert(out.end(), block_values.begin(), block_values.end());
                    } else {
                        std::cerr << "unsupported output dtype in tensor_to_float\n";
                        std::exit(1);
                    }
                }
            }
        }
    }

    return out;
}

static std::vector<uint32_t> tensor_to_u32(const ggml_tensor * t) {
    std::vector<float> fv = tensor_to_float(t);
    std::vector<uint32_t> out(fv.size());
    for (size_t i = 0; i < fv.size(); ++i) {
        out[i] = static_cast<uint32_t>(fv[i]);
    }
    return out;
}

static ggml_backend_buffer_t prepare_graph(
    ggml_backend_t backend,
    ggml_context * ctx,
    ggml_tensor * out,
    ggml_cgraph ** graph_out
) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::cerr << "ggml_backend_alloc_ctx_tensors failed\n";
        std::exit(1);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    *graph_out = gf;
    return buf;
}

static void compute_graph(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::cerr << "ggml_backend_graph_compute failed: " << ggml_status_to_string(st) << "\n";
        std::exit(1);
    }
}

using UnaryFn = ggml_tensor * (*)(ggml_context *, ggml_tensor *);
using BinaryFn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *);

static FixtureCase run_unary_case(
    ggml_backend_t backend,
    const std::string & name,
    UnaryFn fn,
    ggml_type type,
    bool positive_input
) {
    FixtureCase c;
    c.name = name;
    c.op = name;
    c.dtype = (type == GGML_TYPE_F16) ? "f16" : "f32";
    c.shape = {3, 4};
    c.output_shape = c.shape;
    c.a = make_values(12, 0.1f, 1.3f, positive_input);

    ggml_context * ctx = make_ctx();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 4, 3);
    ggml_tensor * out = fn(ctx, a);
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = prepare_graph(backend, ctx, out, &gf);

    set_tensor_f32_or_f16(a, c.a, type);
    compute_graph(backend, gf);

    c.expected = tensor_to_float(out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return c;
}

static FixtureCase run_binary_case(
    ggml_backend_t backend,
    const std::string & name,
    BinaryFn fn,
    ggml_type type,
    bool positive_rhs
) {
    FixtureCase c;
    c.name = name;
    c.op = name;
    c.dtype = (type == GGML_TYPE_F16) ? "f16" : "f32";
    c.shape = {3, 4};
    c.rhs_shape = {3, 4};
    c.output_shape = c.shape;
    c.a = make_values(12, -0.35f, 1.2f, false);
    c.b = make_values(12, 0.55f, 0.9f, positive_rhs);

    ggml_context * ctx = make_ctx();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 4, 3);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, type, 4, 3);
    ggml_tensor * out = fn(ctx, a, b);
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = prepare_graph(backend, ctx, out, &gf);

    set_tensor_f32_or_f16(a, c.a, type);
    set_tensor_f32_or_f16(b, c.b, type);
    compute_graph(backend, gf);

    c.expected = tensor_to_float(out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return c;
}

static FixtureCase run_clamp_case(ggml_backend_t backend, ggml_type type) {
    FixtureCase c;
    c.name = "clamp";
    c.op = "clamp";
    c.dtype = (type == GGML_TYPE_F16) ? "f16" : "f32";
    c.shape = {3, 4};
    c.output_shape = c.shape;
    c.a = make_values(12, 0.0f, 1.5f, false);
    c.has_clamp = true;
    c.clamp_min = -0.7f;
    c.clamp_max = 0.9f;

    ggml_context * ctx = make_ctx();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 4, 3);
    ggml_tensor * out = ggml_clamp(ctx, a, c.clamp_min, c.clamp_max);
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = prepare_graph(backend, ctx, out, &gf);

    set_tensor_f32_or_f16(a, c.a, type);
    compute_graph(backend, gf);

    c.expected = tensor_to_float(out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return c;
}

static FixtureCase run_sum_keepdim_last_case(ggml_backend_t backend, ggml_type type) {
    FixtureCase c;
    c.name = "sum_keepdim_last";
    c.op = "sum_keepdim_last";
    c.dtype = (type == GGML_TYPE_F16) ? "f16" : "f32";
    c.shape = {3, 4};
    c.output_shape = {3, 1};
    c.a = make_values(12, 0.25f, 1.1f, false);

    ggml_context * ctx = make_ctx();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 4, 3);
    ggml_tensor * sum_src = (type == GGML_TYPE_F16) ? ggml_cast(ctx, a, GGML_TYPE_F32) : a;
    ggml_tensor * out = ggml_sum_rows(ctx, sum_src);
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = prepare_graph(backend, ctx, out, &gf);

    set_tensor_f32_or_f16(a, c.a, type);
    compute_graph(backend, gf);

    c.expected = tensor_to_float(out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return c;
}

static FixtureCase run_mean_keepdim_last_case(ggml_backend_t backend, ggml_type type) {
    FixtureCase c;
    c.name = "mean_keepdim_last";
    c.op = "mean_keepdim_last";
    c.dtype = (type == GGML_TYPE_F16) ? "f16" : "f32";
    c.shape = {3, 4};
    c.output_shape = {3, 1};
    c.a = make_values(12, -0.1f, 0.75f, false);

    ggml_context * ctx = make_ctx();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 4, 3);
    ggml_tensor * mean_src = (type == GGML_TYPE_F16) ? ggml_cast(ctx, a, GGML_TYPE_F32) : a;
    ggml_tensor * out = ggml_mean(ctx, mean_src);
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = prepare_graph(backend, ctx, out, &gf);

    set_tensor_f32_or_f16(a, c.a, type);
    compute_graph(backend, gf);

    c.expected = tensor_to_float(out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return c;
}

static FixtureCase run_argmax_keepdim_last_case(ggml_backend_t backend, ggml_type type) {
    FixtureCase c;
    c.name = "argmax_keepdim_last";
    c.op = "argmax_keepdim_last";
    c.dtype = (type == GGML_TYPE_F16) ? "f16" : "f32";
    c.shape = {3, 4};
    c.output_shape = {3, 1};
    c.a = {
        1.0f, 3.0f, -2.0f, 0.5f,
        -1.0f, 4.2f, 4.1f, -3.0f,
        0.0f, -5.0f, 8.0f, 7.25f
    };

    ggml_context * ctx = make_ctx();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 4, 3);
    ggml_tensor * arg_src = (type == GGML_TYPE_F16) ? ggml_cast(ctx, a, GGML_TYPE_F32) : a;
    ggml_tensor * out = ggml_argmax(ctx, arg_src);
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = prepare_graph(backend, ctx, out, &gf);

    set_tensor_f32_or_f16(a, c.a, type);
    compute_graph(backend, gf);

    c.expected_u32 = tensor_to_u32(out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return c;
}

static FixtureCase run_cumsum_last_case(ggml_backend_t backend, ggml_type type) {
    FixtureCase c;
    c.name = "cumsum_last";
    c.op = "cumsum_last";
    c.dtype = (type == GGML_TYPE_F16) ? "f16" : "f32";
    c.shape = {3, 4};
    c.output_shape = c.shape;
    c.a = make_values(12, 0.2f, 0.8f, false);

    ggml_context * ctx = make_ctx();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 4, 3);
    ggml_tensor * csum_src = (type == GGML_TYPE_F16) ? ggml_cast(ctx, a, GGML_TYPE_F32) : a;
    ggml_tensor * out = ggml_cumsum(ctx, csum_src);
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = prepare_graph(backend, ctx, out, &gf);

    set_tensor_f32_or_f16(a, c.a, type);
    compute_graph(backend, gf);

    c.expected = tensor_to_float(out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return c;
}

static FixtureCase run_mul_mat_ggml_case(ggml_backend_t backend, ggml_type type) {
    FixtureCase c;
    c.name = "mul_mat_ggml";
    c.op = "mul_mat_ggml";
    c.dtype = (type == GGML_TYPE_F16) ? "f16" : "f32";
    c.shape = {2, 3};     // A in Candle layout (rows, cols)
    c.rhs_shape = {4, 3}; // B^T in Candle layout, so B = transpose(rhs)
    // ggml_mul_mat output is transposed relative to Candle matmul output.
    c.output_shape = {4, 2};
    c.a = {
        1.0f, 2.0f, -1.0f,
        0.5f, -3.0f, 4.0f
    };
    c.b = {
        1.0f, 0.0f, 2.0f,
        -1.0f, 3.0f, 0.5f,
        2.5f, -0.5f, 1.5f,
        0.0f, -2.0f, 1.0f
    };

    ggml_context * ctx = make_ctx();
    // ggml_mul_mat expects A:[k,m] and B:[k,n] -> out:[m,n]
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 3, 2);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, type, 3, 4);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = prepare_graph(backend, ctx, out, &gf);

    set_tensor_f32_or_f16(a, c.a, type);
    set_tensor_f32_or_f16(b, c.b, type);
    compute_graph(backend, gf);

    c.expected = tensor_to_float(out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return c;
}

static FixtureCase run_index_select_dim0_case(ggml_backend_t backend, ggml_type type) {
    FixtureCase c;
    c.name = "index_select_dim0";
    c.op = "index_select_dim0";
    c.dtype = (type == GGML_TYPE_F16) ? "f16" : "f32";
    c.shape = {3, 4};
    c.output_shape = {2, 4};
    c.a = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        -1.0f, -2.0f, -3.0f, -4.0f
    };
    c.ids = {2, 0};

    ggml_context * ctx = make_ctx();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 4, 3);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(c.ids.size()));
    ggml_tensor * out = ggml_get_rows(ctx, a, ids);
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = prepare_graph(backend, ctx, out, &gf);

    set_tensor_f32_or_f16(a, c.a, type);
    set_tensor_i32(ids, c.ids);
    compute_graph(backend, gf);

    c.expected = tensor_to_float(out);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return c;
}

static void print_i64_vec(std::ostream & os, const std::vector<int64_t> & v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    os << "]";
}

static void print_i32_vec(std::ostream & os, const std::vector<int32_t> & v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    os << "]";
}

static void print_u32_vec(std::ostream & os, const std::vector<uint32_t> & v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    os << "]";
}

static void print_f32_vec(std::ostream & os, const std::vector<float> & v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << std::setprecision(9) << std::fixed << v[i];
    }
    os << "]";
}

static void write_cases_json(const std::string & out_path, const std::vector<FixtureCase> & cases) {
    std::ofstream os(out_path, std::ios::binary);
    if (!os) {
        std::cerr << "failed to open output file: " << out_path << "\n";
        std::exit(1);
    }

    os << "{\n";
    os << "  \"source\": \"ggml\",\n";
    os << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const FixtureCase & c = cases[i];
        os << "    {\n";
        os << "      \"name\": \"" << c.name << "\",\n";
        os << "      \"op\": \"" << c.op << "\",\n";
        os << "      \"dtype\": \"" << c.dtype << "\",\n";
        os << "      \"shape\": ";
        print_i64_vec(os, c.shape);
        os << ",\n";
        if (!c.rhs_shape.empty()) {
            os << "      \"rhs_shape\": ";
            print_i64_vec(os, c.rhs_shape);
            os << ",\n";
        }
        os << "      \"output_shape\": ";
        print_i64_vec(os, c.output_shape);
        os << ",\n";
        os << "      \"a\": ";
        print_f32_vec(os, c.a);
        os << ",\n";
        if (!c.b.empty()) {
            os << "      \"b\": ";
            print_f32_vec(os, c.b);
            os << ",\n";
        }
        if (!c.ids.empty()) {
            os << "      \"ids\": ";
            print_i32_vec(os, c.ids);
            os << ",\n";
        }
        if (c.has_clamp) {
            os << "      \"clamp_min\": " << std::setprecision(9) << std::fixed << c.clamp_min << ",\n";
            os << "      \"clamp_max\": " << std::setprecision(9) << std::fixed << c.clamp_max << ",\n";
        }
        if (!c.expected.empty()) {
            os << "      \"expected\": ";
            print_f32_vec(os, c.expected);
            os << "\n";
        } else {
            os << "      \"expected_u32\": ";
            print_u32_vec(os, c.expected_u32);
            os << "\n";
        }
        os << "    }";
        if (i + 1 != cases.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n";
    os << "}\n";
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: ggml_fixture_gen <output-json-path>\n";
        return 1;
    }
    const std::string out_path = argv[1];

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("CPU", nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!backend) {
        std::cerr << "failed to init ggml CPU backend\n";
        return 1;
    }

    std::vector<FixtureCase> cases;
    const std::vector<ggml_type> dtypes = {GGML_TYPE_F16, GGML_TYPE_F32};
    for (ggml_type type : dtypes) {
        const char * dtype_name = (type == GGML_TYPE_F16) ? "f16" : "f32";
        auto append = [&](const std::string & label, auto fn) {
            std::cout << "[" << dtype_name << "] " << label << "... " << std::flush;
            cases.push_back(fn());
            std::cout << "ok\n";
        };

        append("neg", [&]() { return run_unary_case(backend, "neg", ggml_neg, type, false); });
        append("abs", [&]() { return run_unary_case(backend, "abs", ggml_abs, type, false); });
        append("exp", [&]() { return run_unary_case(backend, "exp", ggml_exp, type, false); });
        append("log", [&]() { return run_unary_case(backend, "log", ggml_log, type, true); });
        append("sin", [&]() { return run_unary_case(backend, "sin", ggml_sin, type, false); });
        append("cos", [&]() { return run_unary_case(backend, "cos", ggml_cos, type, false); });
        append("tanh", [&]() { return run_unary_case(backend, "tanh", ggml_tanh, type, false); });
        append("sqr", [&]() { return run_unary_case(backend, "sqr", ggml_sqr, type, false); });
        append("sqrt", [&]() { return run_unary_case(backend, "sqrt", ggml_sqrt, type, true); });
        append("relu", [&]() { return run_unary_case(backend, "relu", ggml_relu, type, false); });
        append("ceil", [&]() { return run_unary_case(backend, "ceil", ggml_ceil, type, false); });
        append("floor", [&]() { return run_unary_case(backend, "floor", ggml_floor, type, false); });
        append("round", [&]() { return run_unary_case(backend, "round", ggml_round, type, false); });
        append("sign", [&]() { return run_unary_case(backend, "sign", ggml_sgn, type, false); });

        append("add", [&]() { return run_binary_case(backend, "add", ggml_add, type, false); });
        append("sub", [&]() { return run_binary_case(backend, "sub", ggml_sub, type, false); });
        append("mul", [&]() { return run_binary_case(backend, "mul", ggml_mul, type, false); });
        append("div", [&]() { return run_binary_case(backend, "div", ggml_div, type, true); });

        append("clamp", [&]() { return run_clamp_case(backend, type); });
        append("sum_keepdim_last", [&]() { return run_sum_keepdim_last_case(backend, type); });
        append("mean_keepdim_last", [&]() { return run_mean_keepdim_last_case(backend, type); });
        append("argmax_keepdim_last", [&]() { return run_argmax_keepdim_last_case(backend, type); });
        append("cumsum_last", [&]() { return run_cumsum_last_case(backend, type); });
        append("mul_mat_ggml", [&]() { return run_mul_mat_ggml_case(backend, type); });
        append("index_select_dim0", [&]() { return run_index_select_dim0_case(backend, type); });
    }

    write_cases_json(out_path, cases);
    ggml_backend_free(backend);
    std::cout << "wrote " << cases.size() << " cases to " << out_path << "\n";
    return 0;
}
