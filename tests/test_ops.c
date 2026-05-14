#include "mir.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    int passed;
    int failed;
} TestStats;

static int almost_equal(float a, float b, float eps) {
    return fabsf(a - b) <= eps;
}

static void expect(int condition, const char *message, TestStats *stats) {
    if (condition) {
        stats->passed += 1;
        return;
    }
    stats->failed += 1;
    fprintf(stderr, "FAIL: %s\n", message);
}

static MirTensor make_tensor_or_die(const int64_t *dims, uint32_t rank) {
    MirTensor tensor;
    MirShape shape;
    MirStatus status = mir_shape_make(dims, rank, &shape);
    if (status != MIR_OK) {
        fprintf(stderr, "shape setup failed: %s\n", mir_status_str(status));
        memset(&tensor, 0, sizeof(tensor));
        return tensor;
    }
    status = mir_tensor_init(&tensor, shape, true);
    if (status != MIR_OK) {
        fprintf(stderr, "tensor setup failed: %s\n", mir_status_str(status));
        memset(&tensor, 0, sizeof(tensor));
    }
    return tensor;
}

static void test_matmul(TestStats *stats) {
    int64_t a_dims[2] = {2, 3};
    int64_t b_dims[2] = {3, 2};
    MirTensor a = make_tensor_or_die(a_dims, 2);
    MirTensor b = make_tensor_or_die(b_dims, 2);
    MirTensor out = {0};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {1, 2, 3, 4, 5, 6};
    memcpy(a.data, a_data, sizeof(a_data));
    memcpy(b.data, b_data, sizeof(b_data));

    MirStatus status = mir_op_matmul(&a, &b, &out);
    expect(status == MIR_OK, "matmul returns ok", stats);
    expect(out.shape.rank == 2 && out.shape.dims[0] == 2 && out.shape.dims[1] == 2, "matmul shape", stats);
    expect(almost_equal(out.data[0], 22.0f, 1e-5f), "matmul value [0]", stats);
    expect(almost_equal(out.data[1], 28.0f, 1e-5f), "matmul value [1]", stats);
    expect(almost_equal(out.data[2], 49.0f, 1e-5f), "matmul value [2]", stats);
    expect(almost_equal(out.data[3], 64.0f, 1e-5f), "matmul value [3]", stats);

    int64_t bad_dims[2] = {4, 1};
    MirTensor bad = make_tensor_or_die(bad_dims, 2);
    status = mir_op_matmul(&a, &bad, &out);
    expect(status == MIR_ERR_SHAPE_MISMATCH, "matmul shape mismatch", stats);

    mir_tensor_free(&a);
    mir_tensor_free(&b);
    mir_tensor_free(&bad);
    mir_tensor_free(&out);
}

static void test_add(TestStats *stats) {
    int64_t dims[2] = {2, 2};
    MirTensor a = make_tensor_or_die(dims, 2);
    MirTensor b = make_tensor_or_die(dims, 2);
    MirTensor out = {0};

    float a_data[4] = {1.0f, -2.0f, 3.5f, 0.5f};
    float b_data[4] = {0.5f, 2.0f, -1.5f, 0.25f};
    memcpy(a.data, a_data, sizeof(a_data));
    memcpy(b.data, b_data, sizeof(b_data));

    MirStatus status = mir_op_add(&a, &b, &out);
    expect(status == MIR_OK, "add returns ok", stats);
    expect(almost_equal(out.data[0], 1.5f, 1e-6f), "add value [0]", stats);
    expect(almost_equal(out.data[1], 0.0f, 1e-6f), "add value [1]", stats);
    expect(almost_equal(out.data[2], 2.0f, 1e-6f), "add value [2]", stats);
    expect(almost_equal(out.data[3], 0.75f, 1e-6f), "add value [3]", stats);

    int64_t bad_dims[1] = {4};
    MirTensor bad = make_tensor_or_die(bad_dims, 1);
    status = mir_op_add(&a, &bad, &out);
    expect(status == MIR_ERR_SHAPE_MISMATCH, "add shape mismatch", stats);

    mir_tensor_free(&a);
    mir_tensor_free(&b);
    mir_tensor_free(&bad);
    mir_tensor_free(&out);
}

static void test_relu(TestStats *stats) {
    int64_t dims[1] = {5};
    MirTensor input = make_tensor_or_die(dims, 1);
    MirTensor out = {0};
    float values[5] = {-1.0f, 0.0f, 2.0f, -3.0f, 4.0f};
    memcpy(input.data, values, sizeof(values));

    MirStatus status = mir_op_relu(&input, &out);
    expect(status == MIR_OK, "relu returns ok", stats);
    expect(almost_equal(out.data[0], 0.0f, 1e-6f), "relu value [0]", stats);
    expect(almost_equal(out.data[1], 0.0f, 1e-6f), "relu value [1]", stats);
    expect(almost_equal(out.data[2], 2.0f, 1e-6f), "relu value [2]", stats);
    expect(almost_equal(out.data[3], 0.0f, 1e-6f), "relu value [3]", stats);
    expect(almost_equal(out.data[4], 4.0f, 1e-6f), "relu value [4]", stats);

    mir_tensor_free(&input);
    mir_tensor_free(&out);
}

static void test_softmax(TestStats *stats) {
    int64_t dims[2] = {1, 3};
    MirTensor input = make_tensor_or_die(dims, 2);
    MirTensor out = {0};
    input.data[0] = 1.0f;
    input.data[1] = 2.0f;
    input.data[2] = 3.0f;

    MirStatus status = mir_op_softmax(&input, 1, &out);
    expect(status == MIR_OK, "softmax returns ok", stats);
    expect(almost_equal(out.data[0], 0.09003057f, 1e-5f), "softmax value [0]", stats);
    expect(almost_equal(out.data[1], 0.24472847f, 1e-5f), "softmax value [1]", stats);
    expect(almost_equal(out.data[2], 0.66524096f, 1e-5f), "softmax value [2]", stats);

    status = mir_op_softmax(&input, 4, &out);
    expect(status == MIR_ERR_INVALID_ARGUMENT, "softmax invalid axis", stats);

    mir_tensor_free(&input);
    mir_tensor_free(&out);
}

static void test_reshape(TestStats *stats) {
    int64_t in_dims[2] = {2, 3};
    MirTensor input = make_tensor_or_die(in_dims, 2);
    MirTensor out = {0};
    for (int i = 0; i < 6; ++i) {
        input.data[i] = (float)(i + 1);
    }

    MirShape new_shape;
    int64_t new_dims[2] = {3, 2};
    MirStatus status = mir_shape_make(new_dims, 2, &new_shape);
    expect(status == MIR_OK, "reshape shape setup", stats);
    status = mir_op_reshape(&input, new_shape, &out);
    expect(status == MIR_OK, "reshape returns ok", stats);
    expect(out.data == input.data, "reshape shares storage", stats);
    expect(out.shape.rank == 2 && out.shape.dims[0] == 3 && out.shape.dims[1] == 2, "reshape shape", stats);

    int64_t bad_dims[2] = {4, 2};
    status = mir_shape_make(bad_dims, 2, &new_shape);
    expect(status == MIR_OK, "reshape bad shape setup", stats);
    status = mir_op_reshape(&input, new_shape, &out);
    expect(status == MIR_ERR_SHAPE_MISMATCH, "reshape shape mismatch", stats);

    mir_tensor_free(&input);
    mir_tensor_free(&out);
}

static void test_transpose(TestStats *stats) {
    int64_t in_dims[2] = {2, 3};
    MirTensor input = make_tensor_or_die(in_dims, 2);
    MirTensor out = {0};

    float values[6] = {1, 2, 3, 4, 5, 6};
    memcpy(input.data, values, sizeof(values));
    uint32_t perm[2] = {1, 0};
    MirStatus status = mir_op_transpose(&input, perm, 2, &out);
    expect(status == MIR_OK, "transpose returns ok", stats);
    expect(out.shape.rank == 2 && out.shape.dims[0] == 3 && out.shape.dims[1] == 2, "transpose shape", stats);
    expect(almost_equal(out.data[0], 1.0f, 1e-6f), "transpose value [0]", stats);
    expect(almost_equal(out.data[1], 4.0f, 1e-6f), "transpose value [1]", stats);
    expect(almost_equal(out.data[2], 2.0f, 1e-6f), "transpose value [2]", stats);
    expect(almost_equal(out.data[3], 5.0f, 1e-6f), "transpose value [3]", stats);
    expect(almost_equal(out.data[4], 3.0f, 1e-6f), "transpose value [4]", stats);
    expect(almost_equal(out.data[5], 6.0f, 1e-6f), "transpose value [5]", stats);

    uint32_t bad_perm[2] = {0, 0};
    status = mir_op_transpose(&input, bad_perm, 2, &out);
    expect(status == MIR_ERR_INVALID_ARGUMENT, "transpose invalid perm", stats);

    mir_tensor_free(&input);
    mir_tensor_free(&out);
}

static void test_runtime_path(TestStats *stats) {
    MirGraph graph;
    MirStatus status = mir_graph_init(&graph, 12, 8);
    expect(status == MIR_OK, "runtime graph init", stats);
    if (status != MIR_OK) {
        return;
    }

    int64_t in_dims[2] = {1, 3};
    int64_t w_dims[2] = {3, 2};
    int64_t b_dims[2] = {1, 2};
    MirTensor input = make_tensor_or_die(in_dims, 2);
    MirTensor weight = make_tensor_or_die(w_dims, 2);
    MirTensor bias = make_tensor_or_die(b_dims, 2);

    input.data[0] = 1.0f;
    input.data[1] = 2.0f;
    input.data[2] = -1.0f;
    weight.data[0] = 0.5f;
    weight.data[1] = 1.0f;
    weight.data[2] = -0.5f;
    weight.data[3] = 0.25f;
    weight.data[4] = 1.0f;
    weight.data[5] = -1.0f;
    bias.data[0] = 0.1f;
    bias.data[1] = -0.2f;

    size_t t_in;
    size_t t_w;
    size_t t_b;
    size_t t_mm;
    size_t t_add;
    size_t t_relu;
    size_t t_sm;
    size_t t_rs;
    size_t t_out;
    mir_graph_add_tensor(&graph, &input, &t_in);
    mir_graph_add_tensor(&graph, &weight, &t_w);
    mir_graph_add_tensor(&graph, &bias, &t_b);
    mir_graph_add_empty_tensor(&graph, &t_mm);
    mir_graph_add_empty_tensor(&graph, &t_add);
    mir_graph_add_empty_tensor(&graph, &t_relu);
    mir_graph_add_empty_tensor(&graph, &t_sm);
    mir_graph_add_empty_tensor(&graph, &t_rs);
    mir_graph_add_empty_tensor(&graph, &t_out);

    MirNode node = {0};
    node.op = MIR_OP_MATMUL;
    node.input_count = 2;
    node.output_count = 1;
    node.inputs[0] = (uint32_t)t_in;
    node.inputs[1] = (uint32_t)t_w;
    node.outputs[0] = (uint32_t)t_mm;
    mir_graph_add_node(&graph, &node, NULL);

    node.op = MIR_OP_ADD;
    node.inputs[0] = (uint32_t)t_mm;
    node.inputs[1] = (uint32_t)t_b;
    node.outputs[0] = (uint32_t)t_add;
    mir_graph_add_node(&graph, &node, NULL);

    node.op = MIR_OP_RELU;
    node.input_count = 1;
    node.inputs[0] = (uint32_t)t_add;
    node.outputs[0] = (uint32_t)t_relu;
    mir_graph_add_node(&graph, &node, NULL);

    node.op = MIR_OP_SOFTMAX;
    node.attrs.axis = 1;
    node.inputs[0] = (uint32_t)t_relu;
    node.outputs[0] = (uint32_t)t_sm;
    mir_graph_add_node(&graph, &node, NULL);

    int64_t rs_dims[2] = {2, 1};
    mir_shape_make(rs_dims, 2, &node.attrs.reshape_shape);
    node.op = MIR_OP_RESHAPE;
    node.inputs[0] = (uint32_t)t_sm;
    node.outputs[0] = (uint32_t)t_rs;
    mir_graph_add_node(&graph, &node, NULL);

    node.op = MIR_OP_TRANSPOSE;
    node.attrs.perm_rank = 2;
    node.attrs.perm[0] = 1;
    node.attrs.perm[1] = 0;
    node.inputs[0] = (uint32_t)t_rs;
    node.outputs[0] = (uint32_t)t_out;
    mir_graph_add_node(&graph, &node, NULL);

    MirProfileEntry entries[8];
    MirProfileBuffer profile;
    profile.entries = entries;
    profile.capacity = 8;
    profile.count = 0;
    status = mir_execute_graph(&graph, NULL, &profile);
    expect(status == MIR_OK, "runtime execute", stats);
    expect(profile.count == 6, "runtime profile count", stats);

    MirTensor *output = mir_graph_tensor(&graph, t_out);
    expect(output != NULL, "runtime output exists", stats);
    if (output) {
        expect(output->shape.rank == 2 && output->shape.dims[0] == 1 && output->shape.dims[1] == 2, "runtime output shape", stats);
        expect(almost_equal(output->data[0], 0.09112296f, 1e-5f), "runtime output value [0]", stats);
        expect(almost_equal(output->data[1], 0.90887702f, 1e-5f), "runtime output value [1]", stats);
    }

    mir_graph_free(&graph);
}

int main(void) {
    TestStats stats = {0, 0};
    test_matmul(&stats);
    test_add(&stats);
    test_relu(&stats);
    test_softmax(&stats);
    test_reshape(&stats);
    test_transpose(&stats);
    test_runtime_path(&stats);

    printf("passed=%d failed=%d\n", stats.passed, stats.failed);
    return stats.failed == 0 ? 0 : 1;
}
