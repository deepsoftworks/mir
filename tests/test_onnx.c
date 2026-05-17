#include "mir.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    WIRE_VARINT = 0,
    WIRE_LEN = 2
};

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} Buffer;

typedef struct {
    int passed;
    int failed;
} TestStats;

static void expect(TestStats *stats, int condition, const char *message) {
    if (condition) {
        stats->passed += 1;
        return;
    }
    stats->failed += 1;
    fprintf(stderr, "FAIL: %s\n", message);
}

static int almost_equal(float a, float b, float eps) {
    return fabsf(a - b) <= eps;
}

static void buffer_init(Buffer *buffer) {
    memset(buffer, 0, sizeof(*buffer));
}

static void buffer_free(Buffer *buffer) {
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

static int buffer_reserve(Buffer *buffer, size_t extra) {
    if (buffer->len + extra <= buffer->cap) {
        return 1;
    }
    size_t next = buffer->cap == 0 ? 128 : buffer->cap;
    while (next < buffer->len + extra) {
        next *= 2;
    }
    uint8_t *grown = (uint8_t *)realloc(buffer->data, next);
    if (!grown) {
        return 0;
    }
    buffer->data = grown;
    buffer->cap = next;
    return 1;
}

static int buffer_append(Buffer *buffer, const void *data, size_t len) {
    if (!buffer_reserve(buffer, len)) {
        return 0;
    }
    memcpy(buffer->data + buffer->len, data, len);
    buffer->len += len;
    return 1;
}

static int buffer_put_byte(Buffer *buffer, uint8_t value) {
    return buffer_append(buffer, &value, 1);
}

static int buffer_put_varint(Buffer *buffer, uint64_t value) {
    while (value >= 0x80) {
        uint8_t byte = (uint8_t)((value & 0x7f) | 0x80u);
        if (!buffer_put_byte(buffer, byte)) {
            return 0;
        }
        value >>= 7;
    }
    return buffer_put_byte(buffer, (uint8_t)value);
}

static int buffer_put_key(Buffer *buffer, uint32_t field, uint8_t wire_type) {
    uint64_t key = ((uint64_t)field << 3) | wire_type;
    return buffer_put_varint(buffer, key);
}

static int buffer_put_varint_field(Buffer *buffer, uint32_t field, uint64_t value) {
    return buffer_put_key(buffer, field, WIRE_VARINT) && buffer_put_varint(buffer, value);
}

static int buffer_put_bytes_field(Buffer *buffer, uint32_t field, const uint8_t *data, size_t len) {
    return buffer_put_key(buffer, field, WIRE_LEN) &&
           buffer_put_varint(buffer, (uint64_t)len) &&
           buffer_append(buffer, data, len);
}

static int buffer_put_string_field(Buffer *buffer, uint32_t field, const char *text) {
    return buffer_put_bytes_field(buffer, field, (const uint8_t *)text, strlen(text));
}

static int buffer_put_message_field(Buffer *buffer, uint32_t field, const Buffer *message) {
    return buffer_put_bytes_field(buffer, field, message->data, message->len);
}

static int encode_f32_le(float value, uint8_t out[4]) {
    uint32_t raw = 0;
    memcpy(&raw, &value, sizeof(raw));
    out[0] = (uint8_t)(raw & 0xffu);
    out[1] = (uint8_t)((raw >> 8) & 0xffu);
    out[2] = (uint8_t)((raw >> 16) & 0xffu);
    out[3] = (uint8_t)((raw >> 24) & 0xffu);
    return 1;
}

static void encode_i64_le(int64_t value, uint8_t out[8]) {
    uint64_t raw = (uint64_t)value;
    for (unsigned int i = 0; i < 8; ++i) {
        out[i] = (uint8_t)((raw >> (8u * i)) & 0xffu);
    }
}

static int build_tensor_f32(
    Buffer *out,
    const char *name,
    const int64_t *dims,
    size_t rank,
    const float *values,
    size_t count
) {
    buffer_init(out);
    for (size_t i = 0; i < rank; ++i) {
        if (!buffer_put_varint_field(out, 1, (uint64_t)dims[i])) {
            return 0;
        }
    }
    if (!buffer_put_varint_field(out, 2, 1)) {
        return 0;
    }
    if (!buffer_put_string_field(out, 8, name)) {
        return 0;
    }

    Buffer raw;
    buffer_init(&raw);
    for (size_t i = 0; i < count; ++i) {
        uint8_t bytes[4];
        encode_f32_le(values[i], bytes);
        if (!buffer_append(&raw, bytes, sizeof(bytes))) {
            buffer_free(&raw);
            return 0;
        }
    }
    int ok = buffer_put_message_field(out, 9, &raw);
    buffer_free(&raw);
    return ok;
}

static int build_tensor_i64(
    Buffer *out,
    const char *name,
    const int64_t *dims,
    size_t rank,
    const int64_t *values,
    size_t count
) {
    buffer_init(out);
    for (size_t i = 0; i < rank; ++i) {
        if (!buffer_put_varint_field(out, 1, (uint64_t)dims[i])) {
            return 0;
        }
    }
    if (!buffer_put_varint_field(out, 2, 7)) {
        return 0;
    }
    if (!buffer_put_string_field(out, 8, name)) {
        return 0;
    }

    Buffer raw;
    buffer_init(&raw);
    for (size_t i = 0; i < count; ++i) {
        uint8_t bytes[8];
        encode_i64_le(values[i], bytes);
        if (!buffer_append(&raw, bytes, sizeof(bytes))) {
            buffer_free(&raw);
            return 0;
        }
    }
    int ok = buffer_put_message_field(out, 9, &raw);
    buffer_free(&raw);
    return ok;
}

static int build_value_info_f32(Buffer *out, const char *name, const int64_t *dims, size_t rank) {
    buffer_init(out);
    if (!buffer_put_string_field(out, 1, name)) {
        return 0;
    }

    Buffer shape;
    buffer_init(&shape);
    for (size_t i = 0; i < rank; ++i) {
        Buffer dim;
        buffer_init(&dim);
        if (!buffer_put_varint_field(&dim, 1, (uint64_t)dims[i])) {
            buffer_free(&dim);
            buffer_free(&shape);
            return 0;
        }
        if (!buffer_put_message_field(&shape, 1, &dim)) {
            buffer_free(&dim);
            buffer_free(&shape);
            return 0;
        }
        buffer_free(&dim);
    }

    Buffer tensor_type;
    buffer_init(&tensor_type);
    if (!buffer_put_varint_field(&tensor_type, 1, 1) || !buffer_put_message_field(&tensor_type, 2, &shape)) {
        buffer_free(&shape);
        buffer_free(&tensor_type);
        return 0;
    }
    buffer_free(&shape);

    Buffer type_proto;
    buffer_init(&type_proto);
    if (!buffer_put_message_field(&type_proto, 1, &tensor_type)) {
        buffer_free(&tensor_type);
        buffer_free(&type_proto);
        return 0;
    }
    buffer_free(&tensor_type);

    int ok = buffer_put_message_field(out, 2, &type_proto);
    buffer_free(&type_proto);
    return ok;
}

static int build_attr_i(Buffer *out, const char *name, int64_t value) {
    buffer_init(out);
    return buffer_put_string_field(out, 1, name) && buffer_put_varint_field(out, 4, (uint64_t)value);
}

static int build_attr_ints(Buffer *out, const char *name, const int64_t *values, size_t count) {
    buffer_init(out);
    if (!buffer_put_string_field(out, 1, name)) {
        return 0;
    }
    Buffer packed;
    buffer_init(&packed);
    for (size_t i = 0; i < count; ++i) {
        if (!buffer_put_varint(&packed, (uint64_t)values[i])) {
            buffer_free(&packed);
            return 0;
        }
    }
    int ok = buffer_put_message_field(out, 9, &packed);
    buffer_free(&packed);
    return ok;
}

static int build_node(
    Buffer *out,
    const char *op_type,
    const char **inputs,
    size_t input_count,
    const char **outputs,
    size_t output_count,
    const Buffer *attrs,
    size_t attr_count
) {
    buffer_init(out);
    for (size_t i = 0; i < input_count; ++i) {
        if (!buffer_put_string_field(out, 1, inputs[i])) {
            return 0;
        }
    }
    for (size_t i = 0; i < output_count; ++i) {
        if (!buffer_put_string_field(out, 2, outputs[i])) {
            return 0;
        }
    }
    if (!buffer_put_string_field(out, 4, op_type)) {
        return 0;
    }
    for (size_t i = 0; i < attr_count; ++i) {
        if (!buffer_put_message_field(out, 5, &attrs[i])) {
            return 0;
        }
    }
    return 1;
}

static int build_sample_model(Buffer *out) {
    Buffer input_vi;
    Buffer output_vi;
    Buffer weight;
    Buffer bias;
    Buffer reshape_shape;
    Buffer nodes[6];
    Buffer graph;
    Buffer opset;
    buffer_init(&input_vi);
    buffer_init(&output_vi);
    buffer_init(&weight);
    buffer_init(&bias);
    buffer_init(&reshape_shape);
    buffer_init(&graph);
    buffer_init(&opset);
    for (size_t i = 0; i < 6; ++i) {
        buffer_init(&nodes[i]);
    }

    int64_t x_dims[2] = {1, 3};
    int64_t y_dims[2] = {1, 2};
    int64_t w_dims[2] = {3, 2};
    int64_t b_dims[2] = {1, 2};
    int64_t rs_dims[1] = {2};
    float w_values[6] = {0.5f, 1.0f, -0.5f, 0.25f, 1.0f, -1.0f};
    float b_values[2] = {0.1f, -0.2f};
    int64_t rs_values[2] = {2, 1};

    int ok = build_value_info_f32(&input_vi, "X", x_dims, 2) &&
             build_value_info_f32(&output_vi, "Y", y_dims, 2) &&
             build_tensor_f32(&weight, "W", w_dims, 2, w_values, 6) &&
             build_tensor_f32(&bias, "B", b_dims, 2, b_values, 2) &&
             build_tensor_i64(&reshape_shape, "RSShape", rs_dims, 1, rs_values, 2);
    if (!ok) {
        goto cleanup;
    }

    const char *n0_in[] = {"X", "W"};
    const char *n0_out[] = {"MM"};
    const char *n1_in[] = {"MM", "B"};
    const char *n1_out[] = {"A"};
    const char *n2_in[] = {"A"};
    const char *n2_out[] = {"R"};
    const char *n3_in[] = {"R"};
    const char *n3_out[] = {"S"};
    const char *n4_in[] = {"S", "RSShape"};
    const char *n4_out[] = {"RS"};
    const char *n5_in[] = {"RS"};
    const char *n5_out[] = {"Y"};

    Buffer softmax_axis;
    Buffer transpose_perm;
    buffer_init(&softmax_axis);
    buffer_init(&transpose_perm);
    int64_t perm_vals[2] = {1, 0};
    ok = build_attr_i(&softmax_axis, "axis", 1) &&
         build_attr_ints(&transpose_perm, "perm", perm_vals, 2) &&
         build_node(&nodes[0], "MatMul", n0_in, 2, n0_out, 1, NULL, 0) &&
         build_node(&nodes[1], "Add", n1_in, 2, n1_out, 1, NULL, 0) &&
         build_node(&nodes[2], "Relu", n2_in, 1, n2_out, 1, NULL, 0) &&
         build_node(&nodes[3], "Softmax", n3_in, 1, n3_out, 1, &softmax_axis, 1) &&
         build_node(&nodes[4], "Reshape", n4_in, 2, n4_out, 1, NULL, 0) &&
         build_node(&nodes[5], "Transpose", n5_in, 1, n5_out, 1, &transpose_perm, 1);
    buffer_free(&softmax_axis);
    buffer_free(&transpose_perm);
    if (!ok) {
        goto cleanup;
    }

    ok = buffer_put_message_field(&graph, 11, &input_vi) &&
         buffer_put_message_field(&graph, 12, &output_vi) &&
         buffer_put_message_field(&graph, 5, &weight) &&
         buffer_put_message_field(&graph, 5, &bias) &&
         buffer_put_message_field(&graph, 5, &reshape_shape);
    for (size_t i = 0; i < 6 && ok; ++i) {
        ok = buffer_put_message_field(&graph, 1, &nodes[i]);
    }
    if (!ok) {
        goto cleanup;
    }

    ok = buffer_put_varint_field(&opset, 2, 13);
    if (!ok) {
        goto cleanup;
    }

    buffer_init(out);
    ok = buffer_put_message_field(out, 8, &opset) && buffer_put_message_field(out, 7, &graph);

cleanup:
    buffer_free(&input_vi);
    buffer_free(&output_vi);
    buffer_free(&weight);
    buffer_free(&bias);
    buffer_free(&reshape_shape);
    buffer_free(&graph);
    buffer_free(&opset);
    for (size_t i = 0; i < 6; ++i) {
        buffer_free(&nodes[i]);
    }
    return ok;
}

static void test_onnx_load_and_execute(TestStats *stats) {
    Buffer model;
    buffer_init(&model);
    int built = build_sample_model(&model);
    expect(stats, built, "sample ONNX model builds");
    if (!built) {
        buffer_free(&model);
        return;
    }

    MirGraph graph;
    memset(&graph, 0, sizeof(graph));
    MirStatus status = mir_onnx_load_buffer(model.data, model.len, &graph);
    expect(stats, status == MIR_OK, "onnx buffer parse succeeds");
    if (status != MIR_OK) {
        buffer_free(&model);
        return;
    }

    expect(stats, graph.node_count == 6, "onnx graph node count");

    size_t input_id = 0;
    size_t output_id = 0;
    status = mir_graph_find_tensor(&graph, "X", &input_id);
    expect(stats, status == MIR_OK, "input tensor found by name");
    status = mir_graph_find_tensor(&graph, "Y", &output_id);
    expect(stats, status == MIR_OK, "output tensor found by name");

    MirTensor *input = mir_graph_tensor(&graph, input_id);
    expect(stats, input != NULL, "input tensor exists");
    if (!input) {
        mir_graph_free(&graph);
        buffer_free(&model);
        return;
    }

    status = mir_tensor_resize(input, input->shape);
    expect(stats, status == MIR_OK, "input tensor allocates");
    if (status == MIR_OK) {
        input->data[0] = 1.0f;
        input->data[1] = 2.0f;
        input->data[2] = -1.0f;
    }

    status = mir_execute_graph(&graph, NULL, NULL);
    expect(stats, status == MIR_OK, "parsed graph executes");

    MirTensor *output = mir_graph_tensor(&graph, output_id);
    expect(stats, output != NULL, "output tensor exists");
    if (output) {
        expect(stats, output->shape.rank == 2, "output rank");
        expect(stats, output->shape.dims[0] == 1 && output->shape.dims[1] == 2, "output shape");
        expect(stats, almost_equal(output->data[0], 0.09112296f, 1e-5f), "output value [0]");
        expect(stats, almost_equal(output->data[1], 0.90887702f, 1e-5f), "output value [1]");
    }

    const char *tmp_path = "/tmp/mir_sample_model.onnx";
    FILE *tmp = fopen(tmp_path, "wb");
    expect(stats, tmp != NULL, "temporary ONNX file opens");
    if (tmp) {
        size_t written = fwrite(model.data, 1, model.len, tmp);
        fclose(tmp);
        expect(stats, written == model.len, "temporary ONNX file writes");

        MirGraph file_graph;
        memset(&file_graph, 0, sizeof(file_graph));
        status = mir_onnx_load_file(tmp_path, &file_graph);
        expect(stats, status == MIR_OK, "onnx file parse succeeds");
        if (status == MIR_OK) {
            expect(stats, file_graph.node_count == 6, "onnx file node count");
            mir_graph_free(&file_graph);
        }
        remove(tmp_path);
    }

    mir_graph_free(&graph);
    buffer_free(&model);
}

int main(void) {
    TestStats stats = {0, 0};
    test_onnx_load_and_execute(&stats);
    printf("onnx_parser passed=%d failed=%d\n", stats.passed, stats.failed);
    return stats.failed == 0 ? 0 : 1;
}
