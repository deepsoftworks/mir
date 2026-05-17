#include "mir.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ONNX_WIRE_VARINT = 0,
    ONNX_WIRE_64BIT = 1,
    ONNX_WIRE_LEN = 2,
    ONNX_WIRE_32BIT = 5
};

enum {
    ONNX_DTYPE_FLOAT = 1,
    ONNX_DTYPE_INT64 = 7
};

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t off;
} PbCursor;

typedef struct {
    const uint8_t *data;
    size_t len;
} ByteSlice;

typedef struct {
    char *name;
    size_t tensor_id;
} NameEntry;

typedef struct {
    NameEntry *items;
    size_t count;
    size_t capacity;
} NameTable;

typedef struct {
    char *name;
    int64_t *values;
    size_t count;
} ShapeEntry;

typedef struct {
    ShapeEntry *items;
    size_t count;
    size_t capacity;
} ShapeTable;

typedef struct {
    ByteSlice name;
    int32_t dtype;
    int64_t dims[MIR_MAX_DIMS];
    uint32_t rank;
    ByteSlice raw_data;
    float *float_values;
    size_t float_count;
    size_t float_capacity;
    int64_t *int64_values;
    size_t int64_count;
    size_t int64_capacity;
} TensorProtoView;

typedef struct {
    ByteSlice op_type;
    ByteSlice inputs[8];
    size_t input_count;
    ByteSlice outputs[4];
    size_t output_count;
    bool axis_set;
    int64_t axis_value;
    bool perm_set;
    uint32_t perm[MIR_MAX_DIMS];
    uint32_t perm_rank;
} NodeProtoView;

typedef struct {
    ByteSlice graph_proto;
    int64_t default_opset;
} ModelProtoView;

static bool pb_read_varint(PbCursor *cursor, uint64_t *out_value) {
    uint64_t value = 0;
    unsigned int shift = 0;
    while (cursor->off < cursor->size && shift < 64) {
        uint8_t byte = cursor->data[cursor->off++];
        value |= ((uint64_t)(byte & 0x7f)) << shift;
        if ((byte & 0x80u) == 0) {
            *out_value = value;
            return true;
        }
        shift += 7;
    }
    return false;
}

static bool pb_read_key(PbCursor *cursor, uint32_t *out_field, uint8_t *out_wire) {
    uint64_t key = 0;
    if (!pb_read_varint(cursor, &key) || key == 0) {
        return false;
    }
    *out_field = (uint32_t)(key >> 3);
    *out_wire = (uint8_t)(key & 0x7u);
    return true;
}

static bool pb_read_bytes(PbCursor *cursor, ByteSlice *out_slice) {
    uint64_t len = 0;
    if (!pb_read_varint(cursor, &len)) {
        return false;
    }
    if (len > (uint64_t)(cursor->size - cursor->off)) {
        return false;
    }
    out_slice->data = cursor->data + cursor->off;
    out_slice->len = (size_t)len;
    cursor->off += (size_t)len;
    return true;
}

static bool pb_skip(PbCursor *cursor, uint8_t wire_type) {
    ByteSlice bytes = {0};
    uint64_t value = 0;
    switch (wire_type) {
    case ONNX_WIRE_VARINT:
        return pb_read_varint(cursor, &value);
    case ONNX_WIRE_64BIT:
        if (cursor->off + 8 > cursor->size) {
            return false;
        }
        cursor->off += 8;
        return true;
    case ONNX_WIRE_LEN:
        return pb_read_bytes(cursor, &bytes);
    case ONNX_WIRE_32BIT:
        if (cursor->off + 4 > cursor->size) {
            return false;
        }
        cursor->off += 4;
        return true;
    default:
        return false;
    }
}

static bool onnx_slice_eq(ByteSlice slice, const char *text) {
    size_t text_len = strlen(text);
    if (slice.len != text_len) {
        return false;
    }
    return text_len == 0 || memcmp(slice.data, text, text_len) == 0;
}

static char *onnx_copy_string(ByteSlice slice) {
    char *out = (char *)malloc(slice.len + 1);
    if (!out) {
        return NULL;
    }
    if (slice.len > 0) {
        memcpy(out, slice.data, slice.len);
    }
    out[slice.len] = '\0';
    return out;
}

static bool onnx_name_equals_slice(const char *name, ByteSlice slice) {
    size_t name_len = strlen(name);
    if (name_len != slice.len) {
        return false;
    }
    return name_len == 0 || memcmp(name, slice.data, name_len) == 0;
}

static bool onnx_name_table_lookup(const NameTable *table, ByteSlice name, size_t *out_id) {
    for (size_t i = 0; i < table->count; ++i) {
        if (onnx_name_equals_slice(table->items[i].name, name)) {
            *out_id = table->items[i].tensor_id;
            return true;
        }
    }
    return false;
}

static MirStatus onnx_name_table_add(NameTable *table, char *name, size_t tensor_id) {
    if (table->count == table->capacity) {
        size_t next = table->capacity == 0 ? 16 : table->capacity * 2;
        NameEntry *grown = (NameEntry *)realloc(table->items, next * sizeof(NameEntry));
        if (!grown) {
            return MIR_ERR_OUT_OF_MEMORY;
        }
        table->items = grown;
        table->capacity = next;
    }
    table->items[table->count].name = name;
    table->items[table->count].tensor_id = tensor_id;
    table->count += 1;
    return MIR_OK;
}

static const ShapeEntry *onnx_shape_table_lookup(const ShapeTable *table, ByteSlice name) {
    for (size_t i = 0; i < table->count; ++i) {
        if (onnx_name_equals_slice(table->items[i].name, name)) {
            return &table->items[i];
        }
    }
    return NULL;
}

static MirStatus onnx_shape_table_add(ShapeTable *table, ByteSlice name, const int64_t *values, size_t count) {
    if (table->count == table->capacity) {
        size_t next = table->capacity == 0 ? 8 : table->capacity * 2;
        ShapeEntry *grown = (ShapeEntry *)realloc(table->items, next * sizeof(ShapeEntry));
        if (!grown) {
            return MIR_ERR_OUT_OF_MEMORY;
        }
        table->items = grown;
        table->capacity = next;
    }

    char *name_copy = onnx_copy_string(name);
    if (!name_copy) {
        return MIR_ERR_OUT_OF_MEMORY;
    }

    int64_t *copy = NULL;
    if (count > 0) {
        copy = (int64_t *)malloc(count * sizeof(int64_t));
        if (!copy) {
            free(name_copy);
            return MIR_ERR_OUT_OF_MEMORY;
        }
        memcpy(copy, values, count * sizeof(int64_t));
    }

    table->items[table->count].name = name_copy;
    table->items[table->count].values = copy;
    table->items[table->count].count = count;
    table->count += 1;
    return MIR_OK;
}

static void onnx_name_table_free(NameTable *table) {
    free(table->items);
    memset(table, 0, sizeof(*table));
}

static void onnx_shape_table_free(ShapeTable *table) {
    for (size_t i = 0; i < table->count; ++i) {
        free(table->items[i].name);
        free(table->items[i].values);
    }
    free(table->items);
    memset(table, 0, sizeof(*table));
}

static MirStatus onnx_push_float(TensorProtoView *view, float value) {
    if (view->float_count == view->float_capacity) {
        size_t next = view->float_capacity == 0 ? 16 : view->float_capacity * 2;
        float *grown = (float *)realloc(view->float_values, next * sizeof(float));
        if (!grown) {
            return MIR_ERR_OUT_OF_MEMORY;
        }
        view->float_values = grown;
        view->float_capacity = next;
    }
    view->float_values[view->float_count++] = value;
    return MIR_OK;
}

static MirStatus onnx_push_int64(TensorProtoView *view, int64_t value) {
    if (view->int64_count == view->int64_capacity) {
        size_t next = view->int64_capacity == 0 ? 16 : view->int64_capacity * 2;
        int64_t *grown = (int64_t *)realloc(view->int64_values, next * sizeof(int64_t));
        if (!grown) {
            return MIR_ERR_OUT_OF_MEMORY;
        }
        view->int64_values = grown;
        view->int64_capacity = next;
    }
    view->int64_values[view->int64_count++] = value;
    return MIR_OK;
}

static float onnx_read_f32_le(const uint8_t *bytes) {
    uint32_t raw = ((uint32_t)bytes[0]) |
                   ((uint32_t)bytes[1] << 8) |
                   ((uint32_t)bytes[2] << 16) |
                   ((uint32_t)bytes[3] << 24);
    float out = 0.0f;
    memcpy(&out, &raw, sizeof(out));
    return out;
}

static int64_t onnx_read_i64_le(const uint8_t *bytes) {
    uint64_t raw = 0;
    for (unsigned int i = 0; i < 8; ++i) {
        raw |= ((uint64_t)bytes[i]) << (8u * i);
    }
    return (int64_t)raw;
}

static MirStatus onnx_parse_dims_field(TensorProtoView *view, uint8_t wire, PbCursor *cursor) {
    uint64_t value = 0;
    ByteSlice packed = {0};
    if (wire == ONNX_WIRE_VARINT) {
        if (!pb_read_varint(cursor, &value)) {
            return MIR_ERR_RUNTIME;
        }
        if (view->rank >= MIR_MAX_DIMS) {
            return MIR_ERR_UNSUPPORTED;
        }
        view->dims[view->rank++] = (int64_t)value;
        return MIR_OK;
    }
    if (wire != ONNX_WIRE_LEN || !pb_read_bytes(cursor, &packed)) {
        return MIR_ERR_RUNTIME;
    }
    PbCursor sub = {packed.data, packed.len, 0};
    while (sub.off < sub.size) {
        if (!pb_read_varint(&sub, &value)) {
            return MIR_ERR_RUNTIME;
        }
        if (view->rank >= MIR_MAX_DIMS) {
            return MIR_ERR_UNSUPPORTED;
        }
        view->dims[view->rank++] = (int64_t)value;
    }
    return MIR_OK;
}

static MirStatus onnx_parse_float_field(TensorProtoView *view, uint8_t wire, PbCursor *cursor) {
    ByteSlice packed = {0};
    if (wire == ONNX_WIRE_32BIT) {
        if (cursor->off + 4 > cursor->size) {
            return MIR_ERR_RUNTIME;
        }
        float value = onnx_read_f32_le(cursor->data + cursor->off);
        cursor->off += 4;
        return onnx_push_float(view, value);
    }
    if (wire != ONNX_WIRE_LEN || !pb_read_bytes(cursor, &packed)) {
        return MIR_ERR_RUNTIME;
    }
    if ((packed.len % 4) != 0) {
        return MIR_ERR_RUNTIME;
    }
    for (size_t i = 0; i < packed.len; i += 4) {
        MirStatus status = onnx_push_float(view, onnx_read_f32_le(packed.data + i));
        if (status != MIR_OK) {
            return status;
        }
    }
    return MIR_OK;
}

static MirStatus onnx_parse_int64_field(TensorProtoView *view, uint8_t wire, PbCursor *cursor) {
    uint64_t value = 0;
    ByteSlice packed = {0};
    if (wire == ONNX_WIRE_VARINT) {
        if (!pb_read_varint(cursor, &value)) {
            return MIR_ERR_RUNTIME;
        }
        return onnx_push_int64(view, (int64_t)value);
    }
    if (wire != ONNX_WIRE_LEN || !pb_read_bytes(cursor, &packed)) {
        return MIR_ERR_RUNTIME;
    }
    PbCursor sub = {packed.data, packed.len, 0};
    while (sub.off < sub.size) {
        if (!pb_read_varint(&sub, &value)) {
            return MIR_ERR_RUNTIME;
        }
        MirStatus status = onnx_push_int64(view, (int64_t)value);
        if (status != MIR_OK) {
            return status;
        }
    }
    return MIR_OK;
}

static void onnx_tensor_proto_free(TensorProtoView *view) {
    free(view->float_values);
    free(view->int64_values);
    memset(view, 0, sizeof(*view));
}

static MirStatus onnx_parse_tensor_proto(ByteSlice proto, TensorProtoView *out_view) {
    memset(out_view, 0, sizeof(*out_view));
    PbCursor cursor = {proto.data, proto.len, 0};
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        uint64_t value = 0;
        ByteSlice bytes = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }

        switch (field) {
        case 1:
            if (onnx_parse_dims_field(out_view, wire, &cursor) != MIR_OK) {
                return MIR_ERR_RUNTIME;
            }
            break;
        case 2:
            if (wire != ONNX_WIRE_VARINT || !pb_read_varint(&cursor, &value)) {
                return MIR_ERR_RUNTIME;
            }
            out_view->dtype = (int32_t)value;
            break;
        case 4:
            if (onnx_parse_float_field(out_view, wire, &cursor) != MIR_OK) {
                return MIR_ERR_RUNTIME;
            }
            break;
        case 7:
            if (onnx_parse_int64_field(out_view, wire, &cursor) != MIR_OK) {
                return MIR_ERR_RUNTIME;
            }
            break;
        case 8:
            if (wire != ONNX_WIRE_LEN || !pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            out_view->name = bytes;
            break;
        case 9:
            if (wire != ONNX_WIRE_LEN || !pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            out_view->raw_data = bytes;
            break;
        default:
            if (!pb_skip(&cursor, wire)) {
                return MIR_ERR_RUNTIME;
            }
            break;
        }
    }
    return MIR_OK;
}

static MirStatus onnx_make_shape_from_dims(const int64_t *dims, uint32_t rank, MirShape *out_shape) {
    if (rank == 0) {
        return mir_shape_make(NULL, 0, out_shape);
    }
    for (uint32_t i = 0; i < rank; ++i) {
        if (dims[i] <= 0) {
            return MIR_ERR_UNSUPPORTED;
        }
    }
    return mir_shape_make(dims, rank, out_shape);
}

static MirStatus onnx_decode_f32_values(const TensorProtoView *view, size_t elem_count, float **out_values) {
    float *values = (float *)malloc(elem_count * sizeof(float));
    if (!values) {
        return MIR_ERR_OUT_OF_MEMORY;
    }

    if (view->raw_data.len > 0) {
        if (view->raw_data.len != elem_count * sizeof(float)) {
            free(values);
            return MIR_ERR_UNSUPPORTED;
        }
        for (size_t i = 0; i < elem_count; ++i) {
            values[i] = onnx_read_f32_le(view->raw_data.data + i * sizeof(float));
        }
    } else {
        if (view->float_count != elem_count) {
            free(values);
            return MIR_ERR_UNSUPPORTED;
        }
        memcpy(values, view->float_values, elem_count * sizeof(float));
    }

    *out_values = values;
    return MIR_OK;
}

static MirStatus onnx_decode_i64_values(const TensorProtoView *view, size_t elem_count, int64_t **out_values) {
    int64_t *values = (int64_t *)malloc(elem_count * sizeof(int64_t));
    if (!values) {
        return MIR_ERR_OUT_OF_MEMORY;
    }

    if (view->raw_data.len > 0) {
        if (view->raw_data.len != elem_count * sizeof(int64_t)) {
            free(values);
            return MIR_ERR_UNSUPPORTED;
        }
        for (size_t i = 0; i < elem_count; ++i) {
            values[i] = onnx_read_i64_le(view->raw_data.data + i * sizeof(int64_t));
        }
    } else {
        if (view->int64_count != elem_count) {
            free(values);
            return MIR_ERR_UNSUPPORTED;
        }
        memcpy(values, view->int64_values, elem_count * sizeof(int64_t));
    }

    *out_values = values;
    return MIR_OK;
}

static MirStatus onnx_create_tensor_entry(
    MirGraph *graph,
    NameTable *names,
    ByteSlice name,
    const MirShape *shape,
    const float *values,
    size_t *out_id
) {
    MirTensor tensor;
    memset(&tensor, 0, sizeof(tensor));

    MirStatus status = MIR_OK;
    if (shape) {
        status = mir_tensor_init(&tensor, *shape, values != NULL);
        if (status != MIR_OK) {
            return status;
        }
        if (values && tensor.elem_count > 0) {
            memcpy(tensor.data, values, tensor.elem_count * sizeof(float));
        }
    } else {
        tensor.dtype = MIR_DTYPE_F32;
    }

    tensor.name = onnx_copy_string(name);
    if (!tensor.name) {
        mir_tensor_free(&tensor);
        return MIR_ERR_OUT_OF_MEMORY;
    }
    tensor.owns_name = true;

    size_t id = 0;
    status = mir_graph_add_tensor(graph, &tensor, &id);
    if (status != MIR_OK) {
        mir_tensor_free(&tensor);
        return status;
    }

    status = onnx_name_table_add(names, tensor.name, id);
    if (status != MIR_OK) {
        return status;
    }

    if (out_id) {
        *out_id = id;
    }
    return MIR_OK;
}

static MirStatus onnx_get_or_create_tensor(
    MirGraph *graph,
    NameTable *names,
    ByteSlice name,
    size_t *out_id
) {
    size_t id = 0;
    if (onnx_name_table_lookup(names, name, &id)) {
        *out_id = id;
        return MIR_OK;
    }
    return onnx_create_tensor_entry(graph, names, name, NULL, NULL, out_id);
}

static MirStatus onnx_register_initializer(
    MirGraph *graph,
    NameTable *names,
    ShapeTable *shape_consts,
    const TensorProtoView *view
) {
    if (view->name.len == 0) {
        return MIR_ERR_RUNTIME;
    }

    MirShape shape;
    MirStatus status = onnx_make_shape_from_dims(view->dims, view->rank, &shape);
    if (status != MIR_OK) {
        return status;
    }

    size_t elem_count = mir_shape_elem_count(&shape);
    if (view->dtype == ONNX_DTYPE_FLOAT) {
        float *values = NULL;
        status = onnx_decode_f32_values(view, elem_count, &values);
        if (status != MIR_OK) {
            return status;
        }
        status = onnx_create_tensor_entry(graph, names, view->name, &shape, values, NULL);
        free(values);
        return status;
    }
    if (view->dtype == ONNX_DTYPE_INT64) {
        int64_t *values = NULL;
        status = onnx_decode_i64_values(view, elem_count, &values);
        if (status != MIR_OK) {
            return status;
        }
        status = onnx_shape_table_add(shape_consts, view->name, values, elem_count);
        free(values);
        return status;
    }

    return MIR_ERR_UNSUPPORTED;
}

static MirStatus onnx_parse_dim(ByteSlice dim_proto, int64_t *out_value, bool *out_has_value) {
    PbCursor cursor = {dim_proto.data, dim_proto.len, 0};
    *out_has_value = false;
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        uint64_t value = 0;
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }
        if (field == 1 && wire == ONNX_WIRE_VARINT) {
            if (!pb_read_varint(&cursor, &value)) {
                return MIR_ERR_RUNTIME;
            }
            *out_has_value = true;
            *out_value = (int64_t)value;
        } else if (!pb_skip(&cursor, wire)) {
            return MIR_ERR_RUNTIME;
        }
    }
    return MIR_OK;
}

static MirStatus onnx_parse_tensor_shape(ByteSlice shape_proto, MirShape *out_shape, bool *out_has_shape) {
    int64_t dims[MIR_MAX_DIMS];
    uint32_t rank = 0;
    bool complete = true;
    PbCursor cursor = {shape_proto.data, shape_proto.len, 0};
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        ByteSlice dim = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }
        if (field != 1 || wire != ONNX_WIRE_LEN) {
            if (!pb_skip(&cursor, wire)) {
                return MIR_ERR_RUNTIME;
            }
            continue;
        }
        if (!pb_read_bytes(&cursor, &dim)) {
            return MIR_ERR_RUNTIME;
        }
        if (rank >= MIR_MAX_DIMS) {
            return MIR_ERR_UNSUPPORTED;
        }
        bool has_value = false;
        int64_t value = 0;
        MirStatus status = onnx_parse_dim(dim, &value, &has_value);
        if (status != MIR_OK) {
            return status;
        }
        if (!has_value || value <= 0) {
            complete = false;
        }
        dims[rank++] = value;
    }

    if (rank == 0 || !complete) {
        *out_has_shape = false;
        return MIR_OK;
    }

    *out_has_shape = true;
    return onnx_make_shape_from_dims(dims, rank, out_shape);
}

static MirStatus onnx_parse_type_proto(ByteSlice type_proto, MirShape *out_shape, bool *out_has_shape) {
    PbCursor cursor = {type_proto.data, type_proto.len, 0};
    *out_has_shape = false;
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        ByteSlice tensor_type = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }
        if (field != 1 || wire != ONNX_WIRE_LEN) {
            if (!pb_skip(&cursor, wire)) {
                return MIR_ERR_RUNTIME;
            }
            continue;
        }
        if (!pb_read_bytes(&cursor, &tensor_type)) {
            return MIR_ERR_RUNTIME;
        }

        PbCursor tensor_cursor = {tensor_type.data, tensor_type.len, 0};
        while (tensor_cursor.off < tensor_cursor.size) {
            uint32_t tf = 0;
            uint8_t tw = 0;
            ByteSlice shape_proto = {0};
            uint64_t ignored = 0;
            if (!pb_read_key(&tensor_cursor, &tf, &tw)) {
                return MIR_ERR_RUNTIME;
            }
            if (tf == 2 && tw == ONNX_WIRE_LEN) {
                if (!pb_read_bytes(&tensor_cursor, &shape_proto)) {
                    return MIR_ERR_RUNTIME;
                }
                return onnx_parse_tensor_shape(shape_proto, out_shape, out_has_shape);
            }
            if (tf == 1 && tw == ONNX_WIRE_VARINT) {
                if (!pb_read_varint(&tensor_cursor, &ignored)) {
                    return MIR_ERR_RUNTIME;
                }
            } else if (!pb_skip(&tensor_cursor, tw)) {
                return MIR_ERR_RUNTIME;
            }
        }
    }
    return MIR_OK;
}

static MirStatus onnx_parse_value_info(ByteSlice proto, ByteSlice *out_name, bool *out_has_shape, MirShape *out_shape) {
    PbCursor cursor = {proto.data, proto.len, 0};
    memset(out_name, 0, sizeof(*out_name));
    *out_has_shape = false;
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        ByteSlice bytes = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }
        if (field == 1 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            *out_name = bytes;
            continue;
        }
        if (field == 2 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            MirStatus status = onnx_parse_type_proto(bytes, out_shape, out_has_shape);
            if (status != MIR_OK) {
                return status;
            }
            continue;
        }
        if (!pb_skip(&cursor, wire)) {
            return MIR_ERR_RUNTIME;
        }
    }
    if (out_name->len == 0) {
        return MIR_ERR_RUNTIME;
    }
    return MIR_OK;
}

static MirStatus onnx_register_value_info_input(MirGraph *graph, NameTable *names, ByteSlice proto) {
    ByteSlice name = {0};
    bool has_shape = false;
    MirShape shape;
    MirStatus status = onnx_parse_value_info(proto, &name, &has_shape, &shape);
    if (status != MIR_OK) {
        return status;
    }

    size_t existing = 0;
    if (onnx_name_table_lookup(names, name, &existing)) {
        return MIR_OK;
    }

    if (has_shape) {
        return onnx_create_tensor_entry(graph, names, name, &shape, NULL, NULL);
    }
    return onnx_create_tensor_entry(graph, names, name, NULL, NULL, NULL);
}

static MirStatus onnx_parse_attribute(NodeProtoView *node, ByteSlice attr_proto) {
    ByteSlice attr_name = {0};
    bool has_i = false;
    int64_t i_value = 0;
    int64_t ints[MIR_MAX_DIMS];
    size_t ints_count = 0;
    PbCursor cursor = {attr_proto.data, attr_proto.len, 0};
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        uint64_t value = 0;
        ByteSlice bytes = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }

        if (field == 1 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            attr_name = bytes;
            continue;
        }

        if (field == 4 && wire == ONNX_WIRE_VARINT) {
            if (!pb_read_varint(&cursor, &value)) {
                return MIR_ERR_RUNTIME;
            }
            has_i = true;
            i_value = (int64_t)value;
            continue;
        }

        if (field == 9) {
            if (wire == ONNX_WIRE_VARINT) {
                if (!pb_read_varint(&cursor, &value)) {
                    return MIR_ERR_RUNTIME;
                }
                if (ints_count >= MIR_MAX_DIMS) {
                    return MIR_ERR_UNSUPPORTED;
                }
                ints[ints_count++] = (int64_t)value;
                continue;
            }
            if (wire == ONNX_WIRE_LEN) {
                if (!pb_read_bytes(&cursor, &bytes)) {
                    return MIR_ERR_RUNTIME;
                }
                PbCursor packed = {bytes.data, bytes.len, 0};
                while (packed.off < packed.size) {
                    if (!pb_read_varint(&packed, &value)) {
                        return MIR_ERR_RUNTIME;
                    }
                    if (ints_count >= MIR_MAX_DIMS) {
                        return MIR_ERR_UNSUPPORTED;
                    }
                    ints[ints_count++] = (int64_t)value;
                }
                continue;
            }
        }

        if (!pb_skip(&cursor, wire)) {
            return MIR_ERR_RUNTIME;
        }
    }

    if (onnx_slice_eq(attr_name, "axis") && has_i) {
        node->axis_set = true;
        node->axis_value = i_value;
    }

    if (onnx_slice_eq(attr_name, "perm") && ints_count > 0) {
        node->perm_set = true;
        node->perm_rank = (uint32_t)ints_count;
        for (size_t i = 0; i < ints_count; ++i) {
            if (ints[i] < 0) {
                return MIR_ERR_INVALID_ARGUMENT;
            }
            node->perm[i] = (uint32_t)ints[i];
        }
    }

    return MIR_OK;
}

static MirStatus onnx_parse_node_proto(ByteSlice proto, NodeProtoView *node) {
    memset(node, 0, sizeof(*node));
    PbCursor cursor = {proto.data, proto.len, 0};
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        ByteSlice bytes = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }

        if (field == 1 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            if (node->input_count >= 8) {
                return MIR_ERR_UNSUPPORTED;
            }
            node->inputs[node->input_count++] = bytes;
            continue;
        }

        if (field == 2 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            if (node->output_count >= 4) {
                return MIR_ERR_UNSUPPORTED;
            }
            node->outputs[node->output_count++] = bytes;
            continue;
        }

        if (field == 4 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            node->op_type = bytes;
            continue;
        }

        if (field == 5 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            MirStatus status = onnx_parse_attribute(node, bytes);
            if (status != MIR_OK) {
                return status;
            }
            continue;
        }

        if (!pb_skip(&cursor, wire)) {
            return MIR_ERR_RUNTIME;
        }
    }
    return node->op_type.len == 0 ? MIR_ERR_RUNTIME : MIR_OK;
}

static MirStatus onnx_make_reshape_shape(
    const MirTensor *input,
    const int64_t *shape_values,
    size_t shape_count,
    MirShape *out_shape
) {
    if (!input || !shape_values || shape_count == 0 || shape_count > MIR_MAX_DIMS) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    int64_t dims[MIR_MAX_DIMS];
    int infer_index = -1;
    size_t known_product = 1;

    for (size_t i = 0; i < shape_count; ++i) {
        int64_t dim = shape_values[i];
        if (dim == -1) {
            if (infer_index >= 0) {
                return MIR_ERR_UNSUPPORTED;
            }
            infer_index = (int)i;
            dims[i] = 1;
            continue;
        }
        if (dim == 0) {
            if (i >= input->shape.rank) {
                return MIR_ERR_SHAPE_MISMATCH;
            }
            dim = input->shape.dims[i];
        }
        if (dim <= 0) {
            return MIR_ERR_UNSUPPORTED;
        }
        dims[i] = dim;
        known_product *= (size_t)dim;
    }

    if (infer_index >= 0) {
        if (known_product == 0 || input->elem_count == 0 || (input->elem_count % known_product) != 0) {
            return MIR_ERR_SHAPE_MISMATCH;
        }
        dims[infer_index] = (int64_t)(input->elem_count / known_product);
    }

    return mir_shape_make(dims, (uint32_t)shape_count, out_shape);
}

static MirStatus onnx_append_node(
    MirGraph *graph,
    NameTable *names,
    ShapeTable *shape_consts,
    const NodeProtoView *parsed,
    int softmax_axis_default
) {
    size_t output_id = 0;
    MirStatus status = onnx_get_or_create_tensor(graph, names, parsed->outputs[0], &output_id);
    if (status != MIR_OK) {
        return status;
    }

    MirNode node;
    memset(&node, 0, sizeof(node));
    node.output_count = 1;
    node.outputs[0] = (uint32_t)output_id;

    if (onnx_slice_eq(parsed->op_type, "MatMul")) {
        if (parsed->input_count != 2 || parsed->output_count != 1) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        size_t a = 0;
        size_t b = 0;
        status = onnx_get_or_create_tensor(graph, names, parsed->inputs[0], &a);
        if (status != MIR_OK) {
            return status;
        }
        status = onnx_get_or_create_tensor(graph, names, parsed->inputs[1], &b);
        if (status != MIR_OK) {
            return status;
        }
        node.op = MIR_OP_MATMUL;
        node.input_count = 2;
        node.inputs[0] = (uint32_t)a;
        node.inputs[1] = (uint32_t)b;
        return mir_graph_add_node(graph, &node, NULL);
    }

    if (onnx_slice_eq(parsed->op_type, "Add")) {
        if (parsed->input_count != 2 || parsed->output_count != 1) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        size_t a = 0;
        size_t b = 0;
        status = onnx_get_or_create_tensor(graph, names, parsed->inputs[0], &a);
        if (status != MIR_OK) {
            return status;
        }
        status = onnx_get_or_create_tensor(graph, names, parsed->inputs[1], &b);
        if (status != MIR_OK) {
            return status;
        }
        node.op = MIR_OP_ADD;
        node.input_count = 2;
        node.inputs[0] = (uint32_t)a;
        node.inputs[1] = (uint32_t)b;
        return mir_graph_add_node(graph, &node, NULL);
    }

    if (onnx_slice_eq(parsed->op_type, "Relu")) {
        if (parsed->input_count != 1 || parsed->output_count != 1) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        size_t input = 0;
        status = onnx_get_or_create_tensor(graph, names, parsed->inputs[0], &input);
        if (status != MIR_OK) {
            return status;
        }
        node.op = MIR_OP_RELU;
        node.input_count = 1;
        node.inputs[0] = (uint32_t)input;
        return mir_graph_add_node(graph, &node, NULL);
    }

    if (onnx_slice_eq(parsed->op_type, "Softmax")) {
        if (parsed->input_count != 1 || parsed->output_count != 1) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        size_t input = 0;
        status = onnx_get_or_create_tensor(graph, names, parsed->inputs[0], &input);
        if (status != MIR_OK) {
            return status;
        }
        node.op = MIR_OP_SOFTMAX;
        node.input_count = 1;
        node.inputs[0] = (uint32_t)input;
        node.attrs.axis = parsed->axis_set ? (int)parsed->axis_value : softmax_axis_default;
        return mir_graph_add_node(graph, &node, NULL);
    }

    if (onnx_slice_eq(parsed->op_type, "Reshape")) {
        if (parsed->input_count != 2 || parsed->output_count != 1) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        size_t data_input_id = 0;
        status = onnx_get_or_create_tensor(graph, names, parsed->inputs[0], &data_input_id);
        if (status != MIR_OK) {
            return status;
        }
        const ShapeEntry *shape_entry = onnx_shape_table_lookup(shape_consts, parsed->inputs[1]);
        if (!shape_entry) {
            return MIR_ERR_UNSUPPORTED;
        }
        const MirTensor *data_input = mir_graph_tensor_const(graph, data_input_id);
        MirShape reshape_shape;
        status = onnx_make_reshape_shape(data_input, shape_entry->values, shape_entry->count, &reshape_shape);
        if (status != MIR_OK) {
            return status;
        }
        node.op = MIR_OP_RESHAPE;
        node.input_count = 1;
        node.inputs[0] = (uint32_t)data_input_id;
        node.attrs.reshape_shape = reshape_shape;
        return mir_graph_add_node(graph, &node, NULL);
    }

    if (onnx_slice_eq(parsed->op_type, "Transpose")) {
        if (parsed->input_count != 1 || parsed->output_count != 1) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        size_t input = 0;
        status = onnx_get_or_create_tensor(graph, names, parsed->inputs[0], &input);
        if (status != MIR_OK) {
            return status;
        }
        node.op = MIR_OP_TRANSPOSE;
        node.input_count = 1;
        node.inputs[0] = (uint32_t)input;
        if (parsed->perm_set) {
            node.attrs.perm_rank = parsed->perm_rank;
            for (uint32_t i = 0; i < parsed->perm_rank; ++i) {
                node.attrs.perm[i] = parsed->perm[i];
            }
        } else {
            const MirTensor *input_tensor = mir_graph_tensor_const(graph, input);
            if (!input_tensor || input_tensor->shape.rank == 0) {
                return MIR_ERR_UNSUPPORTED;
            }
            node.attrs.perm_rank = input_tensor->shape.rank;
            for (uint32_t i = 0; i < input_tensor->shape.rank; ++i) {
                node.attrs.perm[i] = input_tensor->shape.rank - 1 - i;
            }
        }
        return mir_graph_add_node(graph, &node, NULL);
    }

    return MIR_ERR_UNSUPPORTED;
}

static MirStatus onnx_parse_graph_initial(
    ByteSlice graph_proto,
    MirGraph *graph,
    NameTable *names,
    ShapeTable *shape_consts
) {
    PbCursor cursor = {graph_proto.data, graph_proto.len, 0};
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        ByteSlice bytes = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }

        if (field == 5 && wire == ONNX_WIRE_LEN) {
            TensorProtoView tensor;
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            MirStatus status = onnx_parse_tensor_proto(bytes, &tensor);
            if (status == MIR_OK) {
                status = onnx_register_initializer(graph, names, shape_consts, &tensor);
            }
            onnx_tensor_proto_free(&tensor);
            if (status != MIR_OK) {
                return status;
            }
            continue;
        }

        if ((field == 11 || field == 12) && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            MirStatus status = onnx_register_value_info_input(graph, names, bytes);
            if (status != MIR_OK) {
                return status;
            }
            continue;
        }

        if (!pb_skip(&cursor, wire)) {
            return MIR_ERR_RUNTIME;
        }
    }
    return MIR_OK;
}

static MirStatus onnx_parse_graph_nodes(
    ByteSlice graph_proto,
    MirGraph *graph,
    NameTable *names,
    ShapeTable *shape_consts,
    int softmax_axis_default
) {
    PbCursor cursor = {graph_proto.data, graph_proto.len, 0};
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        ByteSlice bytes = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }
        if (field == 1 && wire == ONNX_WIRE_LEN) {
            NodeProtoView node;
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            MirStatus status = onnx_parse_node_proto(bytes, &node);
            if (status != MIR_OK) {
                return status;
            }
            status = onnx_append_node(graph, names, shape_consts, &node, softmax_axis_default);
            if (status != MIR_OK) {
                return status;
            }
            continue;
        }
        if (!pb_skip(&cursor, wire)) {
            return MIR_ERR_RUNTIME;
        }
    }
    return MIR_OK;
}

static MirStatus onnx_parse_opset_import(ByteSlice proto, int64_t *main_opset) {
    ByteSlice domain = {0};
    int64_t version = 0;
    bool has_version = false;
    PbCursor cursor = {proto.data, proto.len, 0};
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        uint64_t value = 0;
        ByteSlice bytes = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }
        if (field == 1 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &bytes)) {
                return MIR_ERR_RUNTIME;
            }
            domain = bytes;
            continue;
        }
        if (field == 2 && wire == ONNX_WIRE_VARINT) {
            if (!pb_read_varint(&cursor, &value)) {
                return MIR_ERR_RUNTIME;
            }
            version = (int64_t)value;
            has_version = true;
            continue;
        }
        if (!pb_skip(&cursor, wire)) {
            return MIR_ERR_RUNTIME;
        }
    }

    if (domain.len == 0 && has_version) {
        *main_opset = version;
    }
    return MIR_OK;
}

static MirStatus onnx_parse_model(ByteSlice bytes, ModelProtoView *model) {
    memset(model, 0, sizeof(*model));
    model->default_opset = 13;
    PbCursor cursor = {bytes.data, bytes.len, 0};
    while (cursor.off < cursor.size) {
        uint32_t field = 0;
        uint8_t wire = 0;
        ByteSlice value = {0};
        if (!pb_read_key(&cursor, &field, &wire)) {
            return MIR_ERR_RUNTIME;
        }

        if (field == 7 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &value)) {
                return MIR_ERR_RUNTIME;
            }
            model->graph_proto = value;
            continue;
        }

        if (field == 8 && wire == ONNX_WIRE_LEN) {
            if (!pb_read_bytes(&cursor, &value)) {
                return MIR_ERR_RUNTIME;
            }
            MirStatus status = onnx_parse_opset_import(value, &model->default_opset);
            if (status != MIR_OK) {
                return status;
            }
            continue;
        }

        if (!pb_skip(&cursor, wire)) {
            return MIR_ERR_RUNTIME;
        }
    }
    return model->graph_proto.len == 0 ? MIR_ERR_RUNTIME : MIR_OK;
}

MirStatus mir_onnx_load_buffer(const void *data, size_t size, MirGraph *graph) {
    if (!data || size == 0 || !graph) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    ModelProtoView model;
    MirStatus status = onnx_parse_model((ByteSlice){(const uint8_t *)data, size}, &model);
    if (status != MIR_OK) {
        return status;
    }

    MirGraph local_graph;
    status = mir_graph_init(&local_graph, 16, 16);
    if (status != MIR_OK) {
        return status;
    }

    NameTable names;
    ShapeTable shape_consts;
    memset(&names, 0, sizeof(names));
    memset(&shape_consts, 0, sizeof(shape_consts));

    status = onnx_parse_graph_initial(model.graph_proto, &local_graph, &names, &shape_consts);
    if (status == MIR_OK) {
        int default_axis = model.default_opset < 13 ? 1 : -1;
        status = onnx_parse_graph_nodes(model.graph_proto, &local_graph, &names, &shape_consts, default_axis);
    }

    onnx_name_table_free(&names);
    onnx_shape_table_free(&shape_consts);

    if (status != MIR_OK) {
        mir_graph_free(&local_graph);
        return status;
    }

    *graph = local_graph;
    return MIR_OK;
}

MirStatus mir_onnx_load_file(const char *path, MirGraph *graph) {
    if (!path || !graph) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    FILE *file = fopen(path, "rb");
    if (!file) {
        return MIR_ERR_RUNTIME;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return MIR_ERR_RUNTIME;
    }
    long file_len = ftell(file);
    if (file_len <= 0) {
        fclose(file);
        return MIR_ERR_RUNTIME;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return MIR_ERR_RUNTIME;
    }

    size_t size = (size_t)file_len;
    uint8_t *buffer = (uint8_t *)malloc(size);
    if (!buffer) {
        fclose(file);
        return MIR_ERR_OUT_OF_MEMORY;
    }

    size_t read_count = fread(buffer, 1, size, file);
    fclose(file);
    if (read_count != size) {
        free(buffer);
        return MIR_ERR_RUNTIME;
    }

    MirStatus status = mir_onnx_load_buffer(buffer, size, graph);
    free(buffer);
    return status;
}
