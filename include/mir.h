#ifndef MIR_H
#define MIR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum { MIR_MAX_DIMS = 8, MIR_MAX_NODE_INPUTS = 4, MIR_MAX_NODE_OUTPUTS = 2 };

typedef enum {
    MIR_OK = 0,
    MIR_ERR_INVALID_ARGUMENT,
    MIR_ERR_OUT_OF_MEMORY,
    MIR_ERR_SHAPE_MISMATCH,
    MIR_ERR_UNSUPPORTED,
    MIR_ERR_RUNTIME
} MirStatus;

typedef enum {
    MIR_DTYPE_F32 = 0
} MirDType;

typedef struct {
    uint32_t rank;
    int64_t dims[MIR_MAX_DIMS];
} MirShape;

typedef struct {
    MirDType dtype;
    MirShape shape;
    int64_t strides[MIR_MAX_DIMS];
    size_t elem_count;
    size_t bytes;
    float *data;
    char *name;
    bool owns_name;
    bool owns_data;
} MirTensor;

typedef enum {
    MIR_OP_MATMUL = 0,
    MIR_OP_ADD,
    MIR_OP_RELU,
    MIR_OP_SOFTMAX,
    MIR_OP_RESHAPE,
    MIR_OP_TRANSPOSE
} MirOpType;

typedef struct {
    int axis;
    MirShape reshape_shape;
    uint32_t perm_rank;
    uint32_t perm[MIR_MAX_DIMS];
} MirNodeAttrs;

typedef struct {
    MirOpType op;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t inputs[MIR_MAX_NODE_INPUTS];
    uint32_t outputs[MIR_MAX_NODE_OUTPUTS];
    MirNodeAttrs attrs;
    const char *name;
} MirNode;

typedef struct {
    MirTensor *tensors;
    size_t tensor_count;
    size_t tensor_capacity;
    MirNode *nodes;
    size_t node_count;
    size_t node_capacity;
} MirGraph;

typedef struct {
    uint8_t *base;
    size_t capacity;
    size_t offset;
    bool owns_memory;
} MirArena;

typedef void (*MirLogFn)(void *user, const char *message);

typedef struct {
    MirLogFn log_fn;
    void *log_user;
    bool trace_execution;
} MirExecutionOptions;

typedef struct {
    size_t node_index;
    MirOpType op;
    uint64_t duration_ns;
} MirProfileEntry;

typedef struct {
    MirProfileEntry *entries;
    size_t capacity;
    size_t count;
} MirProfileBuffer;

const char *mir_status_str(MirStatus status);

MirStatus mir_shape_make(const int64_t *dims, uint32_t rank, MirShape *out_shape);
size_t mir_shape_elem_count(const MirShape *shape);
bool mir_shape_equal(const MirShape *a, const MirShape *b);

MirStatus mir_tensor_init(MirTensor *tensor, MirShape shape, bool allocate_data);
MirStatus mir_tensor_init_with_data(MirTensor *tensor, MirShape shape, float *data, bool owns_data);
MirStatus mir_tensor_resize(MirTensor *tensor, MirShape shape);
MirStatus mir_tensor_view(MirTensor *tensor, const MirTensor *source, MirShape shape);
void mir_tensor_free(MirTensor *tensor);
void mir_tensor_dump(const MirTensor *tensor, FILE *out, size_t max_elements);

MirStatus mir_graph_init(MirGraph *graph, size_t tensor_capacity, size_t node_capacity);
void mir_graph_free(MirGraph *graph);
MirStatus mir_graph_add_tensor(MirGraph *graph, const MirTensor *tensor, size_t *out_id);
MirStatus mir_graph_add_empty_tensor(MirGraph *graph, size_t *out_id);
MirTensor *mir_graph_tensor(MirGraph *graph, size_t id);
const MirTensor *mir_graph_tensor_const(const MirGraph *graph, size_t id);
MirStatus mir_graph_find_tensor(const MirGraph *graph, const char *name, size_t *out_id);
MirStatus mir_graph_add_node(MirGraph *graph, const MirNode *node, size_t *out_id);
void mir_graph_dump(const MirGraph *graph, FILE *out);

MirStatus mir_arena_init(MirArena *arena, size_t capacity);
MirStatus mir_arena_from_buffer(MirArena *arena, void *buffer, size_t capacity);
void *mir_arena_alloc(MirArena *arena, size_t size, size_t alignment);
void mir_arena_reset(MirArena *arena);
void mir_arena_free(MirArena *arena);

MirStatus mir_op_matmul(const MirTensor *a, const MirTensor *b, MirTensor *out);
MirStatus mir_op_add(const MirTensor *a, const MirTensor *b, MirTensor *out);
MirStatus mir_op_relu(const MirTensor *input, MirTensor *out);
MirStatus mir_op_softmax(const MirTensor *input, int axis, MirTensor *out);
MirStatus mir_op_reshape(const MirTensor *input, MirShape new_shape, MirTensor *out);
MirStatus mir_op_transpose(
    const MirTensor *input,
    const uint32_t *perm,
    uint32_t perm_rank,
    MirTensor *out
);

MirStatus mir_execute_graph(
    MirGraph *graph,
    const MirExecutionOptions *options,
    MirProfileBuffer *profile
);

MirStatus mir_onnx_load_buffer(const void *data, size_t size, MirGraph *graph);
MirStatus mir_onnx_load_file(const char *path, MirGraph *graph);

#endif
