#include "mir.h"

#include <stdlib.h>
#include <string.h>

static const char *mir_op_name(MirOpType op) {
    switch (op) {
    case MIR_OP_MATMUL:
        return "MatMul";
    case MIR_OP_ADD:
        return "Add";
    case MIR_OP_RELU:
        return "ReLU";
    case MIR_OP_SOFTMAX:
        return "Softmax";
    case MIR_OP_RESHAPE:
        return "Reshape";
    case MIR_OP_TRANSPOSE:
        return "Transpose";
    default:
        return "Unknown";
    }
}

MirStatus mir_graph_init(MirGraph *graph, size_t tensor_capacity, size_t node_capacity) {
    if (!graph || tensor_capacity == 0 || node_capacity == 0) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    memset(graph, 0, sizeof(*graph));

    graph->tensors = (MirTensor *)calloc(tensor_capacity, sizeof(MirTensor));
    graph->nodes = (MirNode *)calloc(node_capacity, sizeof(MirNode));
    if (!graph->tensors || !graph->nodes) {
        mir_graph_free(graph);
        return MIR_ERR_OUT_OF_MEMORY;
    }

    graph->tensor_capacity = tensor_capacity;
    graph->node_capacity = node_capacity;
    return MIR_OK;
}

void mir_graph_free(MirGraph *graph) {
    if (!graph) {
        return;
    }

    if (graph->tensors) {
        for (size_t i = 0; i < graph->tensor_count; ++i) {
            mir_tensor_free(&graph->tensors[i]);
        }
        free(graph->tensors);
    }
    free(graph->nodes);
    memset(graph, 0, sizeof(*graph));
}

MirStatus mir_graph_add_tensor(MirGraph *graph, const MirTensor *tensor, size_t *out_id) {
    if (!graph || !tensor) {
        return MIR_ERR_INVALID_ARGUMENT;
    }
    if (graph->tensor_count >= graph->tensor_capacity) {
        return MIR_ERR_OUT_OF_MEMORY;
    }

    size_t id = graph->tensor_count++;
    graph->tensors[id] = *tensor;
    if (out_id) {
        *out_id = id;
    }
    return MIR_OK;
}

MirStatus mir_graph_add_empty_tensor(MirGraph *graph, size_t *out_id) {
    if (!graph) {
        return MIR_ERR_INVALID_ARGUMENT;
    }
    if (graph->tensor_count >= graph->tensor_capacity) {
        return MIR_ERR_OUT_OF_MEMORY;
    }

    size_t id = graph->tensor_count++;
    memset(&graph->tensors[id], 0, sizeof(MirTensor));
    graph->tensors[id].dtype = MIR_DTYPE_F32;
    if (out_id) {
        *out_id = id;
    }
    return MIR_OK;
}

MirTensor *mir_graph_tensor(MirGraph *graph, size_t id) {
    if (!graph || id >= graph->tensor_count) {
        return NULL;
    }
    return &graph->tensors[id];
}

const MirTensor *mir_graph_tensor_const(const MirGraph *graph, size_t id) {
    if (!graph || id >= graph->tensor_count) {
        return NULL;
    }
    return &graph->tensors[id];
}

MirStatus mir_graph_add_node(MirGraph *graph, const MirNode *node, size_t *out_id) {
    if (!graph || !node) {
        return MIR_ERR_INVALID_ARGUMENT;
    }
    if (node->input_count > MIR_MAX_NODE_INPUTS || node->output_count > MIR_MAX_NODE_OUTPUTS) {
        return MIR_ERR_INVALID_ARGUMENT;
    }
    if (graph->node_count >= graph->node_capacity) {
        return MIR_ERR_OUT_OF_MEMORY;
    }

    for (uint32_t i = 0; i < node->input_count; ++i) {
        if (node->inputs[i] >= graph->tensor_count) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
    }
    for (uint32_t i = 0; i < node->output_count; ++i) {
        if (node->outputs[i] >= graph->tensor_count) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
    }

    size_t id = graph->node_count++;
    graph->nodes[id] = *node;
    if (out_id) {
        *out_id = id;
    }
    return MIR_OK;
}

void mir_graph_dump(const MirGraph *graph, FILE *out) {
    if (!graph || !out) {
        return;
    }

    fprintf(out, "graph(tensors=%zu, nodes=%zu)\n", graph->tensor_count, graph->node_count);
    for (size_t i = 0; i < graph->tensor_count; ++i) {
        const MirTensor *t = &graph->tensors[i];
        fprintf(out, "  t%zu: shape=[", i);
        for (uint32_t d = 0; d < t->shape.rank; ++d) {
            fprintf(out, "%lld%s", (long long)t->shape.dims[d], d + 1 < t->shape.rank ? "," : "");
        }
        fprintf(out, "] data=%s\n", t->data ? "set" : "null");
    }

    for (size_t i = 0; i < graph->node_count; ++i) {
        const MirNode *n = &graph->nodes[i];
        fprintf(out, "  n%zu: %s(", i, mir_op_name(n->op));
        for (uint32_t in = 0; in < n->input_count; ++in) {
            fprintf(out, "t%u%s", n->inputs[in], in + 1 < n->input_count ? "," : "");
        }
        fprintf(out, ") -> ");
        for (uint32_t out_id = 0; out_id < n->output_count; ++out_id) {
            fprintf(out, "t%u%s", n->outputs[out_id], out_id + 1 < n->output_count ? "," : "");
        }
        if (n->name) {
            fprintf(out, " # %s", n->name);
        }
        fprintf(out, "\n");
    }
}
