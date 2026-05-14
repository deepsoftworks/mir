#define _POSIX_C_SOURCE 200809L

#include "mir.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static uint64_t mir_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

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

static void mir_log(const MirExecutionOptions *opts, const char *message) {
    if (opts && opts->log_fn) {
        opts->log_fn(opts->log_user, message);
    }
}

static MirStatus mir_require_counts(const MirNode *node, uint32_t in_count, uint32_t out_count) {
    if (node->input_count != in_count || node->output_count != out_count) {
        return MIR_ERR_INVALID_ARGUMENT;
    }
    return MIR_OK;
}

static MirStatus mir_dispatch_node(MirGraph *graph, const MirNode *node) {
    const MirTensor *in0 = NULL;
    const MirTensor *in1 = NULL;
    MirTensor *out0 = NULL;

    if (node->input_count > 0) {
        in0 = mir_graph_tensor_const(graph, node->inputs[0]);
    }
    if (node->input_count > 1) {
        in1 = mir_graph_tensor_const(graph, node->inputs[1]);
    }
    if (node->output_count > 0) {
        out0 = mir_graph_tensor(graph, node->outputs[0]);
    }
    if ((node->input_count > 0 && !in0) || (node->input_count > 1 && !in1) || !out0) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    switch (node->op) {
    case MIR_OP_MATMUL:
        if (mir_require_counts(node, 2, 1) != MIR_OK) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        return mir_op_matmul(in0, in1, out0);
    case MIR_OP_ADD:
        if (mir_require_counts(node, 2, 1) != MIR_OK) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        return mir_op_add(in0, in1, out0);
    case MIR_OP_RELU:
        if (mir_require_counts(node, 1, 1) != MIR_OK) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        return mir_op_relu(in0, out0);
    case MIR_OP_SOFTMAX:
        if (mir_require_counts(node, 1, 1) != MIR_OK) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        return mir_op_softmax(in0, node->attrs.axis, out0);
    case MIR_OP_RESHAPE:
        if (mir_require_counts(node, 1, 1) != MIR_OK) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        return mir_op_reshape(in0, node->attrs.reshape_shape, out0);
    case MIR_OP_TRANSPOSE:
        if (mir_require_counts(node, 1, 1) != MIR_OK) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        return mir_op_transpose(in0, node->attrs.perm, node->attrs.perm_rank, out0);
    default:
        return MIR_ERR_UNSUPPORTED;
    }
}

MirStatus mir_execute_graph(
    MirGraph *graph,
    const MirExecutionOptions *options,
    MirProfileBuffer *profile
) {
    if (!graph) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    if (profile) {
        profile->count = 0;
    }

    char line[160];
    for (size_t i = 0; i < graph->node_count; ++i) {
        const MirNode *node = &graph->nodes[i];
        if (options && options->trace_execution) {
            snprintf(line, sizeof(line), "node[%zu] begin %s", i, mir_op_name(node->op));
            mir_log(options, line);
        }

        uint64_t start = mir_now_ns();
        MirStatus status = mir_dispatch_node(graph, node);
        uint64_t end = mir_now_ns();
        if (status != MIR_OK) {
            snprintf(
                line,
                sizeof(line),
                "node[%zu] failed %s status=%s",
                i,
                mir_op_name(node->op),
                mir_status_str(status)
            );
            mir_log(options, line);
            return status;
        }

        if (profile && profile->entries && profile->count < profile->capacity) {
            MirProfileEntry *entry = &profile->entries[profile->count++];
            entry->node_index = i;
            entry->op = node->op;
            entry->duration_ns = end - start;
        }

        if (options && options->trace_execution) {
            snprintf(line, sizeof(line), "node[%zu] end %s", i, mir_op_name(node->op));
            mir_log(options, line);
        }
    }

    return MIR_OK;
}
