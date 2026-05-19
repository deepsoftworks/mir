#include "mir.h"

#include <stdlib.h>
#include <string.h>

#define PLAN_ALIGN 16

typedef struct {
    size_t tensor_id;
    size_t bytes;
    size_t born;
    size_t dies;
    size_t offset;
    bool is_view;
} PlanEntry;

static size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

static MirStatus infer_output_shape(const MirGraph *graph, const MirNode *node, MirShape *out) {
    const MirTensor *in0 = &graph->tensors[node->inputs[0]];

    switch (node->op) {
    case MIR_OP_MATMUL: {
        const MirTensor *in1 = &graph->tensors[node->inputs[1]];
        int64_t dims[2] = {in0->shape.dims[0], in1->shape.dims[1]};
        return mir_shape_make(dims, 2, out);
    }
    case MIR_OP_ADD:
    case MIR_OP_RELU:
    case MIR_OP_SOFTMAX:
        *out = in0->shape;
        return MIR_OK;
    case MIR_OP_RESHAPE:
        *out = node->attrs.reshape_shape;
        return MIR_OK;
    case MIR_OP_TRANSPOSE: {
        out->rank = in0->shape.rank;
        for (uint32_t i = 0; i < node->attrs.perm_rank; ++i) {
            out->dims[i] = in0->shape.dims[node->attrs.perm[i]];
        }
        return MIR_OK;
    }
    default:
        return MIR_ERR_UNSUPPORTED;
    }
}

static bool lifetimes_overlap(const PlanEntry *a, const PlanEntry *b) {
    return a->born <= b->dies && b->born <= a->dies;
}

static PlanEntry *find_entry(PlanEntry *entries, size_t count, size_t tensor_id) {
    for (size_t i = 0; i < count; ++i) {
        if (entries[i].tensor_id == tensor_id) {
            return &entries[i];
        }
    }
    return NULL;
}

static MirStatus mir_plan_internal(MirGraph *graph, MirArena *arena, size_t *out_size) {
    /* shape inference */
    for (size_t i = 0; i < graph->node_count; ++i) {
        const MirNode *node = &graph->nodes[i];
        for (uint32_t j = 0; j < node->output_count; ++j) {
            MirTensor *out = &graph->tensors[node->outputs[j]];
            if (out->data) {
                continue;
            }
            MirShape shape;
            MirStatus status = infer_output_shape(graph, node, &shape);
            if (status != MIR_OK) {
                return status;
            }
            out->shape = shape;
            out->elem_count = mir_shape_elem_count(&shape);
            out->bytes = out->elem_count * sizeof(float);
        }
    }

    /* build plans for intermediate tensors */
    size_t plan_count = 0;
    for (size_t i = 0; i < graph->tensor_count; ++i) {
        if (!graph->tensors[i].data) {
            plan_count++;
        }
    }

    if (plan_count == 0) {
        if (out_size) {
            *out_size = 0;
        }
        return MIR_OK;
    }

    PlanEntry *entries = (PlanEntry *)calloc(plan_count, sizeof(PlanEntry));
    if (!entries) {
        return MIR_ERR_OUT_OF_MEMORY;
    }

    size_t ei = 0;
    for (size_t i = 0; i < graph->tensor_count; ++i) {
        if (graph->tensors[i].data) {
            continue;
        }
        entries[ei].tensor_id = i;
        entries[ei].bytes = graph->tensors[i].bytes;
        entries[ei].born = graph->node_count;
        entries[ei].dies = 0;
        entries[ei].is_view = false;
        ei++;
    }

    /* lifetimes */
    for (size_t ni = 0; ni < graph->node_count; ++ni) {
        const MirNode *node = &graph->nodes[ni];

        for (uint32_t j = 0; j < node->output_count; ++j) {
            PlanEntry *e = find_entry(entries, plan_count, node->outputs[j]);
            if (e) {
                e->born = ni;
                if (node->op == MIR_OP_RESHAPE) {
                    e->is_view = true;
                }
            }
        }

        for (uint32_t j = 0; j < node->input_count; ++j) {
            PlanEntry *e = find_entry(entries, plan_count, node->inputs[j]);
            if (e && ni > e->dies) {
                e->dies = ni;
            }
        }
    }

    /* Tensors never consumed (graph outputs) live until the last node */
    for (size_t i = 0; i < plan_count; ++i) {
        if (entries[i].dies < entries[i].born) {
            entries[i].dies = graph->node_count > 0 ? graph->node_count - 1 : 0;
        }
    }

    /* Extend source lifetime through reshape views */
    for (size_t ni = 0; ni < graph->node_count; ++ni) {
        const MirNode *node = &graph->nodes[ni];
        if (node->op != MIR_OP_RESHAPE) {
            continue;
        }
        PlanEntry *src = find_entry(entries, plan_count, node->inputs[0]);
        PlanEntry *dst = find_entry(entries, plan_count, node->outputs[0]);
        if (src && dst && dst->dies > src->dies) {
            src->dies = dst->dies;
        }
    }

    /* greedy offset assignment */
    for (size_t i = 0; i < plan_count; ++i) {
        if (entries[i].is_view) {
            continue;
        }
        for (size_t j = i + 1; j < plan_count; ++j) {
            if (entries[j].is_view) {
                continue;
            }
            if (entries[j].bytes > entries[i].bytes) {
                PlanEntry tmp = entries[i];
                entries[i] = entries[j];
                entries[j] = tmp;
            }
        }
    }

    size_t peak = 0;
    for (size_t i = 0; i < plan_count; ++i) {
        if (entries[i].is_view || entries[i].bytes == 0) {
            continue;
        }

        size_t offset = 0;
        for (;;) {
            offset = align_up(offset, PLAN_ALIGN);
            bool conflict = false;
            for (size_t j = 0; j < i; ++j) {
                if (entries[j].is_view) {
                    continue;
                }
                if (!lifetimes_overlap(&entries[i], &entries[j])) {
                    continue;
                }
                size_t j_end = entries[j].offset + entries[j].bytes;
                size_t i_end = offset + entries[i].bytes;
                if (offset < j_end && entries[j].offset < i_end) {
                    offset = align_up(j_end, PLAN_ALIGN);
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                break;
            }
        }

        entries[i].offset = offset;
        size_t end = offset + entries[i].bytes;
        if (end > peak) {
            peak = end;
        }
    }

    if (out_size) {
        *out_size = peak;
    }

    /* assign data pointers from arena */
    if (arena) {
        if (arena->capacity < peak) {
            free(entries);
            return MIR_ERR_OUT_OF_MEMORY;
        }
        for (size_t i = 0; i < plan_count; ++i) {
            if (entries[i].is_view || entries[i].bytes == 0) {
                continue;
            }
            MirTensor *t = &graph->tensors[entries[i].tensor_id];
            t->data = (float *)(arena->base + entries[i].offset);
            t->owns_data = false;
            t->arena = arena;
        }
    }

    free(entries);
    return MIR_OK;
}

size_t mir_graph_plan_size(const MirGraph *graph) {
    size_t size = 0;
    mir_plan_internal((MirGraph *)graph, NULL, &size);
    return size;
}

MirStatus mir_graph_plan(MirGraph *graph, MirArena *arena) {
    return mir_plan_internal(graph, arena, NULL);
}
