#include "mir.h"

#include <string.h>

static MirStatus mir_validate_perm(uint32_t rank, const uint32_t *perm, uint32_t perm_rank) {
    if (!perm || perm_rank != rank || rank == 0 || rank > MIR_MAX_DIMS) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    bool seen[MIR_MAX_DIMS];
    memset(seen, 0, sizeof(seen));
    for (uint32_t i = 0; i < rank; ++i) {
        if (perm[i] >= rank || seen[perm[i]]) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        seen[perm[i]] = true;
    }
    return MIR_OK;
}

MirStatus mir_op_transpose(
    const MirTensor *input,
    const uint32_t *perm,
    uint32_t perm_rank,
    MirTensor *out
) {
    if (!input || !out || !input->data) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    MirStatus status = mir_validate_perm(input->shape.rank, perm, perm_rank);
    if (status != MIR_OK) {
        return status;
    }

    int64_t out_dims[MIR_MAX_DIMS];
    for (uint32_t i = 0; i < input->shape.rank; ++i) {
        out_dims[i] = input->shape.dims[perm[i]];
    }

    MirShape out_shape;
    status = mir_shape_make(out_dims, input->shape.rank, &out_shape);
    if (status != MIR_OK) {
        return status;
    }

    status = mir_tensor_resize(out, out_shape);
    if (status != MIR_OK) {
        return status;
    }

    int64_t coords[MIR_MAX_DIMS];
    for (size_t out_idx = 0; out_idx < out->elem_count; ++out_idx) {
        size_t rem = out_idx;
        for (uint32_t d = 0; d < out->shape.rank; ++d) {
            coords[d] = (int64_t)(rem / (size_t)out->strides[d]);
            rem %= (size_t)out->strides[d];
        }

        size_t in_idx = 0;
        for (uint32_t d = 0; d < input->shape.rank; ++d) {
            in_idx += (size_t)(coords[d] * input->strides[perm[d]]);
        }
        out->data[out_idx] = input->data[in_idx];
    }

    return MIR_OK;
}
