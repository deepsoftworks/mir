#include "mir.h"

MirStatus mir_op_add(const MirTensor *a, const MirTensor *b, MirTensor *out) {
    if (!a || !b || !out || !a->data || !b->data) {
        return MIR_ERR_INVALID_ARGUMENT;
    }
    if (!mir_shape_equal(&a->shape, &b->shape)) {
        return MIR_ERR_SHAPE_MISMATCH;
    }

    MirStatus status = mir_tensor_resize(out, a->shape);
    if (status != MIR_OK) {
        return status;
    }

    for (size_t i = 0; i < a->elem_count; ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return MIR_OK;
}
