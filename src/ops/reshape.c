#include "mir.h"

MirStatus mir_op_reshape(const MirTensor *input, MirShape new_shape, MirTensor *out) {
    if (!input || !out || !input->data) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    if (mir_shape_elem_count(&new_shape) != input->elem_count) {
        return MIR_ERR_SHAPE_MISMATCH;
    }

    return mir_tensor_view(out, input, new_shape);
}
