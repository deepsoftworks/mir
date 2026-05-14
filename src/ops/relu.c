#include "mir.h"

MirStatus mir_op_relu(const MirTensor *input, MirTensor *out) {
    if (!input || !out || !input->data) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    MirStatus status = mir_tensor_resize(out, input->shape);
    if (status != MIR_OK) {
        return status;
    }

    for (size_t i = 0; i < input->elem_count; ++i) {
        float value = input->data[i];
        out->data[i] = value > 0.0f ? value : 0.0f;
    }

    return MIR_OK;
}
