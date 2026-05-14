#include "mir.h"

#include <math.h>

static MirStatus mir_normalize_axis(uint32_t rank, int axis, uint32_t *out_axis) {
    if (!out_axis || rank == 0) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    int normalized = axis;
    if (normalized < 0) {
        normalized += (int)rank;
    }
    if (normalized < 0 || normalized >= (int)rank) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    *out_axis = (uint32_t)normalized;
    return MIR_OK;
}

MirStatus mir_op_softmax(const MirTensor *input, int axis, MirTensor *out) {
    if (!input || !out || !input->data) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    uint32_t softmax_axis = 0;
    MirStatus status = mir_normalize_axis(input->shape.rank, axis, &softmax_axis);
    if (status != MIR_OK) {
        return status;
    }

    status = mir_tensor_resize(out, input->shape);
    if (status != MIR_OK) {
        return status;
    }

    size_t outer = 1;
    size_t inner = 1;
    size_t axis_size = (size_t)input->shape.dims[softmax_axis];
    for (uint32_t i = 0; i < softmax_axis; ++i) {
        outer *= (size_t)input->shape.dims[i];
    }
    for (uint32_t i = softmax_axis + 1; i < input->shape.rank; ++i) {
        inner *= (size_t)input->shape.dims[i];
    }

    for (size_t outer_i = 0; outer_i < outer; ++outer_i) {
        for (size_t inner_i = 0; inner_i < inner; ++inner_i) {
            size_t base = outer_i * axis_size * inner + inner_i;
            float max_value = input->data[base];
            for (size_t axis_i = 1; axis_i < axis_size; ++axis_i) {
                float v = input->data[base + axis_i * inner];
                if (v > max_value) {
                    max_value = v;
                }
            }

            float sum = 0.0f;
            for (size_t axis_i = 0; axis_i < axis_size; ++axis_i) {
                float shifted = input->data[base + axis_i * inner] - max_value;
                float e = expf(shifted);
                out->data[base + axis_i * inner] = e;
                sum += e;
            }

            float inv = 1.0f / sum;
            for (size_t axis_i = 0; axis_i < axis_size; ++axis_i) {
                out->data[base + axis_i * inner] *= inv;
            }
        }
    }

    return MIR_OK;
}
