#include "mir.h"

MirStatus mir_op_matmul(const MirTensor *a, const MirTensor *b, MirTensor *out) {
    if (!a || !b || !out || !a->data || !b->data) {
        return MIR_ERR_INVALID_ARGUMENT;
    }
    if (a->shape.rank != 2 || b->shape.rank != 2) {
        return MIR_ERR_UNSUPPORTED;
    }

    int64_t m = a->shape.dims[0];
    int64_t k = a->shape.dims[1];
    int64_t k2 = b->shape.dims[0];
    int64_t n = b->shape.dims[1];
    if (k != k2) {
        return MIR_ERR_SHAPE_MISMATCH;
    }

    int64_t out_dims[2] = {m, n};
    MirShape out_shape;
    MirStatus status = mir_shape_make(out_dims, 2, &out_shape);
    if (status != MIR_OK) {
        return status;
    }

    status = mir_tensor_resize(out, out_shape);
    if (status != MIR_OK) {
        return status;
    }

    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                float lhs = a->data[row * k + i];
                float rhs = b->data[i * n + col];
                sum += lhs * rhs;
            }
            out->data[row * n + col] = sum;
        }
    }

    return MIR_OK;
}
