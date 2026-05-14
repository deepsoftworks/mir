#include "mir.h"

#include <stdlib.h>
#include <string.h>

static MirStatus mir_compute_strides(const MirShape *shape, int64_t *out_strides) {
    if (!shape || !out_strides) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    if (shape->rank > MIR_MAX_DIMS) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    if (shape->rank == 0) {
        return MIR_OK;
    }

    out_strides[shape->rank - 1] = 1;
    for (int i = (int)shape->rank - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * shape->dims[i + 1];
    }
    return MIR_OK;
}

const char *mir_status_str(MirStatus status) {
    switch (status) {
    case MIR_OK:
        return "ok";
    case MIR_ERR_INVALID_ARGUMENT:
        return "invalid_argument";
    case MIR_ERR_OUT_OF_MEMORY:
        return "out_of_memory";
    case MIR_ERR_SHAPE_MISMATCH:
        return "shape_mismatch";
    case MIR_ERR_UNSUPPORTED:
        return "unsupported";
    case MIR_ERR_RUNTIME:
        return "runtime_error";
    default:
        return "unknown";
    }
}

MirStatus mir_shape_make(const int64_t *dims, uint32_t rank, MirShape *out_shape) {
    if (!out_shape || rank > MIR_MAX_DIMS) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    memset(out_shape, 0, sizeof(*out_shape));
    out_shape->rank = rank;
    for (uint32_t i = 0; i < rank; ++i) {
        if (!dims || dims[i] <= 0) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        out_shape->dims[i] = dims[i];
    }
    return MIR_OK;
}

size_t mir_shape_elem_count(const MirShape *shape) {
    if (!shape) {
        return 0;
    }

    if (shape->rank == 0) {
        return 1;
    }

    size_t total = 1;
    for (uint32_t i = 0; i < shape->rank; ++i) {
        if (shape->dims[i] <= 0) {
            return 0;
        }
        total *= (size_t)shape->dims[i];
    }
    return total;
}

bool mir_shape_equal(const MirShape *a, const MirShape *b) {
    if (!a || !b) {
        return false;
    }
    if (a->rank != b->rank) {
        return false;
    }
    for (uint32_t i = 0; i < a->rank; ++i) {
        if (a->dims[i] != b->dims[i]) {
            return false;
        }
    }
    return true;
}

MirStatus mir_tensor_init(MirTensor *tensor, MirShape shape, bool allocate_data) {
    if (!tensor) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    memset(tensor, 0, sizeof(*tensor));
    tensor->dtype = MIR_DTYPE_F32;
    tensor->shape = shape;
    tensor->elem_count = mir_shape_elem_count(&shape);
    tensor->bytes = tensor->elem_count * sizeof(float);

    MirStatus status = mir_compute_strides(&shape, tensor->strides);
    if (status != MIR_OK) {
        return status;
    }

    if (!allocate_data) {
        return MIR_OK;
    }

    if (tensor->bytes == 0) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    tensor->data = (float *)calloc(1, tensor->bytes);
    if (!tensor->data) {
        return MIR_ERR_OUT_OF_MEMORY;
    }
    tensor->owns_data = true;
    return MIR_OK;
}

MirStatus mir_tensor_init_with_data(MirTensor *tensor, MirShape shape, float *data, bool owns_data) {
    if (!tensor || !data) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    MirStatus status = mir_tensor_init(tensor, shape, false);
    if (status != MIR_OK) {
        return status;
    }

    tensor->data = data;
    tensor->owns_data = owns_data;
    return MIR_OK;
}

MirStatus mir_tensor_resize(MirTensor *tensor, MirShape shape) {
    if (!tensor) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    size_t elem_count = mir_shape_elem_count(&shape);
    size_t bytes = elem_count * sizeof(float);
    MirStatus status = mir_compute_strides(&shape, tensor->strides);
    if (status != MIR_OK) {
        return status;
    }

    if (tensor->owns_data) {
        if (bytes == 0) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        float *resized = (float *)realloc(tensor->data, bytes);
        if (!resized) {
            return MIR_ERR_OUT_OF_MEMORY;
        }
        tensor->data = resized;
    } else {
        if (bytes == 0) {
            return MIR_ERR_INVALID_ARGUMENT;
        }
        tensor->data = (float *)calloc(1, bytes);
        if (!tensor->data) {
            return MIR_ERR_OUT_OF_MEMORY;
        }
        tensor->owns_data = true;
    }

    tensor->dtype = MIR_DTYPE_F32;
    tensor->shape = shape;
    tensor->elem_count = elem_count;
    tensor->bytes = bytes;
    return MIR_OK;
}

MirStatus mir_tensor_view(MirTensor *tensor, const MirTensor *source, MirShape shape) {
    if (!tensor || !source || !source->data) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    if (mir_shape_elem_count(&shape) != source->elem_count) {
        return MIR_ERR_SHAPE_MISMATCH;
    }

    if (tensor->owns_data && tensor->data) {
        free(tensor->data);
    }

    memset(tensor, 0, sizeof(*tensor));
    tensor->dtype = source->dtype;
    tensor->shape = shape;
    tensor->elem_count = source->elem_count;
    tensor->bytes = source->bytes;
    tensor->data = source->data;
    tensor->owns_data = false;

    return mir_compute_strides(&shape, tensor->strides);
}

void mir_tensor_free(MirTensor *tensor) {
    if (!tensor) {
        return;
    }
    if (tensor->owns_data && tensor->data) {
        free(tensor->data);
    }
    memset(tensor, 0, sizeof(*tensor));
}

void mir_tensor_dump(const MirTensor *tensor, FILE *out, size_t max_elements) {
    if (!tensor || !out) {
        return;
    }

    fprintf(out, "tensor(shape=[");
    for (uint32_t i = 0; i < tensor->shape.rank; ++i) {
        fprintf(out, "%lld%s", (long long)tensor->shape.dims[i], i + 1 < tensor->shape.rank ? "," : "");
    }
    fprintf(out, "], elems=%zu)\n", tensor->elem_count);

    if (!tensor->data) {
        fprintf(out, "  data=<null>\n");
        return;
    }

    size_t limit = tensor->elem_count < max_elements ? tensor->elem_count : max_elements;
    fprintf(out, "  data=[");
    for (size_t i = 0; i < limit; ++i) {
        fprintf(out, "%.6f%s", tensor->data[i], i + 1 < limit ? ", " : "");
    }
    if (limit < tensor->elem_count) {
        fprintf(out, ", ...");
    }
    fprintf(out, "]\n");
}
