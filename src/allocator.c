#include "mir.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

MirStatus mir_arena_init(MirArena *arena, size_t capacity) {
    if (!arena || capacity == 0) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    memset(arena, 0, sizeof(*arena));
    arena->base = (uint8_t *)malloc(capacity);
    if (!arena->base) {
        return MIR_ERR_OUT_OF_MEMORY;
    }

    arena->capacity = capacity;
    arena->owns_memory = true;
    return MIR_OK;
}

MirStatus mir_arena_from_buffer(MirArena *arena, void *buffer, size_t capacity) {
    if (!arena || !buffer || capacity == 0) {
        return MIR_ERR_INVALID_ARGUMENT;
    }

    memset(arena, 0, sizeof(*arena));
    arena->base = (uint8_t *)buffer;
    arena->capacity = capacity;
    arena->owns_memory = false;
    return MIR_OK;
}

void *mir_arena_alloc(MirArena *arena, size_t size, size_t alignment) {
    if (!arena || !arena->base || size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return NULL;
    }

    uintptr_t base = (uintptr_t)arena->base;
    uintptr_t current = base + arena->offset;
    uintptr_t aligned = (current + alignment - 1) & ~(uintptr_t)(alignment - 1);
    size_t next_offset = (size_t)(aligned - base) + size;
    if (next_offset > arena->capacity) {
        return NULL;
    }

    arena->offset = next_offset;
    return (void *)aligned;
}

void mir_arena_reset(MirArena *arena) {
    if (!arena) {
        return;
    }
    arena->offset = 0;
}

void mir_arena_free(MirArena *arena) {
    if (!arena) {
        return;
    }
    if (arena->owns_memory && arena->base) {
        free(arena->base);
    }
    memset(arena, 0, sizeof(*arena));
}
