CC ?= cc
CFLAGS ?= -std=c11 -O2 -g -Wall -Wextra -Wpedantic -Werror
LDFLAGS ?=

BUILD_DIR := build
SRC := \
	src/tensor.c \
	src/graph.c \
	src/runtime.c \
	src/allocator.c \
	src/ops/matmul.c \
	src/ops/add.c \
	src/ops/relu.c \
	src/ops/softmax.c \
	src/ops/reshape.c \
	src/ops/transpose.c
OBJ := $(SRC:%.c=$(BUILD_DIR)/%.o)

TEST_BIN := $(BUILD_DIR)/test_ops
LIB := $(BUILD_DIR)/libmir.a

.PHONY: all clean test

all: $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/src:
	mkdir -p $(BUILD_DIR)/src

$(BUILD_DIR)/src/ops:
	mkdir -p $(BUILD_DIR)/src/ops

$(BUILD_DIR)/tests:
	mkdir -p $(BUILD_DIR)/tests

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR) $(BUILD_DIR)/src $(BUILD_DIR)/src/ops
	$(CC) $(CFLAGS) -Iinclude -c $< -o $@

$(LIB): $(OBJ)
	ar rcs $@ $(OBJ)

$(TEST_BIN): tests/test_ops.c $(OBJ) | $(BUILD_DIR)/tests
	$(CC) $(CFLAGS) -Iinclude $^ -o $@ $(LDFLAGS) -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
