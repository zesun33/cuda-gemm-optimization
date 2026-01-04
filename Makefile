# CUDA GEMM Optimization - Makefile

# Compiler
NVCC = nvcc

# Compilation flags
NVCCFLAGS = -O3 -std=c++11

# Architecture (change based on your GPU)
# Common values:
#   sm_70: V100
#   sm_75: RTX 2080, Titan RTX
#   sm_80: A100
#   sm_86: RTX 3090, RTX 3080
#   sm_89: RTX 4090
ARCH = sm_86

# Directories
SRC_DIR = src
BUILD_DIR = build

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Executables (one per .cu file)
EXECUTABLES = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(SOURCES))

# Default target
all: $(BUILD_DIR) $(EXECUTABLES)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile each .cu file to executable
$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
	@echo "Built: $@"

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Run naive GEMM
run_naive: $(BUILD_DIR)/01_naive_gemm
	@echo "Running Naive GEMM (1024x1024x1024)..."
	./$(BUILD_DIR)/01_naive_gemm 1024 1024 1024

# Help
help:
	@echo "CUDA GEMM Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build all kernels (default)"
	@echo "  clean       - Remove build directory"
	@echo "  run_naive   - Run naive GEMM with default size"
	@echo "  help        - Show this message"
	@echo ""
	@echo "Variables:"
	@echo "  ARCH        - GPU architecture (default: sm_80 for A100)"
	@echo "                Set with: make ARCH=sm_86"

.PHONY: all clean run_naive help
