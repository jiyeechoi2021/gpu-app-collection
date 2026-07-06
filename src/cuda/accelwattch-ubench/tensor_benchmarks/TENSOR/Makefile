EXECUTABLE := TENSOR
CUFILES := tensorcore.cu

ROOT_DIR := $(abspath ../..)
BINDIR := $(ROOT_DIR)/bin/linux/release
OBJDIR := obj/x86_64/release

DETECTED_NVCC := $(shell command -v nvcc 2>/dev/null)
NVCC ?= $(if $(DETECTED_NVCC),$(DETECTED_NVCC),/usr/local/cuda/bin/nvcc)

DETECTED_CUDA_HOME := $(shell dirname "$$(dirname "$$(command -v $(NVCC) 2>/dev/null || printf '%s\n' "$(NVCC)")")")
CUDA_HOME ?= $(DETECTED_CUDA_HOME)

DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.' | awk '{print $$1}')
CUDA_SM ?= $(if $(DETECTED_SM),$(DETECTED_SM),70)

NVCCFLAGS ?= -O3 -std=c++17 --expt-relaxed-constexpr
NVCCFLAGS += -arch=sm_$(CUDA_SM) -I$(CUDA_HOME)/include
LDLIBS ?= -L$(CUDA_HOME)/lib64 -lcublas -lcurand

.PHONY: all release clean info

all: release

release: $(BINDIR)/$(EXECUTABLE)

$(BINDIR)/$(EXECUTABLE): $(OBJDIR)/tensorcore.cu.o Makefile | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

$(OBJDIR)/tensorcore.cu.o: $(CUFILES) Makefile | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BINDIR) $(OBJDIR):
	mkdir -p $@

info:
	@printf 'NVCC=%s\n' '$(NVCC)'
	@printf 'CUDA_HOME=%s\n' '$(CUDA_HOME)'
	@printf 'CUDA_SM=%s\n' '$(CUDA_SM)'
	@printf 'OUTPUT=%s\n' '$(BINDIR)/$(EXECUTABLE)'

clean:
	rm -rf $(OBJDIR) $(BINDIR)/$(EXECUTABLE)
