export HIP_PLATFORM=nvidia

LD_FLAGS=-L/opt/rocm/lib

.PHONY: all
all: axpy

axpy: axpy.cpp
	hipcc $< -o $@ ${LD_FLAGS} -lhipblas

.PHONY: clean
clean:
	rm -f axpy
