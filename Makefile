export HIP_PLATFORM=nvidia

ifdef USE_HIPBLAS
CFLAGS=-DUSE_HIPBLAS
LD_FLAGS=-L/opt/rocm/lib
LIBS=-lhipblas
else
LIBS=-lcublas
endif

.PHONY: all
all: axpy gemm

axpy: axpy.cpp
	hipcc $< -o $@ ${CFLAGS} ${LD_FLAGS} ${LIBS}

gemm: gemm.cpp
	hipcc $< -o $@ ${CFLAGS} ${LD_FLAGS} ${LIBS}

.PHONY: clean
clean:
	rm -f axpy gemm
