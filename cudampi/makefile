CC      = nvcc
NVCC    = nvcc
MPICPP  = mpic++
NCFLAGS = -arch=sm_20 --compiler-options "" -Xptxas -v
MCFLAGS = 
# --debug --device-debug
# -C -E -P"

MPI_CFLAGS = $(shell mpicc --showme:compile)
MPI_LFLAGS = $(shell mpicc --showme:link)

PROGRAM  = md5cuda
#MPI_ROOT only needed if mpi and cuda in same sourcefile, else the two objects can be linked
ifeq ($(OS),Windows_NT)
	IF EXIST %VS110COMNTOOLS% (
		CALL "%VS110COMNTOOLS%\vsvars32.bat"
	)
	MPI_ROOT = 
	NCFLAGS += -m32
	MCFLAGS += -m32
else
	MPI_ROOT = /opt/pkg/mpi/openmpi/1.6.0/intel/12.0/64/opt/
endif

all:    md5main statistics.hpp

md5kernel.o:	md5kernel.cu md5kernel.hpp statistics.hpp
	${NVCC} $< ${NCFLAGS} -dc -o $(basename $<).o -I ${MPI_ROOT}/include/ -L ${MPI_ROOT}/lib/ -l mpi
#nvcc md5kernel.cu -o md5kernel.o -arch=sm_20 -c

md5main.o:	md5main.cpp md5main.hpp
	${NVCC} ${NCFLAGS} -dc $< -o $@

md5main:	md5main.o md5kernel.o
	${NVCC} ${NCFLAGS} $^
	./$@

clean:
	rm -f ${PROGRAM}
	
#nvcc -arch=sm_20 -m32 -dc md5main.cpp
#nvcc -arch=sm_20 -m32 -dc md5kernel.cu
#nvcc -arch=sm_20 -m32     md5main.obj md5kernel.obj -o md5main.exe

#round command was included in cuda.h! can't find it anywhere else Oo used floor instead
