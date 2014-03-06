### Basic Makefile for MPI

CC      = mpicc
CFLAGS  = -g -O0 -std=c99
LD      = mpicc
LDFLAGS = -g

PROGRAM = nbody

all:    ${PROGRAM}.exe

${PROGRAM}.exe:         ${PROGRAM}.o
	${LD} ${LDFLAGS} $< -o ${PROGRAM}.exe

${PROGRAM}.o:           ${PROGRAM}.c
	${CC} ${CFLAGS} -c $< -o ${PROGRAM}.o

clean:
	rm -f ${PROGRAM}.o ${PROGRAM}.exe

#Results in: mpicc -g -O0 -c mp_test.c -o mpi_test.o
#            mpicc -g mp_test.o -o mpi_test.exe
