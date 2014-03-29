### Basic Makefile for MPI

CC      = nvcc
CFLAGS  = -arch=sm_20 --compiler-options "" --ptxas-options=-v
# --debug --device-debug
# -C -E -P"

PROGRAM = md5cm

all:    ${PROGRAM}

#$< gives dependant which is mor recent than file
${PROGRAM}:	${PROGRAM}.cu ${PROGRAM}.h
	rm -f $(basename $<)
	${CC} $< ${CFLAGS} -o $(basename $<)
	./$(basename $<)

clean:
	rm -f ${PROGRAM}

#ToDo: correct distance with cell distance!
