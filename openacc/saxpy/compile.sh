#!/bin/bash
#
pgcc -acc -ta=nvidia -Minfo=accel -o saxpy_acc.o main.c

#-Minline
#-Minline=levels:<N>

# run time info

export PGI_ACC_TIME=1
