@echo off
IF EXIST %VS110COMNTOOLS% (
	CALL "%VS110COMNTOOLS%\vsvars32.bat"
@echo on
nvcc -arch=sm_20 -m32 -dc md5main.cpp
nvcc -arch=sm_20 -m32 -dc md5kernel.cu
nvcc -arch=sm_20 -m32     md5main.obj md5kernel.obj -o md5main.exe
.\md5main.exe
