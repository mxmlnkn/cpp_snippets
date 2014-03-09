del md5cuda.exe
@rem -arch=sm_20 needed for printf in __device__ functions
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc.exe" md5cuda.cu -o md5cuda.exe --compiler-bindir "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" --machine 32 -arch=sm_20
@rem --compiler-options "/O2"
@rem Create precossed files to check what kind of optimziations like #pragma unroll are done
@rem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc.exe" md5.cu -o md5.exe --compiler-bindir "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" --machine 32 --compiler-options "/O2 /C /P"

@rem nvcc matrixmult.c -o matrixmult --compiler-options "-std=c99 -O2  -C -E -P"
md5cuda.exe