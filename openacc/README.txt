

#pragma acc <directives>
============================================

#pragma acc data ...
... copy(A), create(Anew)

#pragma acc kernels ...
... reduction(max:err)
... loop independent
... async(1)
  -> do not wait for kernel completion
-> implied barrier at the end of a kernel
   region, with not async

#pragma acc wait(2) or wait
... wait an all outstanding async calls
    or only on kernel 2

#pragma acc parallel
... loop
-> fine grained parallelization
-> for example: just for the next loop
... async(2)
  -> (potentially) execute concurrently
     with kernels

#pragma acc update ...
... host(a)
  -> sync dev2host
... device(a)
  -> sync host2dev
... host( a(...) )
  -> slice update only (contiguous chunks!)
  -> may compress to a linear buffer first
     (for example for boundaries)


OpenACC Execution Model
============================================
- three levels:
    gang
    |_worker
      |_vector
- allows mapping to an architecture that is
  organized in differnt PE lvls
- CUDA right now (5.5):
  - gnag  ==block
  - worker==warp
  - vector==threads
  if one ignores worker, than gang  ==block,
                              vector==threads
  of a block.


Interop.
=============================================
- connections to CUDA and libs are available
- for example openacc device data can be used
  by a FFT lib without copying it back to the
  host

#pragma acc data deviceptr(d_input)
  declares that the pointer is (?) already on
  the device

#pragma acc use_device( list, ... )
  tells the compiler to use the device adress
  for any var in the list

Advice
=============================================
- (nested) for loops are best for parallelization
- large loop count needed to overcome transfer overhead
- fixed bound loop counts!
- help compiler: __retrict__ keyword, independet clause
- avoid pointer arithmetic
  - use subscripted arrays rather than pointer-indexed
- function calls must be inlineable if in acc region


Tips and Tricks
=============================================
- contoguous mem for multi-dim arr
- use data regions, nested possible - as far out as possible
- kernels are not possible



OpenGL
=============================================

Roadmap:
  Init: - create OpenGL buffer
        - inform CUDA about buffer
          #pragma acc data copy(u,v)
        - glutMainLoop()
  _
  |
  v 

  GLUT (UI, Eventloop)

  ^
  |
  v
  _
  | Map OpenGL buffer to CUDA
  |
  | #pragma acc kernels deviceptr(vboPtr)
  | Modify buffer in OpenACC
  | Return control over buffer back to OpenGL
  v glDrawElements()

