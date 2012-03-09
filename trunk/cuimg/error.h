#ifndef CUIMG_ERROR_H_
# define CUIMG_ERROR_H_

# include <cuimg/gpu/cuda.h>
# include <iostream>
# include <cassert>

namespace cuimg
{


  inline void check_cuda_error()
  {
# ifndef NO_CUDA
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      std::cerr << "Cuda error: " << cudaGetErrorString(error) << std::endl;
      exit(1);
      assert(error == cudaSuccess);
    }
# endif
  }

}

#endif
