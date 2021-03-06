#ifndef CUIMG_COPY_H_
# define CUIMG_COPY_H_

# include <cstring>
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/cpu/host_image2d.h>
# include <cuimg/gpu/device_image3d.h>
# include <cuimg/cpu/host_image3d.h>
# include <cuimg/error.h>

namespace cuimg
{
#ifndef NO_CUDA
  template <typename T>
  void copy(const device_image2d<T>& in, host_image2d<T>& out)
  {
    assert(in.domain() == out.domain());
    if (in.border() == out.border())
      cudaMemcpy2D(out.data(), out.pitch(), in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows(),
		   cudaMemcpyDeviceToHost);
    else
      for (unsigned i = 0; i < in.nrows(); i++)
	cudaMemcpy(out.row(i), in.row(i), in.ncols() * sizeof(T), cudaMemcpyDeviceToHost);

    check_cuda_error();

  }
#endif

  template <typename T>
  void copy(const host_image2d<T>& in, host_image2d<T>& out)
  {
    assert(in.domain() == out.domain());
    if (in.border() == out.border())
      memcpy(out.data(), in.data(), in.nrows() * in.pitch());
    else
      for (unsigned i = 0; i < in.nrows(); i++)
	memcpy(out.row(i), in.row(i), in.ncols() * sizeof(T));

  }


  template <typename T, typename U>
  void copy(const host_image2d<T>& in, host_image2d<U>& out)
  {
    assert(in.domain() == out.domain());
    for (unsigned r = 0; r < in.nrows(); r++)
    for (unsigned c = 0; c < in.ncols(); c++)
      out(r, c) = in(r, c);
  }

#ifndef NO_CUDA

  template <typename T>
  void copy(const host_image2d<T>& in, device_image2d<T>& out)
  {
    assert(in.domain() == out.domain());
    if (in.border() == out.border())
    cudaMemcpy2D(out.data(), out.pitch(), in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows(),
               cudaMemcpyHostToDevice);
    else
      for (unsigned i = 0; i < in.nrows(); i++)
	cudaMemcpy(out.row(i), in.row(i), in.ncols() * sizeof(T), cudaMemcpyHostToDevice);
    check_cuda_error();
  }

  template <typename T>
  void copy(const device_image2d<T>& in, device_image2d<T>& out)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy2D(out.data(), out.pitch(), in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows(),
               cudaMemcpyDeviceToDevice);
    check_cuda_error();
  }

  template <typename T>
  void copy(const device_image3d<T>& in, host_image3d<T>& out)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy3DParms p = {0};
    //memset(&p, 0, sizeof(cudaMemcpy3DParms));
    p.srcPtr = make_cudaPitchedPtr((void*)in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows());
    p.dstPtr = make_cudaPitchedPtr((void*)out.data(), out.pitch(), out.ncols() * sizeof(T), out.nrows());
    p.extent.width = in.ncols() * sizeof(T);
    p.extent.height = in.nrows();
    p.extent.depth = in.nslices();
    p.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&p);
    check_cuda_error();
  }

  template <typename T>
  void copy(const host_image3d<T>& in, device_image3d<T>& out)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy3DParms p = {0};
    //memset(&p, 0, sizeof(cudaMemcpy3DParms));
    p.srcPtr = make_cudaPitchedPtr((void*)in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows());
    p.dstPtr = make_cudaPitchedPtr((void*)out.data(), out.pitch(), out.ncols() * sizeof(T), out.nrows());
    p.extent.width = in.ncols() * sizeof(T);
    p.extent.height = in.nrows();
    p.extent.depth = in.nslices();
    p.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p);
    check_cuda_error();
  }

  template <typename T>
  void copy(const device_image3d<T>& in, device_image3d<T>& out)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy3DParms p = {0};
    //memset(&p, 0, sizeof(cudaMemcpy3DParms));
    p.srcPtr = make_cudaPitchedPtr((void*)in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows());
    p.dstPtr = make_cudaPitchedPtr((void*)out.data(), out.pitch(), out.ncols() * sizeof(T), out.nrows());
    p.extent.width = in.ncols() * sizeof(T);
    p.extent.height = in.nrows();
    p.extent.depth = in.nslices();
    p.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&p);
    check_cuda_error();
  }

  template <typename T>
  void copy(const device_image3d<T>& in, device_image2d<T>& out, unsigned slice)
  {
    assert(in.nrows() == out.nrows() && in.ncols() == out.ncols());
    cudaMemcpy2D(out.data(), out.pitch(), in.slice_data(slice), in.pitch(),
                 in.ncols() * sizeof(T), in.nrows(),
                 cudaMemcpyDeviceToDevice);
    check_cuda_error();
  }


  template <typename T>
  void copy_async(const device_image2d<T>& in, host_image2d<T>& out, cudaStream_t stream = 0)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy2DAsync(out.data(), out.pitch(), in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows(),
               cudaMemcpyDeviceToHost, stream);
    check_cuda_error();
  }

#endif

  template <typename T>
  void copy_async(const host_image2d<T>& in, host_image2d<T>& out, cudaStream_t stream = 0)
  {
    assert(in.domain() == out.domain());
    memcpy(out.data(), in.data(), in.nrows() * in.pitch());
  }

#ifndef NO_CUDA

  template <typename T>
  void copy_async(const host_image2d<T>& in, device_image2d<T>& out, cudaStream_t stream = 0)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy2DAsync(out.data(), out.pitch(), in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows(),
               cudaMemcpyHostToDevice, stream);
    check_cuda_error();
  }

  template <typename T>
  void copy_async(const device_image2d<T>& in, device_image2d<T>& out, cudaStream_t stream = 0)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy2DAsync(out.data(), out.pitch(), in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows(),
                      cudaMemcpyDeviceToDevice, stream);
    check_cuda_error();
  }

  template <typename T>
  void copy_async(const device_image3d<T>& in, host_image3d<T>& out, cudaStream_t stream = 0)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy3DParms p = {0};
    //memset(&p, 0, sizeof(cudaMemcpy3DParms));
    p.srcPtr = make_cudaPitchedPtr((void*)in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows());
    p.dstPtr = make_cudaPitchedPtr((void*)out.data(), out.pitch(), out.ncols() * sizeof(T), out.nrows());
    p.extent.width = in.ncols() * sizeof(T);
    p.extent.height = in.nrows();
    p.extent.depth = in.nslices();
    p.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3DAsync(&p, stream);
    check_cuda_error();
  }

  template <typename T>
  void copy_async(const host_image3d<T>& in, device_image3d<T>& out, cudaStream_t stream = 0)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy3DParms p = {0};
    //memset(&p, 0, sizeof(cudaMemcpy3DParms));
    p.srcPtr = make_cudaPitchedPtr((void*)in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows());
    p.dstPtr = make_cudaPitchedPtr((void*)out.data(), out.pitch(), out.ncols() * sizeof(T), out.nrows());
    p.extent.width = in.ncols() * sizeof(T);
    p.extent.height = in.nrows();
    p.extent.depth = in.nslices();
    p.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3DAsync(&p, stream);
    check_cuda_error();
  }

  template <typename T>
  void copy_async(const device_image3d<T>& in, device_image3d<T>& out, cudaStream_t stream = 0)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy3DParms p = {0};
    //memset(&p, 0, sizeof(cudaMemcpy3DParms));
    p.srcPtr = make_cudaPitchedPtr((void*)in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows());
    p.dstPtr = make_cudaPitchedPtr((void*)out.data(), out.pitch(), out.ncols() * sizeof(T), out.nrows());
    p.extent.width = in.ncols() * sizeof(T);
    p.extent.height = in.nrows();
    p.extent.depth = in.nslices();
    p.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3DAsync(&p, stream);
    check_cuda_error();
  }

  template <typename T>
  void copy_async(const device_image3d<T>& in, device_image2d<T>& out,
                  unsigned slice, cudaStream_t stream = 0)
  {
    assert(in.nrows() == out.nrows() && in.ncols() == out.ncols());
    cudaMemcpy2DAsync(out.data(), out.pitch(), in.slice_data(slice), in.pitch(),
                 in.ncols() * sizeof(T), in.nrows(),
                 cudaMemcpyDeviceToDevice, stream);
    check_cuda_error();
  }

#endif

}

#endif
