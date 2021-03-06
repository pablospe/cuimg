// -*- mode: cuda-mode -*-

#ifndef CUIMG_IMAGE2D_H_
# define CUIMG_IMAGE2D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/architectures.h>
# include <boost/shared_ptr.hpp>
# include <cuimg/target.h>
# include <cuimg/concepts.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/dsl/expr.h>

namespace cuimg
{

  struct cuda_gpu;

  template <typename V>
  class kernel_image2d;

  template <typename V>
  class device_image2d : public Image2d<device_image2d<V> >
  {
  public:
    static const cuimg::target target = GPU;
    typedef cuda_gpu architecture;
    typedef int is_expr;
    typedef boost::shared_ptr<V> PT;
    typedef V value_type;
    typedef point2d<int> point;
    typedef obox2d domain_type;
    typedef kernel_image2d<V> kernel_type;

    inline device_image2d();
    inline device_image2d(unsigned nrows, unsigned ncols, unsigned _border = 0);
    inline device_image2d(V* data, unsigned nrows, unsigned ncols, unsigned pitch);
    inline device_image2d(const domain_type& d, unsigned border = 0);

    void swap(device_image2d<V>& o);

    __host__ __device__ inline device_image2d(const device_image2d<V>& d);

    __host__ __device__ inline device_image2d<V>& operator=(const device_image2d<V>& d);

    template <typename E>
    __host__ inline device_image2d<V>& operator=(const expr<E>& e);

    __host__ __device__ inline const domain_type& domain() const;
    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;
    __host__ __device__ inline unsigned border() const { return border_; }
    __host__ __device__ inline size_t pitch() const;

    __host__ inline V read_back_pixel(const point& p) const;
    __host__ inline void set_pixel(const point& p, const V& e);

    __host__ __device__ inline V* data() const;
    __host__ __device__ inline V* begin() const;
    __host__ __device__ inline V* end() const;
    //__host__ __device__ inline const V* data() const;

    __host__ __device__ inline bool has(const point& p) const;
    __host__ __device__ inline i_int2 index_to_point(unsigned int idx) const;

    __host__ __device__ inline const PT data_sptr() const;
    __host__ __device__ inline PT data_sptr();

    __host__ __device__ inline V* row(int i);
    __host__ __device__ inline const V* row(int i) const;

  private:
    domain_type domain_;
    unsigned border_;
    size_t pitch_;
    PT data_;
    V* begin_;
    V* data_ptr_;
  };


  template <typename T>
  struct return_type;

  template <typename V>
  struct return_type<device_image2d<V> >
  {
    typedef typename device_image2d<V>::value_type ret;
  };

}

# include <cuimg/gpu/kernel_image2d.h>

# include <cuimg/gpu/device_image2d.hpp>

#endif
