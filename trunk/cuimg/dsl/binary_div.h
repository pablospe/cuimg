#ifndef CUIMG_GPU_BINARY_DIV_H_
# define CUIMG_GPU_BINARY_DIV_H_

# include <cuimg/meta.h>
# include <cuimg/dsl/binary_op.h>

namespace cuimg
{
  struct binary_div {};

  template <>
  struct evaluator<binary_div>
  {
    template <typename A, typename B>
      static inline __host__ __device__
      typename return_type<binary_op<binary_div, A, B, 0, 0> >::ret
      run(A& a, B& b, point2d<int> p)
    {
      return eval(a, p) / eval(b, p);
    }

  };

  template <typename A1, typename A2, unsigned P1, unsigned P2>
  struct return_type<binary_op<binary_div, A1, A2, P1, P2> >
  {
    typedef typename meta::type_div<typename return_type<A1>::ret,
                                     typename return_type<A2>::ret>::ret ret;
  };

  template <typename A, template <class> class AP, typename S>
  inline
  typename binary_op_<binary_div, image2d<A, AP>, S>::ret
  operator/(const image2d<A, AP>& a, const S s)
  {
    typedef typename binary_op_<binary_div, image2d<A, AP>, S>::ret return_type;
    return return_type(a, s);
  }

  template <typename E, typename S>
  inline
  typename first<typename binary_op_<binary_div, E, S>::ret, typename E::is_expr>::ret
  operator/(E& a, const S s)
  {
    typedef typename binary_op_<binary_div, E, S>::ret return_type;
    return return_type(a, s);
  }

}

#endif
