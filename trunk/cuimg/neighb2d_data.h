#ifndef CUIMG_NEIGHB2D_DATA_H_
# define CUIMG_NEIGHB2D_DATA_H_

# include <cuda_runtime.h>

namespace cuimg
{
  __constant__ const int c4[4][2] =
  {
    {0, 1},
    {-1, 0}, {1, 0},
    {0, -1}
  };

  __constant__ const int c5[5][2] =
  {
    {0, 1},
    {-1, 0}, {0, 0}, {1, 0},
    {0, -1}
  };


  __constant__ const int c8[8][2] =
  {
    {-1, 1}, {0, 1}, {1, 1},
    {-1, 0},         {1, 0},
    {-1, -1}, {0, -1}, {1, -1}
  };

  __constant__ const int c9[9][2] =
  {
    {-1, 1}, {0, 1}, {1, 1},
    {-1, 0}, {0, 0}, {1, 0},
    {-1, -1}, {0, -1}, {1, -1}
  };

  __constant__ const int c24[24][2] =
  {
    {-2, 2},  {-1,  2}, {0,  2}, {1,  2}, {2,  2},
    {-2, 1},  {-1,  1}, {0,  1}, {1,  1}, {2,  1},
    {-2, 0},  {-1,  0},          {1,  0}, {2,  0},
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}
  };

  __constant__ const int c25[25][2] =
  {
    {-2, 2},  {-1,  2}, {0,  2}, {1,  2}, {2,  2},
    {-2, 1},  {-1,  1}, {0,  1}, {1,  1}, {2,  1},
    {-2, 0},  {-1,  0}, {0,  0}, {1,  0}, {2,  0},
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}
  };

  const int c4_h[4][2] =
  {
    {0, 1},
    {-1, 0}, {1, 0},
    {0, -1}
  };

  const int c5_h[5][2] =
  {
    {0, 1},
    {-1, 0}, {0, 0}, {1, 0},
    {0, -1}
  };


  const int c8_h[8][2] =
  {
    {-1, 1}, {0, 1}, {1, 1},
    {-1, 0},         {1, 0},
    {-1, -1}, {0, -1}, {1, -1}
  };

  const int c9_h[9][2] =
  {
    {-1, 1}, {0, 1}, {1, 1},
    {-1, 0}, {0, 0}, {1, 0},
    {-1, -1}, {0, -1}, {1, -1}
  };

  const int c24_h[24][2] =
  {
    {-2, 2},  {-1,  2}, {0,  2}, {1,  2}, {2,  2},
    {-2, 1},  {-1,  1}, {0,  1}, {1,  1}, {2,  1},
    {-2, 0},  {-1,  0},          {1,  0}, {2,  0},
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}
  };

  const int c25_h[25][2] =
  {
    {-2, 2},  {-1,  2}, {0,  2}, {1,  2}, {2,  2},
    {-2, 1},  {-1,  1}, {0,  1}, {1,  1}, {2,  1},
    {-2, 0},  {-1,  0}, {0,  0}, {1,  0}, {2,  0},
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}
  };

#define for_all_in_static_neighb2d(p, n, dps) \
  neighb_iterator2d<static_neighb2d<sizeof(dps) / (2 * sizeof(int))> > n(p, dps); \
  for_all(n)

}
#endif