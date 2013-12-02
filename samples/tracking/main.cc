#include <iostream>
#include <vector>
#include <sys/time.h>

#include <cuimg/profiler.h>
#include <cuimg/dsl/all.h>
#include <cuimg/cpu/host_image2d.h>
#include <cuimg/draw.h>

#include <cuimg/tracking2/tracker.h>

using namespace cuimg;

int64_t get_systemtime_usecs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (int64_t) tv.tv_sec * 1000000LL + (int64_t) tv.tv_usec;
}

// trajectory store a short term trajectory.
struct trajectory
{
  trajectory() : alive(true) {}
  trajectory(i_int2 pos) : alive(true) { history.push_back(pos); }

  void move(trajectory&& t)
  {
    history.swap(t.history);
    alive = t.alive;
  }

  std::deque<i_int2> history;
  bool alive;
};


// Update a trajectory when a particle moves.
template <typename TR>
void update_trajectories(std::vector<trajectory>& v, TR& pset)
{
  const auto& parts = pset.dense_particles();
  for(unsigned i = 0; i < v.size(); i++)
    v[i].alive = false;
  for(unsigned i = 0; i < v.size(); i++)
    if (parts[i].age > 0)
    {
      assert(parts[i].age != 1 || v[i].history.empty());
      v[i].history.push_back(parts[i].pos);
      v[i].alive = true;
      if (v[i].history.size() > 20) v[i].history.pop_front();
    }
    else
    {
      v[i].history.clear();
    }
}

int main(int argc, char* argv[])
{
  cv::VideoCapture video;

  if (argc == 4)
  {
    video.open(argv[3]);
  }
  else if (argc == 3)
  {
    // try open camera
    video.open(0);
  }
  else
  {
    std::cout << "Usage: ./tracking_sample nscales detector_threshold [video_file]" << std::endl;
    std::cout << "Hint for a first run: nscale = 3, detector_threshold = 20 " << std::endl;
    return -1;
  }

  if(!video.isOpened())
  {
    std::cout << "Cannot open " << argv[3] << std::endl;
    return -1;
  }

  int NSCALES = atoi(argv[1]);
  if (NSCALES <= 0 or NSCALES >= 10)
  {
    std::cout << "NSCALE should be > 0 and < 10, got " << argv[1] << std::endl;
    return -1;
  }

  // Detector threshold (lower it for more points) - 50 for example
  int detector_threshold = atoi(argv[2]);

  obox2d domain(video.get(CV_CAP_PROP_FRAME_HEIGHT), video.get(CV_CAP_PROP_FRAME_WIDTH));
  host_image2d<gl8u> frame_gl(domain);

  host_image2d<i_uchar3> display(domain);

  // Tracker definition
  typedef tracker<tracking_strategies::bc2s_fast_gradient_cpu> T1;
  T1 tr1(domain, NSCALES);

  // Tracker settings
  tr1.strategy().set_detector_frequency(10);
  tr1.strategy().set_filtering_frequency(1);
  for (unsigned s = 0; s < NSCALES; s++)
    tr1.scale(s).strategy().detector().set_n(9).set_fast_threshold(detector_threshold);

  // Record trajectories at each scales.
  std::vector<std::vector<trajectory> > trajectories(NSCALES);

  cv::Mat input_;
  while (video.read(input_)) // For each frame
  {
    host_image2d<i_uchar3> frame(input_);
    frame_gl = get_x(frame); // Basic Gray level conversion.
    int64_t t = get_systemtime_usecs();
    tr1.run(frame_gl);
    std::cout << "tr1.run took " << (get_systemtime_usecs() - t)/1000.0 << "ms" << std::endl;

    for (unsigned s = 0; s < NSCALES; s++)
    {
      // Sync trajectories buffer with particles
      tr1.scale(s).pset().sync_attributes(trajectories[s], trajectory());
      // Update trajectories.
      update_trajectories(trajectories[s], tr1.scale(s).pset());
    }

    // Display trajectories at the finest scale.
    copy(frame, display);
    for (auto& t : trajectories[0])
    {
      if (t.history.size() > 0)
      {
        auto& v = t.history;
        for (int i = std::max(int(v.size() - 10), 0); i < v.size() - 1; i++)
          draw_line2d(display, v[i], v[i+1], i_uchar3(0,0,0));
        draw_c8(display, v.back(), i_uchar3(0,0,255));
      }
    }

    cv::imshow("Trajectories", cv::Mat(display));
    cv::waitKey(10);
  }

  return 0;
}
