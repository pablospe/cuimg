#include <iostream>
#include <vector>
#include <sys/time.h>

#include <cuimg/profiler.h>
#include <cuimg/video_capture.h>
#include <cuimg/improved_builtin.h>
#include <cuimg/target.h>

#include <cuimg/copy.h>
#include <cuimg/dsl/all.h>
// #include <cuimg/gpu/device_image2d.h>
#include <cuimg/cpu/host_image2d.h>

#include <cuimg/tracking2/tracker.h>


using namespace cuimg;
using namespace cv;
using namespace std;

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

  void move(trajectory&& t)
  {
    history.swap(t.history);
    alive = t.alive;
  }

  deque<Point> history;
  bool alive;
};


// Update a trajectory when a particle moves.
template <typename TR>
void update_trajectories(vector<trajectory>& v, TR& pset)
{
  const auto& parts = pset.dense_particles();
  for(unsigned i = 0; i < v.size(); i++)
    v[i].alive = false;
  for(unsigned i = 0; i < v.size(); i++)
    if (parts[i].age > 0)
    {
      assert(parts[i].age != 1 || v[i].history.empty());
      v[i].history.push_back( Point(parts[i].pos.c(),parts[i].pos.r()) );
      v[i].alive = true;
      if (v[i].history.size() > 20) v[i].history.pop_front();
    }
    else
    {
      v[i].history.clear();
    }
}

void draw_trajectory(trajectory &current, Mat &img)
{
	if (current.alive)
	{
		int n = current.history.size();
		int i=0;
		for(auto it=current.history.begin(); it!=current.history.end(); ++it)
		{
// 			img.at<Vec3b>(*it) = Vec3b(255*i/n, 124*i/n + 125, 0);
// 			circle(img, *it, 0, Scalar(255*i/n, 124*i/n + 125, 0), 2);
			line(img, *it, *it, Scalar(255*i/n, 124*i/n + 125, 0), 2);
			i++;
		}
	}


// 	// showing only the last one
// 	if (current.alive)
// 	{
// 		Point p = current.history.back();
// // 		Point p(i_p.c(), i_p.r());
// 		line(img, p, p, Scalar(0,255,0), 3);
// 	}

}

void draw_trajectories(vector<trajectory> &trajectories, Mat &img)
{
    for (unsigned i = 0; i < trajectories.size(); i++)
	{
		draw_trajectory(trajectories[i], img);
	}
}


int main(int argc, char* argv[])
{
  VideoCapture video;

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
    cout << "Usage: ./tracking_sample nscales detector_threshold [video_file]" << endl;
    return -1;
  }

  if(!video.isOpened())
  {
    cout << "Cannot open " << argv[3] << endl;
    return -1;
  }

  int NSCALES = atoi(argv[1]);
  if (NSCALES <= 0 or NSCALES >= 10)
  {
    cout << "NSCALE should be > 0 and < 10, got " << argv[1] << endl;
    return -1;
  }

  // Detector threshold (lower it for more points) - 50 for example
  int detector_threshold = atoi(argv[2]);

  obox2d domain(video.get(CV_CAP_PROP_FRAME_HEIGHT), video.get(CV_CAP_PROP_FRAME_WIDTH));
  host_image2d<gl8u> frame_gl(domain);

  // Tracker definition
  typedef tracker<tracking_strategies::bc2s_fast_gradient_cpu> T1;
//   typedef tracker<tracking_strategies::bc2s_mdfl2s_gradient_cpu> T1;
  T1 tr1(domain, NSCALES);

  // Tracker settings
  tr1.strategy().set_detector_frequency(1);
  tr1.strategy().set_filtering_frequency(1);
  for (unsigned s = 0; s < NSCALES; s++)
    tr1.scale(s).strategy().detector().set_n(9).set_fast_threshold(detector_threshold);
//     tr1.scale(s).strategy().detector().set_dev_threshold(detector_threshold); // bc2s_mdfl2s_gradient_cpu

  // Record trajectories at each scales.
  vector<vector<trajectory> > trajectories(NSCALES);

  namedWindow( "input", CV_WINDOW_AUTOSIZE );// Create a window for display.
  Mat input_;
  double total = 0;
  unsigned n=0;
  while( video.read(input_) )
  {
//     printf("processing...\n");
    host_image2d<i_uchar3> frame(input_);
    frame_gl = get_x(frame); // Basic Gray level conversion.
    tr1.run(frame_gl);

    double tic = (double) getTickCount();

    for (unsigned s = 0; s < NSCALES; s++)
    {
      // Sync trajectories buffer with particles
      tr1.scale(s).pset().sync_attributes(trajectories[s], trajectory());
      // Update trajectories.
      update_trajectories(trajectories[s], tr1.scale(s).pset());
    }

    double time_update_trajectories = ((double) getTickCount() - tic)*1000./getTickFrequency();


    int64_t t = get_systemtime_usecs();
    tic = (double) getTickCount();
    int s = 0;
    draw_trajectories(trajectories[s], input_);
    double time_draw_trajectories = ((double) getTickCount() - tic)*1000./getTickFrequency();  // time with OpenCV
    cout << "update_trajectories (milisec): "   << time_update_trajectories;
    cout << " -- draw_trajectories (milisec): " << time_draw_trajectories << endl;
    std::cout << "tr1.run took " << (get_systemtime_usecs() - t)/1000.0 << "ms" << std::endl;  // time with gettimeofday()

//     total += time_update_trajectories + time_draw_trajectories;
//     total += time_draw_trajectories;
    total += time_update_trajectories;

    imshow("input", input_);
    waitKey(1);

    n++;
  }

  cout << "Avegare time: " << total/n << " milisec.\n";

  return 0;
}
