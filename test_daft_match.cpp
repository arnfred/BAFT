/**
 * @file test_daft.cpp
 * @brief Main program for testing OpenCV daft port in an image matching application
 * @date Jun 05, 2014
 * @author Pablo F. Alcantarilla
 */

#include "./src/utils.h"
#include "./src/daft.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// System
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

/* ************************************************************************* */
int main(int argc, char *argv[]) {

  if (argc < 4) {
    cerr << "Error introducing input arguments!" << endl;
    cerr << "The format needs to be: ./test_daft_match testset number descriptor" << endl;
    return -1;
  }

  cv::Mat img1, imgN;
  const string testset = argv[1];
  const string imgNumber = argv[2];
  string fileEnding = ".ppm";
  if (testset == "boat" || testset == "abs_x1" || testset == "abs_x4" || testset == "abs_x10")
    fileEnding = ".pgm";

  std::ostringstream img1Stream, imgNStream, HStream;
  img1Stream << "../data/" << testset << "/img1" << fileEnding;
  imgNStream << "../data/" << testset << "/img" << imgNumber << fileEnding;
  HStream << "../data/" << testset << "/H1to" << imgNumber << "p";
  const string img1File = img1Stream.str();
  const string imgNFile = imgNStream.str();
  const string HFile = HStream.str();
  const string desc_type = argv[3];
  cout << "img1: " << img1File << "\t imgN: " << imgNFile << "\t H: " << HFile << "\n";
  int size = 128;
  float nndr = 0.8;
  int patch = 30;
  float scaleFactor = 1.2;
  int limit = 3000;
  if (argc > 4)
    size = (int)atoi(argv[4]);
  if (argc > 5)
    nndr = (float)atof(argv[5]);
  if (argc > 6)
    patch = (int)atoi(argv[6]);
  if (argc > 7)
    limit = (int)atoi(argv[7]);
  string desc_matcher = "BruteForce-Hamming";
  if (desc_type == "sift")
      string desc_matcher = "BruteForce";

  // Open the input image
  img1 = imread(img1File, 1);
  imgN = imread(imgNFile, 1);
  cv::Mat H1toN = read_homography(HFile);

  // Create daft object
  Ptr<Feature2D> ddaft;
  if (desc_type == "orb")
      ddaft = ORB::create(limit);
  else if (desc_type == "sift")
      ddaft = xfeatures2d::SIFT::create(limit);
  else
      ddaft = DAFT::create(limit, size, patch, scaleFactor);

  // Timing information
  double t1 = 0.0, t2 = 0.0;
  double tdaft = 0.0, tmatch = 0.0;

  // Detect daft features in the images
  vector<cv::KeyPoint> kpts1, kptsN;
  cv::Mat desc1, descN;

  t1 = cv::getTickCount();
  ddaft->detectAndCompute(img1, cv::noArray(), kpts1, desc1);
  ddaft->detectAndCompute(imgN, cv::noArray(), kptsN, descN);
  t2 = cv::getTickCount();
  tdaft = 1000.0*(t2-t1) / cv::getTickFrequency();

  int nr_kpts1 = kpts1.size();
  int nr_kptsN = kptsN.size();

  // Match the descriptors using NNDR matching strategy
  vector<vector<cv::DMatch> > dmatches;
  vector<cv::Point2f> matches, inliers;
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(desc_matcher);

  t1 = cv::getTickCount();
  matcher->knnMatch(desc1, descN, dmatches, 2);
  matches2points_nndr(kpts1, kptsN, dmatches, matches, nndr);
  t2 = cv::getTickCount();
  tmatch = 1000.0*(t2-t1) / cv::getTickFrequency();

  // Compute the inliers using the ground truth homography
  float max_h_error = 2.5;
  compute_inliers_homography(matches, inliers, H1toN, max_h_error);

  // Compute the inliers statistics
  int nr_matches = matches.size()/2;
  int nr_inliers = inliers.size()/2;
  int nr_outliers = nr_matches - nr_inliers;
  float ratio = 100.0*((float) nr_inliers / (float) nr_matches);

  cout << "daft Matching Results" << endl;
  cout << "*******************************" << endl;
  cout << "# Keypoints 1:                        \t" << nr_kpts1 << endl;
  cout << "# Keypoints N:                        \t" << nr_kptsN << endl;
  cout << "# Matches:                            \t" << nr_matches << endl;
  cout << "# Inliers:                            \t" << nr_inliers << endl;
  cout << "# Outliers:                           \t" << nr_outliers << endl;
  cout << "Inliers Ratio (%):                    \t" << ratio << endl;
  cout << "Time Detection+Description (ms):      \t" << tdaft << endl;
  cout << "Time Matching (ms):                   \t" << tmatch << endl;
  cout << endl;

  // Visualization
  cv::Mat img_com = cv::Mat(cv::Size(2*img1.cols, img1.rows), CV_8UC3);
  draw_keypoints(img1, kpts1);
  draw_keypoints(imgN, kptsN);
  draw_inliers(img1, imgN, img_com, inliers);

  cv::namedWindow("daft Matching", cv::WINDOW_KEEPRATIO);
  cv::imshow("daft Matching", img_com);
  cv::waitKey(0);

  return 1;
}
