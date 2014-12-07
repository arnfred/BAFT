/**
 * @file write_descriptors.cpp
 * @brief Write out descriptors for daft
 * @date December 7, 2014
 * @author Jonas Toft Arnfred
 */

#include "./src/utils.h"
#include "./src/daft.h"

// System
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

/* ************************************************************************* */
int main(int argc, char *argv[]) {

  if (argc < 2) {
    cerr << "Error introducing input arguments!" << endl;
    cerr << "The format needs to be: ./write_descriptors testset" << endl;
    return -1;
  }

  cv::Mat imgN;
  const string testset = argv[1];
  string fileEnding = ".ppm";
  if (testset == "boat")
    fileEnding = ".pgm";

  // Define sizes and other variables
  std::vector<int> sizes { 16, 32, 64, 128 };
  int patch = 30;
  float scaleFactor = 1.2;
  string desc_matcher = "BruteForce-Hamming";

  for (int i = 0; i < 6; i++)
  {
    // Save descriptors from image `i`
    std::ostringstream imgNStream;
    imgNStream << "../data/" << testset << "/img" << (i+1) << fileEnding;
    const string imgNFile = imgNStream.str();
    imgN = imread(imgNFile, 1);

    for (int j = 0; j < (int)sizes.size(); j++)
    {
      int size = sizes[j];
      cout << testset << (i+1) << ": " << size << " bytes\n";
      // Create daft object
      Ptr<Feature2D> ddaft;
      ddaft = DAFT::create(3000, size, patch, scaleFactor);
      // Detect daft features in the images
      vector<cv::KeyPoint> kptsN;
      cv::Mat descN;
      ddaft->detectAndCompute(imgN, cv::noArray(), kptsN, descN);

      // Write out descriptors
      std::ostringstream desc_file_name;
      desc_file_name << "../descriptors/" << testset << "_img" << i << "_" << size << ".dsc";
      FileStorage file(desc_file_name.str(), FileStorage::WRITE);
      file << "descriptors" << descN;
    }
  }

  return 1;
}
