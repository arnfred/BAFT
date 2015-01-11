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
#include <fstream>

using namespace std;
using namespace cv;

/* ************************************************************************* */
int main(int argc, char *argv[]) {

  cv::Mat imgN;

  // Define sizes and other variables
  std::vector<int> sizes { 16, 32, 64, 128 };
  //std::vector<int> sizes { 1024 };
  std::vector<string> testsets { "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall", "abs_x1", "abs_x4", "abs_x10" };

  int limit = 3000; // default value
  if (argc > 1)
    limit = (int)atoi(argv[1]);

  for (int j = 0; j < (int)sizes.size(); j++)
  {
    int size = sizes[j];

    for (int k = 0; k < (int)testsets.size(); k++)
    {

      // How many images?
      int nb_files = 6;
      const string testset = testsets[k];
      if (testset == "abs_x1" or testset == "abs_x4" or testset == "abs_x10")
        nb_files = 9;

      // Find file ending
      string fileEnding = ".ppm";
      if (testset == "boat" or testset == "abs_x1" or testset == "abs_x4" or testset == "abs_x10")
        fileEnding = ".pgm";

      for (int i = 0; i < nb_files; i++)
      {

        // Save descriptors from image `i`
        std::ostringstream imgNStream;
        imgNStream << "../data/" << testset << "/img" << (i+1) << fileEnding;
        const string imgNFile = imgNStream.str();
        imgN = imread(imgNFile, 1);

        cout << testset << "_" << (i+1) << ": " << size << " bytes\n";
        // Create daft object
        Ptr<Feature2D> ddaft;
        ddaft = DAFT::create(limit, size);
        // Detect daft features in the images
        vector<cv::KeyPoint> kptsN;
        cv::Mat descN;
        ddaft->detectAndCompute(imgN, cv::noArray(), kptsN, descN);

        // Write out descriptors
        std::ostringstream desc_file_name;
        desc_file_name << "../descriptors/" << testset << "_img" << (i+1) << "_" << size << ".dsc";
        ofstream desc_file;
        desc_file.open(desc_file_name.str());
        std::ostringstream pos_file_name;
        pos_file_name << "../descriptors/" << testset << "_img" << (i+1) << ".pos";
        ofstream pos_file;
        pos_file.open(pos_file_name.str());
        uchar* desc_p = descN.ptr<uchar>();
        for (int m = 0; m < descN.rows; m++)
        {
          for (int n = 0; n < descN.cols; n++)
          {
            desc_file << (int)desc_p[m*descN.cols + n] << " ";
          }
          desc_file << "\n";
          pos_file << kptsN[m].pt.x << " " << kptsN[m].pt.y << "\n";
        }
        desc_file.close();
        pos_file.close();
      }
    }
  }

  return 1;
}
