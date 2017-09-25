## README - BAFT: Binary Affine Feature Transform

BAFT is a fast binary and quasi affine invariant local image feature. It combines the affine invariance of Harris Affine feature descriptors with the speed of binary descriptors such as BRISK and ORB. BAFT derives its speed and precision from sampling local image patches in a pattern that depends on the second moment matrix of the same image patch. This approach results in a fast but discriminative descriptor, especially for image pairs with large perspective changes.

Version: 0.6
Date: 2017-06-15

You can get the latest version of the code from github:
`https://github.com/arnfred/BAFT`

More details about the feature transform and its performance/benchmarking can be found in the following paper:

J. T. Arnfred, V. D. Nguyen, S. Winkler. BAFT: Binary Affine Feature Transform. In Proc. IEEE International Conference on Image Processing (ICIP), Beijing, China, Sept. 17-20, 2017. ([available for download](http://vintage.winklerbros.net/publications.html))

Please cite the above paper if you use BAFT. 

### How to install and run:
As a prerequisite you will have to install opencv 2.4.10. [Download the sources](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.10/opencv-2.4.10.zip/download), navigate to the directory and follow these steps:
```
mkdir build
cd build
cmake -G "Unix Makefiles" ..
make -j8
sudo make install
```

To compile, clone this git repo and cd to the directory. Then do:
```
mkdir build
cd build
cmake ..
make
```

You should now be able to run `testbaft ../data/graf/img4.ppm`
