#ifndef __OPENCV_BAFT_HPP__
#define __OPENCV_BAFT_HPP__

#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

class BAFT : public Feature2D
{
public:

    CV_WRAP static Ptr<BAFT> create(int nfeatures=1000, int size=128,
            int patchSize=30, int gaussianBlurSize = 0, bool fullRotation=0,
            float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=45,
            int fastThreshold=20);

    CV_WRAP virtual void setMaxFeatures(int maxFeatures) = 0;
    CV_WRAP virtual int getMaxFeatures() const = 0;

    CV_WRAP virtual void setSize(int s) = 0;
    CV_WRAP virtual int getSize() const = 0;

    CV_WRAP virtual void setFullRotation(bool fullRotation) = 0;
    CV_WRAP virtual bool getFullRotation() const  = 0;

    CV_WRAP virtual void setGaussianBlurSize(int gaussianBlurSize) = 0;
    CV_WRAP virtual int getGaussianBlurSize() const = 0;

    CV_WRAP virtual void setScaleFactor(double scaleFactor) = 0;
    CV_WRAP virtual double getScaleFactor() const = 0;

    CV_WRAP virtual void setNLevels(int nlevels) = 0;
    CV_WRAP virtual int getNLevels() const = 0;

    CV_WRAP virtual void setEdgeThreshold(int edgeThreshold) = 0;
    CV_WRAP virtual int getEdgeThreshold() const = 0;

    CV_WRAP virtual void setPatchSize(int patchSize) = 0;
    CV_WRAP virtual int getPatchSize() const = 0;

    CV_WRAP virtual void setFastThreshold(int fastThreshold) = 0;
    CV_WRAP virtual int getFastThreshold() const = 0;
};


#endif
