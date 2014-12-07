/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/** Authors: Ethan Rublee, Vincent Rabaud, Gary Bradski */
/** Modified to create DAFT by: Jonas Arnfred */

#include "daft.h"

#include <opencv2/core.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iterator>
#include <iostream>

#ifndef CV_IMPL_ADD
#define CV_IMPL_ADD(x)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

const float HARRIS_K = 0.02f;


/**
 * Function that computes the Harris responses in a
 * blockSize x blockSize patch at given points in the image
 */
static void
HarrisResponses(const Mat& img, std::vector<KeyPoint>& pts,
                Mat& response, int blockSize, float harris_k)
{
    CV_Assert( img.type() == CV_8UC1 && blockSize*blockSize <= 2048 );

    size_t ptidx, ptsize = pts.size();

    const uchar* ptr00 = img.ptr<uchar>();
    int step = img.step1();
    int r = blockSize/2;

    float scale = 1.f/((1 << 2) * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;

    AutoBuffer<int> ofsbuf(blockSize*blockSize);
    int* ofs = ofsbuf;
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i*blockSize + j] = (int)(i*step + j);

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int x0 = (int)pts[ptidx].pt.x;
        int y0 = (int)pts[ptidx].pt.y;
        float xd = 1 - (pts[ptidx].pt.x - (float)x0);
        float yd = 1 - (pts[ptidx].pt.y - (float)y0);
        float xd_inv = 1 - xd;
        float yd_inv = 1 - yd;

        const uchar* ptr0 = ptr00 + (y0 - r)*step + x0 - r;
        float a = 0, b = 0, c = 0;

        for( int k = 0; k < blockSize*blockSize; k++ )
        {
            const uchar* ptr = ptr0 + ofs[k];
            float Ix = ((ptr[0]*xd  + ptr[1]  + ptr[2]*xd_inv + ptr[-step+1]*yd + ptr[step+1]*yd_inv) -
                        (ptr[-2]*xd + ptr[-1] + ptr[0]*xd_inv + ptr[-step-1]*yd + ptr[step-1]*yd_inv));
            float Iy = ((ptr[0]*yd + ptr[step] + ptr[2*step]*yd_inv + ptr[step-1]*xd + ptr[step+1]*xd_inv) -
                        (ptr[-2*step]*yd + ptr[-step] + ptr[0]*yd_inv + ptr[-step-1]*xd + ptr[-step+1]*xd_inv));
            a += Ix*Ix;
            b += Iy*Iy;
            c += Ix*Iy;
        }
        response.at<float>(ptidx,0) = a;
        response.at<float>(ptidx,1) = b;
        response.at<float>(ptidx,2) = c;
        pts[ptidx].response = (a * b - c * c -
                               harris_k * (a + b) * (a + b))*scale_sq_sq;
        pts[ptidx].class_id = ptidx;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline float pick( const Mat& img, float x, float y, int step )
{
    int x0 = (int)x;
    int y0 = (int)y;
    float xd = 1 - (x - (float)x0);
    float yd = 1 - (y - (float)y0);
    const uchar* ptr = img.ptr<uchar>() + y0*step + x0;
    return ptr[-1]*xd + ptr[0] + ptr[1]*(1 - xd) + ptr[-step]*yd + ptr[step]*(1 - yd);
}

static void
computeSkew( const Mat& responses, Mat& skew, int nkeypoints )
{
    int i;
    Mat corr(2, 2, CV_32F), eig_vec, eig_val;
    float* corr_p = corr.ptr<float>();
    for (i = 0; i < nkeypoints; i++)
    {
        // Declare pointers to the rows corresponding to keypoint `i`
        const float* resp_row = responses.ptr<float>(i);
        float* skew_row = skew.ptr<float>(i);

        corr_p[0] = resp_row[0];
        corr_p[1] = resp_row[2];
        corr_p[2] = resp_row[2];
        corr_p[3] = resp_row[1];

        // Find eigenvectors
        eigen(corr, eig_val, eig_vec);
        float* val_p = eig_val.ptr<float>();
        const float* vec_p = eig_vec.ptr<float>();

        // Normalize eigen values
        const float val_sq = std::sqrt(val_p[0]*val_p[1]);
        val_p[0] = std::sqrt(val_p[0] / val_sq);
        val_p[1] = std::sqrt(val_p[1] / val_sq);

        // Calculate transformation matrix based on the matrix multiplication
        // of skew `diag(eig_val)` and rotate [-1*vec[1] vec[0]; vec[0] vec[1]]
        skew_row[0] = -1*vec_p[1]*val_p[0];
        skew_row[1] = vec_p[0]*val_p[1];
        skew_row[2] = vec_p[0]*val_p[0];
        skew_row[3] = vec_p[1]*val_p[1];
    }
}

static void generatePoints( Mat& points, int npoints, int patchSize )
{
    RNG rng(0x34985714);
    float u;
    for( int i = 0; i < npoints; i++ )
    {
        u = rng.uniform((float)-4.0, (float)4.0);
        if (u >= 0)
            points.at<float>(i,0) = exp(-1*u);
        else
            points.at<float>(i,0) = -1*exp(u);
        u = rng.uniform((float)-4.0, (float)4.0);
        if (u >= 0)
            points.at<float>(i,1) = exp(-1*u);
        else
            points.at<float>(i,1) = -1*exp(u);
    }
    points *= patchSize;
    //cout << "points:\n" << points << "\n\n";
}

static void
computeDAFTDescriptors( const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
                       const std::vector<float>& layerScale, const Mat& harrisResponse, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, int dsize, int patchSize )
{
    // Compute skew matrix for each keypoint
    int nkeypoints = (int)keypoints.size();
    Mat skew(nkeypoints, 4, CV_32F), points_kp, s, img_roi;
    computeSkew(harrisResponse, skew, nkeypoints);

    // Gather a set of points
    // We need four points for every four bit element, which means eight per byte
    int npoints = dsize*8;
    Mat points(npoints, 2, CV_32F);
    generatePoints(points, npoints, patchSize); // TODO: load fixed set of points instead of generating them

    // Now for each keypoint, collect data for each point and construct the descriptor
    KeyPoint kp;
    for (int i = 0; i < nkeypoints; i++)
    {
        // Find image region of interest in image pyramid
        kp = keypoints[i];
        int x = kp.pt.x / layerScale[kp.octave];
        int y = kp.pt.y / layerScale[kp.octave];
        img_roi = imagePyramid(layerInfo[kp.octave]); // TODO: this can break for unsupported keypoints
        int step = img_roi.step1();
        uchar* desc = descriptors.ptr<uchar>(i);
        const float* p = points.ptr<float>();
        const float* s = skew.ptr<float>(i);

        float min_val = 999, max_val = 0, picked = 0;
        int min_idx = 0, max_idx = 0, byte_val = 0;
        for (int j = 0; j < dsize*8; j++)
        {
            // Matrix multiplication by hand
            float x0 = p[2*j]*s[0] + p[2*j+1]*s[2] + x;
            float y0 = p[2*j]*s[1] + p[2*j+1]*s[3] + y;

            picked = pick(img_roi, x0, y0, step);
            if (picked < min_val)
            {
                min_idx = j % 4;
                min_val = picked;
            }
            else if (picked > max_val)
            {
                max_idx = j % 4; // j & 3??
                max_val = picked;
            }

            if (j % 8 == 0)
            {
                desc[(j >> 3)] = (uchar)((byte_val << 4) + (max_idx + (min_idx << 2)));
            }
            else if (j % 4 == 0)
            {
                byte_val = (max_idx + (min_idx << 2));
                min_val = 999; max_val = 0; // Reset
            }
        }
    }
}

static inline float getScale(int level, double scaleFactor)
{
    return (float)std::pow(scaleFactor, (double)(level));
}


class DAFT_Impl : public DAFT
{
public:
    explicit DAFT_Impl(int _nfeatures, int _size, int _patchSize, float _scaleFactor, int _nlevels, int _edgeThreshold,
             int _fastThreshold) :
        nfeatures(_nfeatures), size(_size), scaleFactor(_scaleFactor), nlevels(_nlevels),
        edgeThreshold(_edgeThreshold), patchSize(_patchSize), fastThreshold(_fastThreshold)
    {}

    void setMaxFeatures(int maxFeatures) { nfeatures = maxFeatures; }
    int getMaxFeatures() const { return nfeatures; }

    void setSize(int size_) { size = size_; }
    int getSize() const { return size; }

    void setScaleFactor(double scaleFactor_) { scaleFactor = scaleFactor_; }
    double getScaleFactor() const { return scaleFactor; }

    void setNLevels(int nlevels_) { nlevels = nlevels_; }
    int getNLevels() const { return nlevels; }

    void setEdgeThreshold(int edgeThreshold_) { edgeThreshold = edgeThreshold_; }
    int getEdgeThreshold() const { return edgeThreshold; }

    void setPatchSize(int patchSize_) { patchSize = patchSize_; }
    int getPatchSize() const { return patchSize; }

    void setFastThreshold(int fastThreshold_) { fastThreshold = fastThreshold_; }
    int getFastThreshold() const { return fastThreshold; }

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;
    // returns the default norm type
    int defaultNorm() const;

    // Compute the DAFT_Impl features and descriptors on an image
    void detectAndCompute( InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false );

protected:

    int nfeatures;
    int size;
    double scaleFactor;
    int nlevels;
    int edgeThreshold;
    int patchSize;
    int fastThreshold;
};

int DAFT_Impl::descriptorSize() const
{
    return size;
}

int DAFT_Impl::descriptorType() const
{
    return CV_8U;
}

int DAFT_Impl::defaultNorm() const
{
    return NORM_HAMMING;
}


/** Compute the amount of features for each level of the image pyramid
 * @param nlevels the number of levels in the image pyramid
 * @param nfeatures the total number of features
 * @param scaleFactor the scale difference between two adjacent levels
 */
static std::vector<int> featuresPerLevel(int nlevels,
                                         int nfeatures,
                                         float scaleFactor)
{
    std::vector<int> nfeaturesPerLevel(nlevels);

    // fill the extractors and descriptors for the corresponding scales
    float factor = (float)(1.0 / scaleFactor);
    float ndesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)std::pow((double)factor, (double)nlevels));

    int sumFeatures = 0, level;
    for( level = 0; level < nlevels-1; level++ )
    {
        nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevel[level];
        ndesiredFeaturesPerScale *= factor;
    }
    nfeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
    return nfeaturesPerLevel;
}


// Compute the offsets and sizes of each image layer in the
// image pyramid. Return the total size of the image pyramid buffer
static Size computeLayerInfo(const Mat& image,
                             int border, double scaleFactor,
                             int nLevels,
                             std::vector<Rect>& layerInfo,
                             std::vector<int>& layerOfs,
                             std::vector<float>& layerScale)
{
    int level_dy = image.rows + border*2, level;
    Point level_ofs(0,0);
    Size bufSize((image.cols + border*2 + 15) & -16, 0);

    for( level = 0; level < nLevels; level++ )
    {
        float scale = getScale(level, scaleFactor);
        layerScale[level] = scale;
        Size sz(cvRound(image.cols/scale), cvRound(image.rows/scale));
        Size wholeSize(sz.width + border*2, sz.height + border*2);
        if( level_ofs.x + wholeSize.width > bufSize.width )
        {
            level_ofs = Point(0, level_ofs.y + level_dy);
            level_dy = wholeSize.height;
        }

        Rect linfo(level_ofs.x + border, level_ofs.y + border, sz.width, sz.height);
        layerInfo[level] = linfo;
        layerOfs[level] = linfo.y*bufSize.width + linfo.x;
        level_ofs.x += wholeSize.width;
    }
    bufSize.height = level_ofs.y + level_dy;
    return bufSize;
}

// Compute the image pyramid by laying the resized images next to each other in
// image pyramid buffer starting from largest image in the top to the smallest
// image in the bottom. If two images fit on a row they are put next to each
// other
static void computeImagePyramid(const Mat& image,
                                int border,
                                int nLevels,
                                std::vector<Rect> layerInfo,
                                std::vector<int> layerOfs,
                                std::vector<float> layerScale,
                                Mat& imagePyramid, Mat& mask,
                                Mat& maskPyramid)
{
    // Initialize values for level 0
    Mat prevImg     = image, prevMask = mask;
    Rect linfo      = layerInfo[0];
    Size sz         = Size(linfo.width, linfo.height);
    Rect wholeLinfo = Rect(linfo.x - border, linfo.y - border, sz.width + border*2, sz.height + border*2);
    Mat extImg      = imagePyramid(wholeLinfo), extMask;

    // Pre-compute level 0 of the image pyramid
    copyMakeBorder(image, extImg, border, border, border, border,
                   BORDER_REFLECT_101);

    // If image mask, copy this into mask buffer
    if( !mask.empty() )
        copyMakeBorder(mask, extMask, border, border, border, border,
        BORDER_CONSTANT+BORDER_ISOLATED);

    // Pre-compute the rest of the layer of the image pyramid
    for (int level = 1; level < nLevels; ++level)
    {
        linfo       = layerInfo[level];
        sz          = Size(linfo.width, linfo.height);
        wholeLinfo  = Rect(linfo.x - border, linfo.y - border,
                           sz.width + border*2, sz.height + border*2);
        extImg      = imagePyramid(wholeLinfo);
        Mat currImg = extImg(Rect(border, border, sz.width, sz.height)), extMask, currMask;

        resize(prevImg, currImg, sz, 0, 0, INTER_LINEAR);
        copyMakeBorder(currImg, extImg, border, border, border, border,
                       BORDER_REFLECT_101+BORDER_ISOLATED);

        // Update mask if present
        if( !mask.empty() )
        {
            extMask     = maskPyramid(wholeLinfo);
            currMask    = extMask(Rect(border, border, sz.width, sz.height));
            resize(prevMask, currMask, sz, 0, 0, INTER_LINEAR);
            threshold(currMask, currMask, 254, 0, THRESH_TOZERO);
            copyMakeBorder(currMask, extMask, border, border, border, border,
                           BORDER_CONSTANT+BORDER_ISOLATED);
        }

        // Update image and mask
        prevImg     = currImg;
        prevMask    = currMask;
    }
}


/** Compute the DAFT_Impl keypoints and their Harris Response on an image
 * @param image_pyramid the image pyramid to compute the features and descriptors on
 * @param mask_pyramid the masks to apply at every level
 * @param layerInfo the bounding rectangles of each image layer
 * @param layerScale the scale of each layer
 * @param keypoints the resulting keypoints, clustered per level
 * @param response an empty vector to be filled with HarrisResponse
 * @nfeatures the number of features
 * @param scaleFactor the ratio of scale between to levels in the image pyramid
 * @edgeThreshold how far away from the edge of the image we look for keypoints
 * @patchSize The approximate size we look for points in
 * @fastThreshold Threshold for quality of fast keypoints
 */
static void computeKeyPoints(const Mat& imagePyramid,
                             const Mat& maskPyramid,
                             const std::vector<Rect>& layerInfo,
                             const std::vector<float>& layerScale,
                             std::vector<KeyPoint>& allKeypoints,
                             Mat& response,
                             int nfeatures, double scaleFactor,
                             int edgeThreshold, int patchSize,
                             int fastThreshold  )
{
    int i, nkeypoints, level, nlevels = (int)layerInfo.size();
    std::vector<int> nfeaturesPerLevel = featuresPerLevel(nlevels, nfeatures, scaleFactor);

    allKeypoints.clear();
    std::vector<KeyPoint> keypoints;
    keypoints.reserve(nfeaturesPerLevel[0]*2);

    Mat cur_response(nfeatures, 3, CV_32F);
    int responseOffset = 0;

    for( level = 0; level < nlevels; level++ )
    {
        int featuresNum = nfeaturesPerLevel[level];
        Mat img = imagePyramid(layerInfo[level]);
        Mat mask = maskPyramid.empty() ? Mat() : maskPyramid(layerInfo[level]);

        // Detect FAST features, 20 is a good threshold
        Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(fastThreshold, true);
        fd->detect(img, keypoints, mask);

        // Remove keypoints very close to the border
        KeyPointsFilter::runByImageBorder(keypoints, img.size(), edgeThreshold);

        // Keep more points than necessary as FAST does not give amazing corners
        KeyPointsFilter::retainBest(keypoints, 2 * featuresNum);

        // Filter remaining points based on their Harris Response
        HarrisResponses(img, keypoints, cur_response, 7, HARRIS_K);
        KeyPointsFilter::retainBest(keypoints, featuresNum);

        nkeypoints = (int)keypoints.size();
        int index;
        for( i = 0; i < nkeypoints; i++ )
        {
            index = keypoints[i].class_id;
            keypoints[i].class_id = 0;
            keypoints[i].octave = level;
            keypoints[i].size = patchSize*layerScale[level];
            keypoints[i].pt *= layerScale[level];
            response.at<float>(i + responseOffset, 0) = cur_response.at<float>(index, 0);
            response.at<float>(i + responseOffset, 1) = cur_response.at<float>(index, 1);
            response.at<float>(i + responseOffset, 2) = cur_response.at<float>(index, 2);
        }
        responseOffset += nkeypoints;

        std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(allKeypoints));
    }
}


/** Compute the DAFT_Impl features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 */
void DAFT_Impl::detectAndCompute( InputArray _image, InputArray _mask,
                                 std::vector<KeyPoint>& keypoints,
                                 OutputArray _descriptors, bool useProvidedKeypoints )
{
    CV_Assert(patchSize >= 2);

    bool do_keypoints = !useProvidedKeypoints;
    bool do_descriptors = _descriptors.needed();

    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;

    int border = std::max(edgeThreshold, patchSize);

    Mat image = _image.getMat(), mask = _mask.getMat();
    if( image.type() != CV_8UC1 ) {
        cvtColor(_image, image, COLOR_BGR2GRAY);
    }

    int i, level, nLevels = this->nlevels, nkeypoints = (int)keypoints.size();

    // TODO: The way this should work: Find min size. Find corresponding size to each
    // level, assuming scaleFactor that we have currently and assign correct levels.
    // Note the highest level assigned and use that as nLevels
    if( !do_keypoints )
    {
        nLevels = 0;
        for( i = 0; i < nkeypoints; i++ )
        {
            level = keypoints[i].octave;
            CV_Assert(level >= 0);
            nLevels = std::max(nLevels, level);
        }
        nLevels++;
    }

    // Compute the different aspects of the layers in the image pyramid
    std::vector<Rect> layerInfo(nLevels);
    std::vector<int> layerOfs(nLevels);
    std::vector<float> layerScale(nLevels);
    Mat imagePyramid, maskPyramid;
    Mat harrisResponse(nfeatures, 3, CV_32F);
    Size bufSize = computeLayerInfo(image, border, scaleFactor, nLevels,
                                     layerInfo, layerOfs, layerScale);
    // create image pyramid
    imagePyramid.create(bufSize, CV_8U);
    if( !mask.empty() )
        maskPyramid.create(bufSize, CV_8U);
    computeImagePyramid(image, border, nLevels, layerInfo,
                        layerOfs, layerScale, imagePyramid, mask, maskPyramid);

    // Get keypoints, those will hopefully be far enough from the border that no check will be required for the descriptor
    if( do_keypoints )
        computeKeyPoints(imagePyramid, maskPyramid,
                         layerInfo, layerScale, keypoints, harrisResponse,
                         nfeatures, scaleFactor, edgeThreshold, patchSize, fastThreshold);
    else
    {
        KeyPointsFilter::runByImageBorder(keypoints, image.size(), edgeThreshold);
        HarrisResponses(image, keypoints, harrisResponse, 7, HARRIS_K);
    }

    if( do_descriptors )
    {
        int dsize = descriptorSize();
        nkeypoints = (int)keypoints.size();
        _descriptors.create(nkeypoints, dsize, CV_8U);
        Mat descriptors = _descriptors.getMat();
        computeDAFTDescriptors(imagePyramid, layerInfo, layerScale, harrisResponse,
                               keypoints, descriptors, dsize, patchSize);
    }
}

Ptr<DAFT> DAFT::create(int nfeatures, int size, int patchSize, float scaleFactor, int nlevels, int edgeThreshold,
         int fastThreshold)
{
    return makePtr<DAFT_Impl>(nfeatures, size, patchSize, scaleFactor, nlevels, edgeThreshold,
                              fastThreshold);
}

