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

const float HARRIS_K = 0.04f;

static float points[256*4*2] =
{
    1.29, 31.15, 26.07, -0.92, -18.74, 1.68, 21.54, 22.66,
    1.63, -37.46, -10.22, -45.97, 2.03, -40.41, 22.60, -0.92,
    -4.84, 1.63, 3.17, 47.81, -30.05, -36.72, 1.37, -49.93,
    39.36, 3.77, 1.16, -9.21, -7.09, -5.01, 30.87, -0.98,
    2.80, 9.97, -2.42, -9.60, 30.24, -38.50, 1.00, -2.18,
    -32.93, 2.12, -2.50, -19.57, 1.88, 2.25, -21.66, 8.79,
    4.60, 1.51, -3.03, 1.07, -31.60, -17.92, 1.66, 5.22,
    -8.33, 5.30, 22.46, 15.60, -1.26, -0.95, -1.02, 16.46,
    1.23, -6.28, 9.15, -33.71, 2.31, 8.68, -24.47, -3.20,
    7.63, -3.32, -3.17, 3.80, 17.93, 9.50, 2.03, 6.96,
    9.66, -25.47, -10.17, 1.93, -5.28, -0.97, 1.46, -7.62,
    2.49, 27.63, 17.86, -16.88, -33.43, -1.75, 1.16, 39.67,
    4.78, 4.90, 2.36, 5.53, -0.93, 18.29, -20.58, 26.08,
    33.19, -11.21, -20.88, 2.09, -16.53, -2.09, -5.49, -3.51,
    -1.12, -4.04, -1.51, -1.62, 3.01, -8.26, 6.06, 1.24,
    -23.98, 1.37, 1.97, -1.30, 1.14, -20.94, -6.74, -10.44,
    -25.73, -13.62, -7.60, 1.42, -1.42, 48.21, 7.31, 35.85,
    -1.81, -23.76, 48.29, 11.39, -8.95, 9.19, 14.72, 1.40,
    16.23, 30.53, -13.76, -1.32, 4.61, -1.67, 19.57, 1.01,
    -5.72, -39.03, -3.16, -5.20, -38.39, -4.02, 1.86, -2.68,
    -9.46, 2.35, -10.93, 8.53, 3.35, 2.21, -1.98, -5.77,
    1.74, -1.17, -1.99, 0.95, 8.84, -10.98, 3.96, -1.39,
    -21.15, -8.08, 20.34, -33.37, 1.26, -2.76, 5.27, 21.89,
    1.22, -0.93, 18.36, -7.09, -12.61, -1.01, 4.82, 13.53,
    -5.64, -2.83, -2.08, -2.33, 29.11, 13.71, -18.50, -1.42,
    1.20, 0.92, 2.87, 1.16, -4.41, -3.85, -8.34, 1.40,
    22.04, 0.97, 1.70, 1.06, 25.80, -8.65, -18.54, 3.15,
    38.52, -1.14, 31.79, 8.66, -6.55, -1.09, 19.75, 2.66,
    46.11, -24.71, -22.21, 1.72, -9.79, 7.22, -2.92, -1.76,
    8.84, 4.86, -15.50, 21.08, -1.53, -1.12, -10.97, 3.16,
    -5.26, 0.97, 19.12, 4.95, -7.30, 0.95, 2.96, -6.56,
    23.35, 6.60, -5.36, 6.33, 1.87, 10.04, -16.43, 5.01,
    -2.91, 14.28, -37.59, 7.09, 38.52, 8.95, 1.77, -13.24,
    -1.78, 27.75, -9.67, -2.71, 9.44, 3.33, -6.92, -10.37,
    -7.02, -5.39, -12.58, -9.97, -1.89, 19.86, -3.96, -1.13,
    -2.36, 1.35, 2.41, 15.86, 5.93, -1.38, 10.43, 12.24,
    7.23, -9.62, 5.97, -4.64, 4.10, -37.89, 15.87, -32.22,
    1.98, -7.01, -0.98, -20.08, 12.71, 36.29, -9.92, 1.41,
    -11.22, 30.66, -10.28, -43.66, -1.52, 3.12, 26.76, 20.35,
    12.05, 2.10, -3.40, 1.33, -4.17, -7.77, 34.06, -19.01,
    7.96, 41.90, 13.22, -49.17, 37.13, -32.10, 1.62, 1.14,
    -7.59, -7.36, -15.33, -1.85, -1.67, -1.35, 3.53, -14.63,
    -6.20, -48.15, -1.46, 30.92, 4.37, -1.68, 26.76, -1.21,
    -20.14, 15.14, -21.64, 2.31, -2.12, -5.27, 34.95, -12.47,
    -26.85, -36.69, 38.08, 10.11, -3.17, 20.79, -8.72, 26.83,
    -6.13, -43.19, -2.58, -4.50, -7.84, 9.12, -1.16, -28.37,
    -1.52, -5.50, 1.72, 18.72, 0.93, 19.15, -34.42, 15.53,
    -1.45, -5.20, 6.01, -37.53, -3.67, 33.26, -3.02, 30.42,
    1.28, -4.46, -12.35, 1.86, 3.19, 17.45, -12.10, 3.94,
    5.87, 1.99, 1.46, -5.95, 5.90, -1.98, 21.28, -7.15,
    11.15, 25.46, 5.45, -24.92, 36.34, 8.94, 3.05, 1.03,
    9.26, 17.15, -1.94, 3.90, -45.60, -14.86, -4.91, -3.67,
    11.01, -10.69, 13.55, -46.65, -11.93, -5.29, -3.69, 22.23,
    9.43, 31.18, 9.17, -7.11, -21.87, -1.02, 14.44, 2.16,
    8.89, 1.12, -8.52, 7.59, -8.69, 0.99, 2.99, 31.53,
    4.46, 11.67, 10.22, 9.14, 2.03, -41.27, 27.39, -32.13,
    -3.25, -49.47, 1.35, 26.18, -13.88, 7.76, -3.38, -30.12,
    2.49, 5.03, -38.10, -2.42, -19.50, -1.73, 2.71, -19.07,
    4.04, -3.95, -10.88, -1.97, 1.49, -17.70, -2.29, 26.10,
    -12.92, -48.57, -4.82, -16.30, 7.84, -7.23, -1.50, 36.76,
    -1.88, 7.83, -10.47, 22.92, -5.77, 19.90, -4.24, 39.23,
    -1.08, 16.35, 2.84, 3.75, -4.77, -6.97, -4.45, 45.81,
    1.19, 1.52, -9.77, 1.92, -29.88, -30.25, -3.63, -1.04,
    -3.00, 1.97, 16.43, -31.40, -19.11, 49.62, 6.70, -24.68,
    29.96, -40.14, -8.31, -21.87, -1.14, 7.70, -34.46, 5.74,
    -20.96, 4.87, 26.04, -2.54, 21.39, 4.33, -31.37, 1.17,
    1.28, 13.46, -26.86, 5.20, 5.91, 1.41, 31.96, -1.99,
    -5.02, 19.52, -21.65, 1.04, -40.09, 18.14, 4.33, -17.49,
    -12.14, 21.11, 14.86, 3.66, -4.90, -1.09, -2.34, -1.35,
    3.49, -6.54, 7.64, 5.00, 1.13, -1.07, 38.03, -9.81,
    -6.96, 1.01, 43.25, -22.28, -1.42, 29.81, -1.63, -6.67,
    0.97, 18.53, -1.68, -2.18, 14.68, -3.98, 2.58, -7.20,
    -25.03, -22.59, 13.95, -17.51, 2.79, -8.95, 1.24, 29.28,
    7.60, -2.23, 1.16, -35.20, 18.41, 25.55, 2.63, -12.47,
    -33.05, 1.19, -5.42, 8.29, 3.86, -24.36, -5.65, 12.29,
    1.48, 9.37, -10.51, -4.13, 5.06, -2.12, -38.53, 1.47,
    -31.89, 11.79, 2.11, -39.13, 2.30, 48.21, -4.24, 33.43,
    -14.04, 9.14, 14.15, -1.73, -4.87, 3.64, 2.52, -47.99,
    20.53, 13.52, -3.38, 7.46, -43.61, 16.35, -3.62, 1.36,
    -19.97, 5.66, 1.10, 7.58, -7.04, 35.94, -2.24, 9.73,
    1.89, -26.91, 15.56, 10.45, -2.84, 3.86, 19.44, 4.61,
    4.58, -1.87, -2.22, -1.29, -6.17, 37.21, -1.97, -13.25,
    -2.39, 2.21, 2.36, 3.81, -3.42, 22.42, 17.36, -8.67,
    -1.02, 17.09, -31.92, -28.05, -1.41, 2.43, -26.29, 1.34,
    2.14, -9.63, -1.30, -1.87, -9.12, -17.51, -4.04, 2.64,
    -1.06, 21.74, 29.72, -45.16, -17.48, -2.07, -3.16, -1.00,
    10.63, 18.68, 5.37, 6.20, -1.08, 2.24, -10.33, -11.89,
    14.71, 30.14, 3.82, 33.24, 20.75, -3.60, -19.44, -1.14,
    16.52, -38.39, 29.22, -21.53, -34.56, -2.64, 43.06, 7.73,
    -9.50, 1.46, 3.95, -4.75, -24.54, 1.32, 0.93, 15.32,
    -1.86, 2.88, -2.26, -23.90, 25.23, 2.26, 19.59, -15.67,
    6.78, 22.26, -5.62, 26.91, -14.88, 8.27, -46.19, 11.28,
    45.12, -12.25, -17.21, -1.25, 34.27, -10.54, 19.12, -6.71,
    1.54, -21.54, 7.89, 3.57, -29.46, -2.80, 28.87, -1.44,
    22.62, -36.15, 3.57, -43.84, -13.59, 28.86, 1.56, 4.78,
    1.45, 40.13, -4.53, 1.17, -18.52, 17.59, -3.15, -27.29,
    1.55, 26.88, -2.24, 16.45, -30.58, 1.01, -6.16, -2.87,
    -2.25, -1.37, 28.60, 1.51, -2.26, -27.71, 2.61, -17.85,
    1.66, 21.81, -3.91, 1.28, 15.84, -12.23, -47.40, -46.91,
    1.90, -9.53, -3.76, 14.83, 11.95, -1.01, 7.41, 3.14,
    -47.24, -1.81, -5.81, -0.96, -1.53, 9.24, 34.73, -0.95,
    1.43, 7.11, -36.14, -1.01, -11.09, -33.21, 44.82, -0.97,
    -1.72, 18.24, 1.90, -40.47, 9.47, -1.00, -18.59, 6.93,
    -5.61, -1.03, 9.47, 18.58, 1.58, -1.57, 17.38, 24.29,
    -1.71, 23.92, 2.38, 31.04, -9.69, -2.65, -3.60, 1.00,
    -1.40, 44.10, -17.67, 12.90, -5.41, -6.19, -1.76, -24.79,
    -4.84, 1.35, -3.04, -20.61, -3.38, 30.18, 15.68, -2.88,
    16.88, -36.02, -10.05, -22.82, -36.25, 18.90, 7.42, -14.56,
    -35.02, 14.45, 2.88, -5.28, 3.80, -3.26, 47.51, 3.86,
    -8.13, -13.07, 25.39, 14.58, -1.02, -1.35, -14.28, 1.85,
    2.73, -10.20, -1.33, 10.90, -6.18, -11.05, 1.70, 44.12,
    4.03, 1.11, -3.21, -5.40, -1.21, 6.94, 12.74, -7.92,
    8.99, -39.41, -0.98, 5.44, -8.15, -30.14, 4.51, 5.57,
    -1.54, 8.27, -5.48, 1.44, -4.00, 23.64, 1.00, -1.20,
    25.20, 25.49, 1.84, -10.22, 7.81, 1.00, -4.76, 3.26,
    2.00, 40.03, -14.73, -1.89, 1.41, 1.25, 5.37, -27.11,
    10.37, 0.94, -0.98, -2.28, 8.56, 12.52, -10.18, -16.26,
    -30.07, -5.91, 1.06, 1.68, -21.33, -6.18, 1.64, 8.60,
    1.05, 3.17, 9.10, 2.38, -27.24, 1.91, -2.60, 45.30,
    3.25, -8.01, 17.08, 1.89, -1.78, -4.64, -44.20, 49.81,
    -1.68, 12.04, 1.65, 33.19, 11.29, 5.66, -1.67, -2.05,
    -21.84, 1.59, -3.27, -19.11, 1.18, 4.27, 2.18, 7.22,
    3.23, -4.42, 3.63, 1.66, -3.53, 5.45, -16.83, -2.36,
    3.77, 6.22, -2.80, -7.98, -1.46, -26.73, 9.65, 2.35,
    -9.19, -11.83, -2.04, -25.90, -9.82, -1.71, -7.01, 1.14,
    9.13, 39.44, -14.77, -23.06, -39.55, 41.47, -35.37, 20.83,
    -10.05, -2.44, 7.92, 22.43, -2.70, 47.71, 14.59, -17.47,
    -23.40, -26.93, -35.43, 2.08, 2.76, 2.38, -1.00, -27.03,
    22.18, -7.04, -2.47, -7.90, 0.95, -32.30, 1.17, 17.50,
    -20.41, -21.57, 1.67, 3.86, -24.68, -7.29, -23.53, -39.42,
    -0.93, 18.40, -5.45, -26.41, -44.13, 2.00, 13.95, 9.32,
    6.90, 1.07, -4.62, -13.90, 2.51, -1.61, -1.48, -2.91,
    -2.28, 48.11, 12.52, -6.68, 6.60, -5.05, -33.44, -8.02,
    -1.32, 11.84, 13.85, -10.99, 1.19, 32.44, -35.12, 32.66,
    4.54, 3.08, -3.45, 34.40, 31.49, 1.71, 1.80, -43.32,
    5.01, 24.70, -3.00, -12.14, -2.32, 42.97, 17.37, 4.63,
    7.38, 43.92, 1.41, 3.42, 1.07, -14.54, 8.78, 12.92,
    -32.04, -0.99, -2.43, -4.72, 1.88, 4.37, -21.82, 4.69,
    -5.68, -3.96, -0.98, -7.98, 20.95, -4.13, -3.95, -0.99,
    5.36, 6.94, -1.05, 43.39, -21.22, 2.16, 2.71, 0.99,
    2.66, 20.50, 15.58, -15.82, 1.93, 43.51, 4.33, 19.92,
    1.33, -4.95, 8.11, -45.08, 10.81, -2.29, 10.98, -1.34,
    1.17, 11.87, -25.71, 3.04, 1.06, -1.30, -6.52, -1.17,
    19.06, 22.53, 37.63, 8.98, -1.92, 7.32, -10.89, -3.75,
    1.83, 36.59, 0.97, 24.85, 18.67, -32.28, 3.06, -1.78,
    -8.87, -12.14, -0.92, -8.09, 9.54, -15.50, -2.48, 13.29,
    1.42, 28.57, -4.48, -22.55, -1.06, 2.78, -27.01, -32.94,
    16.85, -19.54, -5.77, 11.53, -17.27, -12.29, 7.25, -16.49,
    -34.14, -9.27, 36.92, 13.86, 2.45, -2.94, -3.02, 0.97,
    -30.24, -3.93, -5.79, 5.34, 26.16, 3.05, -7.60, 6.63,
    8.03, 44.26, -0.96, -11.63, -6.97, 2.75, 4.46, 18.86,
    9.63, -28.24, 1.30, 10.03, -20.34, 3.65, 34.56, 17.74,
    6.07, 45.21, -2.91, 2.61, 1.38, 10.29, -29.64, -2.83,
    1.47, -19.60, 1.06, -20.00, 3.07, -5.75, 6.50, -2.07,
    14.71, -2.31, -6.91, -11.35, -6.69, -7.11, -12.19, -7.55,
    34.06, -4.68, 45.63, -1.25, -1.46, -4.24, -33.64, -1.08,
    -4.14, 16.30, 9.29, 1.77, -8.59, 2.63, -2.40, -18.90,
    -15.66, -19.93, 0.97, -25.38, 11.81, 3.40, 6.04, -1.03,
    -6.34, 20.73, -38.05, 5.11, 1.59, -1.22, 6.17, 10.82,
    18.09, 48.77, -14.13, 20.97, -14.70, -12.76, -4.86, 21.51,
    29.25, -3.70, -13.44, 19.44, -5.20, 10.09, 24.44, -24.71,
    2.33, 2.78, 40.21, -34.04, -7.73, -7.76, 4.39, 2.38,
    11.16, -43.63, 5.46, -5.22, -7.33, -4.80, -1.25, -7.83,
    4.24, -19.20, -1.51, 5.43, -4.42, 2.69, -3.13, 14.20,
    8.61, -10.32, 10.32, -10.72, -27.76, 3.18, -7.85, 19.07,
    1.08, 12.07, -3.94, 2.14, -31.07, 4.65, -1.27, 2.23,
    5.53, -21.91, -4.45, 14.47, -12.24, 2.19, -9.89, -30.58,
    -41.97, 13.64, -39.60, 37.61, 1.45, 17.04, -14.25, -1.36,
    -1.17, -21.82, 4.59, 1.79, 2.51, 3.01, 12.92, -16.69,
    -4.55, -39.52, -7.35, 2.91, 16.04, 1.88, -49.56, -6.63,
    -17.07, 5.58, -5.34, 37.08, 1.89, 1.00, 3.08, 6.77,
    -23.22, 1.25, 13.15, -1.50, 1.55, 14.12, 42.30, 18.58,
    31.11, -3.26, 3.17, -3.65, -45.97, -6.70, 28.95, 3.99,
    -22.60, 7.11, -1.43, -1.71, -0.99, 37.54, 13.31, 7.14,
    -28.62, -9.04, -4.53, 14.27, -26.80, 0.93, -3.19, 23.01,
    -14.23, -1.98, -19.35, 22.73, 1.54, -1.13, -16.98, 18.20,
    27.32, -21.97, -7.77, 1.07, -37.99, 1.75, 2.27, -26.50,
    14.04, -12.60, -3.46, -2.04, 1.31, -5.12, -1.26, 21.52,
    9.27, -5.92, 6.11, 15.33, -16.44, -1.60, -13.02, -2.63,
    -13.12, -47.01, 29.61, -25.97, 21.41, 3.11, 5.69, -4.45,
    -2.46, 26.36, -5.04, -2.55, 3.75, 1.94, -0.98, -11.28,
    1.27, 11.79, -3.39, -5.17, 6.04, -2.69, 1.68, 3.74,
    -6.81, -1.07, -1.71, -2.96, -12.50, -17.52, 41.19, -2.01,
    -23.41, -6.84, 44.72, -29.62, -13.90, 2.08, 8.41, -5.10,
    -5.71, 2.48, -46.09, 4.48, 2.47, 1.74, 1.26, -34.91,
    -6.12, -1.14, 20.87, 9.58, -2.74, -46.82, 3.65, -1.75,
    2.32, -48.82, -15.69, 22.59, -1.67, -7.04, -32.62, -0.94,
    -11.10, 1.54, 2.23, -41.08, 13.74, -3.66, -11.85, -1.01,
    16.35, -1.38, -5.04, -1.76, -3.68, 31.20, -1.26, 1.00,
    8.50, -5.57, 28.74, -1.82, -4.12, 30.34, 2.54, 14.77,
    12.32, 1.99, 2.90, 1.71, -22.47, 5.16, 1.16, -23.29,
    28.54, 28.17, -3.22, -4.13, 4.33, 7.69, -8.07, 5.20,
    12.06, 3.45, 13.72, -4.34, -23.70, -38.40, -10.77, -1.50,
    -13.97, 13.85, -22.38, -39.12, 4.59, 3.73, 36.52, -3.16,
    -35.34, 34.89, -1.76, -1.34, -1.27, 14.33, -4.82, -14.12,
    -19.50, -8.81, 27.08, -31.43, -30.97, -19.69, -16.51, 20.54,
    -2.06, -3.06, 1.13, 18.44, -16.08, 1.63, 38.27, -11.57,
    23.93, -39.34, -3.85, 10.99, 1.28, 5.86, 36.20, 41.56,
    -1.15, 10.82, 1.26, -1.56, -8.30, 4.38, -7.18, -29.85,
    1.21, -19.15, -19.94, 1.90, 5.16, -12.78, -47.90, 7.62,
    -30.25, -14.92, 16.04, 33.21, 8.71, -3.01, 14.10, 1.21,
    -1.07, -19.06, -2.29, 4.60, -1.24, -3.17, 3.75, 18.31,
    -3.91, -3.22, -3.56, -21.83, -3.14, -1.18, 8.36, 3.50,
    -14.16, 18.12, 5.75, -2.17, -3.03, 1.57, 1.41, -1.76,
    3.81, -45.45, 3.98, -29.11, 5.82, -2.81, 2.57, 25.73,
    -22.84, -18.26, -8.50, 2.80, -5.61, -18.95, 4.18, -0.92,
    18.57, -23.16, 6.70, 1.90, 2.20, 3.23, -1.38, -1.07,
    44.07, -23.36, 2.34, 43.04, -7.09, -1.94, 1.47, -1.61,
    7.05, 1.74, 1.25, -4.64, -22.93, 3.85, -8.68, 21.60,
    -17.39, -15.62, 48.48, -5.47, 27.68, -2.13, -4.63, -1.14,
    12.81, 13.93, -39.35, 19.57, 27.37, -32.45, 14.13, 1.19,
    -22.40, -37.91, -1.05, -21.10, 3.05, -15.00, 1.70, 5.51,
    1.10, 45.09, -9.52, 2.82, 1.34, 2.61, 7.54, 2.69,
    -1.46, 10.34, 1.55, 9.70, -2.82, 1.81, 21.66, -3.75,
    42.99, -1.46, 4.07, -3.68, 2.76, 39.85, 19.74, 14.27,
    -42.90, -46.84, 5.53, 2.93, 1.59, -0.96, -7.21, 13.34,
    5.04, 1.60, 1.08, 3.51, -19.20, 4.34, -8.94, -2.88,
    -10.83, -1.79, 1.75, -1.54, 37.78, -4.20, -12.83, -22.01,
    17.06, 2.98, -8.89, -28.54, -20.45, -25.14, 6.18, 10.65,
    20.77, 2.45, -40.54, 2.00, 3.83, -24.28, 40.09, -3.40,
    -3.63, -8.68, -3.90, -1.23, -1.14, -7.89, -7.53, 22.90,
    -3.52, -2.06, 9.70, 26.77, 37.25, -6.06, -19.22, 12.12,
    -29.23, 4.05, 13.39, 13.52, -1.09, -1.87, -11.09, -2.53,
    1.38, -0.96, 4.44, 5.22, 12.65, -10.04, 3.47, 35.58,
    -33.35, -4.73, 1.70, -1.19, -2.48, -2.32, -2.78, 6.59,
    -36.94, 3.23, 12.48, 1.06, -14.84, 1.49, 48.69, 1.46,
    -7.34, -2.94, -4.95, 3.07, -1.41, 41.99, -1.17, -6.62,
    -1.38, 1.38, 7.35, 1.14, 1.74, 2.40, 24.72, -1.58,
    -38.02, -33.85, 1.48, -42.80, -32.40, 29.39, 16.04, -28.30,
    -9.87, 7.77, 1.34, 1.08, 34.54, 3.91, 1.93, 5.78,
    19.27, -4.18, -4.04, -2.46, 20.09, -5.40, 3.41, 13.37,
    40.89, 46.25, -2.33, 3.65, -2.34, 2.06, 7.71, -2.94,
    1.57, -11.26, -6.49, 1.09, -43.05, 4.98, 17.01, 1.21,
    -1.91, 1.46, 18.49, 5.11, 2.37, -37.82, -1.71, 4.60,
    6.73, -12.74, 19.64, 10.51, -17.49, 17.33, -2.86, 16.31,
    3.99, 1.81, -16.56, -7.60, -20.22, 14.22, 29.94, -15.42,
    40.72, -1.12, -6.41, -10.12, 27.86, -23.69, -6.95, -4.20,
    21.71, -1.49, 29.25, -10.39, -1.39, 4.08, 44.89, 1.01,
    17.93, -45.04, -30.97, 3.06, 1.26, -3.98, -18.47, 4.06,
    -16.16, 14.36, 1.15, -13.59, -1.81, 3.73, 28.20, 5.30,
    14.32, -20.23, -2.79, -19.69, -3.33, -1.66, -11.53, -32.62,
    12.62, -41.30, -5.08, -1.76, 7.85, 23.12, -1.39, 2.30,
    -25.46, -3.97, 1.40, 42.06, 7.09, -33.57, -15.34, 0.93,
    8.28, -8.18, -2.55, -3.74, -20.19, 2.76, 19.60, 1.50,
    -2.85, 2.10, 1.02, 2.83, -9.58, -8.47, 4.82, -15.97,
    6.12, -28.33, -3.87, -28.32, 39.62, 8.48, 1.88, 8.62,
    -6.23, 42.82, -3.41, 1.92, -0.98, 12.10, -2.38, 10.26,
    -17.36, 4.82, 2.23, -4.48, 7.05, -3.48, 1.55, -8.72,
    -16.82, -5.33, -16.66, -6.07, -5.32, -47.83, -1.53, -3.64,
    -7.19, 13.56, 12.17, -2.39, 2.83, -1.65, 29.21, 1.95,
    4.08, 40.28, -45.96, -2.44, -45.24, 2.15, -38.20, -16.31,
    -2.06, -31.68, -2.36, 43.35, -5.20, -1.33, -1.33, -4.33,
    1.05, -3.75, 12.79, -41.65, 14.44, 8.98, 6.44, 12.76,
    -8.87, -1.81, -5.71, -6.11, 8.42, -44.53, -1.35, -11.88,
    10.11, 9.47, 21.24, 29.46, -22.33, -10.90, 1.81, 2.03,
    41.33, 1.59, -32.53, 15.81, 16.21, -15.46, 1.88, 2.96
};

/**
 * Function that computes the Harris responses in a
 * blockSize x blockSize patch at given points in the image
 */
static void
HarrisResponses(const Mat& img, const Mat& diff_x, const Mat& diff_y,
                std::vector<KeyPoint>& pts,
                Mat& response, int blockSize, float harris_k)
{
    CV_Assert( img.type() == CV_8UC1 && blockSize*blockSize <= 2048 );

    size_t ptidx, ptsize = pts.size();

    const int* dx00 = diff_x.ptr<int>();
    const int* dy00 = diff_y.ptr<int>();
    int step = diff_x.step1();
    int r = blockSize/2;

    float scale = 1.f/(4 * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;

    AutoBuffer<int> ofsbuf(blockSize*blockSize);
    int* ofs = ofsbuf;
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i*blockSize + j] = (int)(i*step + j);


    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        float kp_x = pts[ptidx].pt.x;
        float kp_y = pts[ptidx].pt.y;
        int x0 = (int)kp_x;
        int y0 = (int)kp_y;

        float xd = kp_x - (float)x0;
        float yd = kp_y - (float)y0;
        //float xd_inv = 1 - xd;
        //float yd_inv = 1 - yd;

        const int* dx0 = dx00 + (y0 - r)*step + x0 - r;
        const int* dy0 = dy00 + (y0 - r)*step + x0 - r;
        float a = 0, b = 0, c = 0, d = 0, e = 0; // e is for debug

        for( int k = 0; k < blockSize*blockSize; k++ )
        {

            const int* dx = dx0 + ofs[k];
            const int* dy = dy0 + ofs[k];
            float Ix = (float)dx[-1]*(1-xd) + (float)dx[0] + (float)dx[1]*xd + (float)dx[-step]*(1-yd) + (float)dx[step]*yd;
            float Iy = (float)dy[-1]*(1-xd) + (float)dy[0] + (float)dy[1]*xd + (float)dy[-step]*(1-yd) + (float)dy[step]*yd;
            //int Ix = 1;
            //int Iy = 1;
            a += (Ix*Ix);
            b += (Iy*Iy);
            c += (Ix*Iy);
            d += Ix;
            e += Iy;
        }
        // Debug info:
        if ( ptidx == 0 && pts[ptidx].octave == 3 )
        {
            cout << "\n\npoint: " << kp_x << ", " << kp_y << "\n";
            cout << "xd: " << xd << ", x0: " << x0 << "\n";
            cout << "a: " << a << "\tb: " << b << "\tc:" << c << "\n";
            //cout << diff_x(Range(kp_y-r,kp_y-r+5), Range(kp_x-r, kp_x-r+5)) << "\n";
            //cout << diff_y(Range(kp_y-r,kp_y-r+5), Range(kp_x-r, kp_x-r+5)) << "\n";
            //cout << diff_y(Range(kp_y-2,kp_y+2+1), Range(kp_x-2, kp_x+2+1)) << "\n";
            //cout << img(Range(kp_y-2,kp_y+2+1), Range(kp_x-2, kp_x+2+1)) << "\n";
        }
        pts[ptidx].response = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;

        response.at<float>(ptidx,0) = (float)a;
        response.at<float>(ptidx,1) = (float)b;
        response.at<float>(ptidx,2) = (float)c;
        response.at<float>(ptidx,3) = (float)d;
        response.at<float>(ptidx,4) = (float)e;
        pts[ptidx].class_id = ptidx;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline float pick( const Mat& img, float x, float y )
{
    int step = img.step;
    int x0 = (int)x;
    int y0 = (int)y;
    float xd = 1 - (x - (float)x0);
    float yd = 1 - (y - (float)y0);
    const uchar* ptr = img.ptr<uchar>() + y0*step + x0;
    return ptr[-1]*xd + ptr[0] + ptr[1]*(1 - xd) + ptr[-step]*yd + ptr[step]*(1 - yd);
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

static void
computeSkew( const Mat& responses, Mat& skew, int nkeypoints, int debug_idx )
{
    Mat corr(2, 2, CV_32F), eig_vec, eig_val;
    float* corr_p = corr.ptr<float>();
    for ( int i = 0; i < nkeypoints; i++ )
    {
        // Declare pointers to the rows corresponding to keypoint `i`
        const float* resp_row = responses.ptr<float>(i);
        float* skew_row = skew.ptr<float>(i);
        int skew_sgn = sgn(resp_row[3]);

        corr_p[0] = resp_row[0];
        corr_p[1] = resp_row[2];
        corr_p[2] = resp_row[2];
        corr_p[3] = resp_row[1];

        // Find eigenvectors
        eigen(corr, eig_val, eig_vec);
        float* val_p = eig_val.ptr<float>();
        const float* vec_p = eig_vec.ptr<float>();
        if ( i == debug_idx )
        {
            cout << "Ix: " << resp_row[0] << ", Iy: " << resp_row[1] << "\n";
            cout << "eig_val:\n" << eig_val << "\n";
        }

        // Normalize eigen values
        const float val_sq = std::sqrt(val_p[0]*val_p[1]);
        val_p[0] = std::sqrt(val_p[0] / val_sq);// * skew_sgn;
        val_p[1] = std::sqrt(val_p[1] / val_sq);// * skew_sgn;

        // Calculate transformation matrix based on the matrix multiplication
        // of skew `diag(eig_val)` and rotate [-1*vec[1] vec[0]; vec[0] vec[1]]
        skew_row[0] = -1*-1*vec_p[1]*val_p[0];
        skew_row[1] = -1*vec_p[0]*val_p[1];
        skew_row[2] = -1*vec_p[0]*val_p[0];
        skew_row[3] = -1*vec_p[1]*val_p[1];

        if ( i == debug_idx )
        {
            cout << "eig_val_norm:\n" << eig_val << "\n";
            cout << "eig_vec:\n" << eig_vec << "\n";
            cout << "skew:\n" << skew.row(i) << "\n";
        }
    }
}

static void generatePoints( Mat& points, int npoints, int patchSize )
{
    RNG rng(0x34985710);
    float u;
    float width = 5;
    float* p = points.ptr<float>();
    int l = npoints*2;
    for( int i = 0; i < l; i++ )
    {
        u = rng.uniform(-width, width);
        if (u >= 0)
            p[i] = exp(-1*u);
        else
            p[i] = -1*exp(u);
    }
    points *= patchSize;
    //cout << "points:\n" << points << "\n\n";
}

static void
computeDAFTDescriptors( const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
                       const std::vector<float>& layerScale, const Mat& harrisResponse, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, int dsize, int patchSize, int debug_idx )
{
    // Compute skew matrix for each keypoint
    int nkeypoints = (int)keypoints.size();
    Mat skew(nkeypoints, 4, CV_32F), points_kp, s, img_roi;
    computeSkew(harrisResponse, skew, nkeypoints, debug_idx);

    // Now for each keypoint, collect data for each point and construct the descriptor
    KeyPoint kp;
    for (int i = 0; i < nkeypoints; i++)
    {
        // Find image region of interest in image pyramid
        kp = keypoints[i];
        //float scale = 1.f / layerScale[kp.octave];
        // TODO: Instead of scaling here, we might as well scale when generating the points
        float scale = layerScale[kp.octave] * 0.58;
        int x = kp.pt.x;
        int y = kp.pt.y;
        //img_roi = imagePyramid(layerInfo[kp.octave]); // TODO: this can break for unsupported keypoints
        img_roi = imagePyramid(layerInfo[0]);
        //int step = img_roi.step;
        uchar* desc = descriptors.ptr<uchar>(i);
        const float* p = (const float*)points; // points are defined at line 57
        const float* s = skew.ptr<float>(i);
        if ( debug_idx == i )
        {
            cout << "kp_x: " << kp.pt.x << ", kp_y: " << kp.pt.y << "\n\nTranslated point:\n";
        }

        float min_val = 9999, max_val = 0, picked = 0;
        unsigned int min_idx = 0, max_idx = 0, byte_val = 0, l = (unsigned int)dsize*8;
        for (unsigned int j = 0; j < l; j++)
        {
            // Matrix multiplication by hand
            //float x0 = (p[2*j]*s[0] + p[2*j+1]*s[2] + x * scale);
            //float y0 = (p[2*j]*s[1] + p[2*j+1]*s[3] + y * scale);
            // points * skew.T + kp_pos
            float x0 = (p[2*j]*s[0] + p[2*j+1]*s[1])*scale + x;
            float y0 = (p[2*j]*s[2] + p[2*j+1]*s[3])*scale + y;
            if ( debug_idx == i && j < 10 )
            {
                cout << "orig  p[" << j << "]: " << p[2*j] + x << ", " << p[2*j+1] + y << "\n";
                cout << "trans p[" << j << "]: " << x0 << ", " << y0 << "\n";
            }

            picked = pick(img_roi, x0, y0);
            if (picked < min_val)
            {
                min_idx = j % 4;
                min_val = picked;
            }
            if (picked > max_val)
            {
                max_idx = j % 4; // j & 3??
                max_val = picked;
            }
            if ((j+1) % 4 == 0) {
                if ((j+1) % 8 == 0)
                    desc[(j >> 3)] = (uchar)((byte_val << 4) + (max_idx + (min_idx << 2)));
                else
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

static void computeDiffX(const Mat& image, Mat& output)
{
    for ( int i = 0; i < image.rows; i++)
    {
        const uchar* img_row = image.ptr<uchar>(i);
        int* out_row = output.ptr<int>(i);
        out_row[0] = 0;
        out_row[image.cols-1] = 0;
        for ( int j = 1; j < image.cols-1; j++)
        {
            out_row[j] = (int)(img_row[j-1] - (int)img_row[j+1]);
        }
    }
}

static void computeDiffY(const Mat& image, Mat& output)
{
    int* out_first = output.ptr<int>(0);
    int* out_last = output.ptr<int>(image.rows-1);
    for ( int j = 0; j < image.cols; j++)
    {
        out_first[j] = 0;
        out_last[j] = 0;
    }
    for ( int i = 1; i < image.rows-1; i++)
    {
        const uchar* img_row_top = image.ptr<uchar>(i-1);
        const uchar* img_row_bot = image.ptr<uchar>(i+1);
        int* out_row = output.ptr<int>(i);
        for ( int j = 0; j < image.cols; j++)
        {
            out_row[j] = (int)(img_row_top[j] - (int)img_row_bot[j]);
        }
    }
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
                                Mat& maskPyramid,
                                Mat& diff_x, Mat& diff_y)
{
    // Initialize values for level 0
    Mat prevImg     = image, prevMask = mask;
    Rect linfo      = layerInfo[0];
    Size sz         = Size(linfo.width, linfo.height);
    Rect wholeLinfo = Rect(linfo.x - border, linfo.y - border, sz.width + border*2, sz.height + border*2);
    Mat extImg      = imagePyramid(wholeLinfo), extMask;
    // Compute diffs for level 0
    Mat cur_diff_x = diff_x(linfo);
    Mat cur_diff_y = diff_y(linfo);
    computeDiffX(image, cur_diff_x);
    computeDiffY(image, cur_diff_y);

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

        // Compute diffs
        cur_diff_x = diff_x(linfo);
        cur_diff_y = diff_y(linfo);
        computeDiffX(currImg, cur_diff_x);
        computeDiffY(currImg, cur_diff_y);

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
static int computeKeyPoints(const Mat& imagePyramid,
                             const Mat& maskPyramid,
                             const Mat& diff_x, const Mat& diff_y,
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

    Mat cur_response(nfeatures, 5, CV_32F);
    int responseOffset = 0;
    int focus_offset = 0;

    for( level = 0; level < nlevels; level++ )
    {
        int featuresNum = nfeaturesPerLevel[level];
        Mat img = imagePyramid(layerInfo[level]);
        Mat dx = diff_x(layerInfo[level]);
        Mat dy = diff_y(layerInfo[level]);
        Mat mask = maskPyramid.empty() ? Mat() : maskPyramid(layerInfo[level]);

        // Detect FAST features, 20 is a good threshold
        Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(fastThreshold, true);
        fd->detect(img, keypoints, mask);

        // Remove keypoints very close to the border
        KeyPointsFilter::runByImageBorder(keypoints, img.size(), edgeThreshold);

        // Keep more points than necessary as FAST does not give amazing corners
        KeyPointsFilter::retainBest(keypoints, 2 * featuresNum);

        // Filter remaining points based on their Harris Response
        HarrisResponses(img, dx, dy, keypoints, cur_response, 7, HARRIS_K);
        KeyPointsFilter::retainBest(keypoints, featuresNum);
        if ( level == 3 )
        {
            keypoints[0].pt.x = 388.f; //
            keypoints[0].pt.y = 255.f;
            focus_offset = responseOffset;
        }


        nkeypoints = (int)keypoints.size();
        int index;

        for( i = 0; i < nkeypoints; i++ )
        {
            //cout << "pt: (" << keypoints[i].pt.x << ", " << keypoints[i].pt.y << ")\n";
            keypoints[i].octave = level;
            keypoints[i].size = patchSize*layerScale[level];
            keypoints[i].pt *= layerScale[level];
        }
        //Mat img_0 = imagePyramid(layerInfo[0]);
        //Mat dx_0 = diff_x(layerInfo[0]);
        //Mat dy_0 = diff_y(layerInfo[0]);
        //HarrisResponses(img_0, dx_0, dy_0, keypoints, cur_response, 21, HARRIS_K);
        //for( i = 0; i < nkeypoints; i++ )
        //{
        //    index = keypoints[i].class_id;
        //    keypoints[i].class_id = 0;
        //    float* response_row = response.ptr<float>(i + responseOffset);
        //    float* cur_row = cur_response.ptr<float>(index);
        //    for ( int j = 0; j < 5; j++ )
        //        response_row[j] = cur_row[j];
        //}

        responseOffset += nkeypoints;
        std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(allKeypoints));
        //if ( level == 3 ) {
        //    cout << "scale: " << layerScale[level] << " level: " << level << "\n";
        //    cout << "pt: (" << allKeypoints[focus_offset].pt.x << ", " << allKeypoints[focus_offset].pt.y << ")\n";
        //    cout << response.row(focus_offset) << "\n";
        //}
    }
    return focus_offset; // For debugging
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
    int debug_idx = 0;

    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;

    int border = std::max(edgeThreshold, patchSize);

    Mat orig_image = _image.getMat();
    Mat image, mask = _mask.getMat();
    if( orig_image.type() != CV_8UC1) {
        image = Mat(orig_image.size(), CV_8U);
        Vec3b* img_ptr = orig_image.ptr<Vec3b>();
        uchar* out = image.ptr<uchar>();
        int limit = orig_image.rows*orig_image.cols;
        for ( int i = 0; i < limit; i++ )
        {
            out[i] = (uchar)( ((int)img_ptr[i][0] + img_ptr[i][1]*2 + img_ptr[i][2])/4 );
        }
        //cvtColor(_image, image, COLOR_BGR2GRAY);
    }
    else
        image = orig_image;
    int kp_x = 670, kp_y = 440;

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
    Mat imagePyramid, maskPyramid, diff_x, diff_y;
    Mat harrisResponse(nfeatures, 5, CV_32F);
    Size bufSize = computeLayerInfo(image, border, scaleFactor, nLevels,
                                     layerInfo, layerOfs, layerScale);
    // create image pyramid
    imagePyramid.create(bufSize, CV_8U);
    diff_y = Mat::zeros(bufSize, CV_32S);
    diff_x = Mat::zeros(bufSize, CV_32S);
    if( !mask.empty() )
        maskPyramid.create(bufSize, CV_8U);
    computeImagePyramid(image, border, nLevels, layerInfo,
                        layerOfs, layerScale, imagePyramid, mask,
                        maskPyramid, diff_x, diff_y);

    // Get keypoints, those will hopefully be far enough from the border that no check will be required for the descriptor
    if( do_keypoints ) {
        debug_idx = computeKeyPoints(imagePyramid, maskPyramid, diff_x, diff_y,
                         layerInfo, layerScale, keypoints, harrisResponse,
                         nfeatures, scaleFactor, edgeThreshold, patchSize, fastThreshold);
        HarrisResponses(image, diff_x(layerInfo[0]), diff_y(layerInfo[0]), keypoints, harrisResponse, 21, HARRIS_K);
    }
    else // supplied keypoints
    {
        KeyPointsFilter::runByImageBorder(keypoints, image.size(), edgeThreshold);
        HarrisResponses(image, diff_x(layerInfo[0]), diff_y(layerInfo[0]), keypoints, harrisResponse, 21, HARRIS_K);
    }

    if( do_descriptors )
    {
        int dsize = descriptorSize();
        nkeypoints = (int)keypoints.size();
        _descriptors.create(nkeypoints, dsize, CV_8U);
        Mat descriptors = _descriptors.getMat();
        computeDAFTDescriptors(imagePyramid, layerInfo, layerScale, harrisResponse,
                               keypoints, descriptors, dsize, patchSize, debug_idx);
    }
}

Ptr<DAFT> DAFT::create(int nfeatures, int size, int patchSize, float scaleFactor, int nlevels, int edgeThreshold,
         int fastThreshold)
{
    return makePtr<DAFT_Impl>(nfeatures, size, patchSize, scaleFactor, nlevels, edgeThreshold,
                              fastThreshold);
}

