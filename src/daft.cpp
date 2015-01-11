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

static float points[256*4*2*8] =
{
    -0.6384, -21.3528, 0.8207, -6.3876, -13.9960, 0.7035, 1.1853, -11.1793,
    15.6191, 3.2871, 0.6527, 0.9638, 17.1387, -0.8499, 21.1871, -1.1104,
    -3.1565, -12.5668, -3.8384, -3.7650, -20.8557, -0.9128, -3.6274, -7.3360,
    5.9993, 21.4109, 3.8764, 6.9227, -0.6282, 7.7259, 1.5888, 28.0376,
    -9.9741, 12.3212, 0.8163, -0.9316, 9.8837, 2.0468, -16.8685, 14.7806,
    25.3404, -5.9502, 13.0839, 2.6834, -1.4065, -1.1926, 19.4758, -0.7156,
    28.3839, -3.5815, 1.9257, -0.8200, 28.1008, -7.8564, 27.7488, -2.7883,
    -20.5217, -5.4825, -5.7094, 9.0267, 1.9414, -8.0059, -28.6256, 5.1320,
    3.5538, 1.6428, 0.5616, 11.1904, 1.8489, -0.6803, 2.4977, 9.3191,
    17.7907, 18.9261, 19.2287, 22.4785, -7.6128, -1.5332, 0.6322, 18.1516,
    -23.1147, -0.7810, -21.3122, 2.1435, -0.9762, 2.8585, -17.3017, 1.1164,
    0.7157, 16.1585, -10.4134, 3.0722, 1.0093, 17.3191, -0.5717, 13.3949,
    -28.2941, -21.9200, 6.2066, -21.4722, 18.1909, -18.4115, -3.7689, -12.5180,
    27.7428, 11.9597, 0.7130, -0.8174, 0.5731, 5.7164, 9.8005, -3.4805,
    1.4656, 1.4337, -1.7338, -1.0667, 5.7616, 1.4493, -1.1746, -0.7338,
    -14.5146, -24.8012, -23.2754, 13.9276, -19.2613, 3.4878, -19.9992, 8.2922,
    -3.1227, 0.6351, -0.5867, -2.0225, -1.0219, 5.8093, -0.9970, -1.7115,
    12.3003, 1.0563, 9.7625, -5.7559, -8.2878, -7.5789, 9.0769, 1.2673,
    -26.7958, 13.3671, -9.0045, 3.8514, -25.6067, -3.6250, -27.0673, 1.3614,
    -9.1505, -18.8630, 16.1054, 0.8658, -1.3243, 17.4471, -1.6084, -19.8980,
    1.9512, -6.1105, 1.5942, -4.5267, -2.3625, -13.6332, 12.6955, -15.3377,
    27.8615, 1.6283, 3.2831, 10.4792, -1.3872, 21.1338, 11.9992, 17.7238,
    -2.1809, -1.2969, 2.1662, 8.3240, -0.8982, -4.7421, -1.4619, 1.9309,
    6.4156, -27.3539, 7.3044, -1.8020, -14.9555, 24.0384, 19.3651, -16.9305,
    -28.6694, 1.4726, 14.6984, -20.9430, -11.5110, 0.6119, -24.2476, 1.9212,
    -2.6102, -2.8944, 4.5745, 4.4727, -4.9730, -6.8620, -3.1474, -3.4958,
    -14.1186, 8.0536, -1.6302, 9.4566, -6.7138, 1.6077, -4.4142, 6.3013,
    2.9902, 26.9599, 1.2044, -1.1486, 20.8771, -0.7093, 3.8865, 23.7145,
    -0.5308, 13.3579, -7.1753, 13.4758, -1.4728, -4.3305, 2.5003, 13.0516,
    13.4007, -1.2119, 23.5521, 24.3387, -18.0932, -1.4405, 21.2746, 6.3623,
    -25.0436, 4.9007, -8.8047, -13.6610, 5.7302, 2.2132, -22.3628, -1.6824,
    5.6507, -20.1184, -3.1747, 0.8201, 1.0186, 6.1959, -2.4043, -25.4219,
    -11.6443, 4.3584, -2.7165, 2.0404, 0.5455, 15.5453, -6.8926, 4.2196,
    26.9457, 3.8882, -6.3845, -0.8883, -5.9665, -1.3117, 20.8025, 9.3396,
    -0.6558, -17.0740, 0.6406, 1.1939, -4.8008, -3.2119, 2.5130, -21.5655,
    -19.4836, -1.8260, -9.6454, -11.7513, 0.9170, 0.5851, -22.4017, 0.5940,
    -2.2736, 21.1066, 4.2728, 2.3857, 0.6221, 0.8637, -15.4731, 25.6663,
    -2.9161, -1.6881, -2.1413, 1.0982, -9.8181, 0.6244, -3.0070, -0.7872,
    -0.5450, 2.0594, 6.4062, -1.8820, -9.1412, 8.9223, 2.1176, 0.7736,
    5.5272, 18.9608, 1.9105, -9.2775, 14.5053, -1.7190, 19.0614, 26.7777,
    8.0997, 2.5003, -0.9175, -20.0905, 2.9406, 10.0284, 10.0169, -1.4332,
    -7.0239, -9.4732, 2.4583, 3.7960, -25.8883, -1.9406, -9.8151, -12.8371,
    23.5628, -7.0248, 4.8863, 16.6966, 0.5636, -27.0077, 12.6591, -10.9332,
    10.5394, 5.7264, -10.4066, 3.9972, 1.6262, 1.1524, 6.0815, 4.7501,
    -22.1078, -27.1891, -3.2071, -6.0159, -1.0997, -13.6381, 1.6220, -23.4967,
    -4.5766, -1.8579, 14.3269, -8.6267, -7.6895, -2.8401, -6.6734, -3.5956,
    -2.1022, 15.5856, 2.4095, 0.9382, 1.0355, 26.0964, 3.0421, 9.5834,
    14.6427, 9.3595, 7.0710, -1.4089, -2.7289, 1.8744, 26.4357, 14.1850,
    -2.6667, 13.5034, 15.9299, -22.2593, 7.1146, -17.9184, -2.6936, 20.8213,
    -26.0997, -7.0820, -6.8707, 12.2058, 19.4988, 2.2483, -24.2703, 0.5628,
    -8.0840, 0.6592, -1.9605, 1.9829, -7.5026, 8.2579, -7.1009, 3.6213,
    8.1935, -4.5019, 24.8847, 1.2688, 3.9525, -9.2594, 4.0228, -7.3273,
    -14.4282, 7.6866, -8.1007, 0.5872, -0.7165, -0.9183, -17.2304, 19.8362,
    0.5562, -11.4426, 2.2924, 2.2433, 0.8989, -2.6350, -3.1257, -14.2733,
    -18.2071, 6.3554, -3.2773, 0.6625, -8.3000, -1.2039, -5.8060, 0.8027,
    19.2040, -2.7752, 3.5271, -11.8640, 11.4861, -19.9860, 27.7389, 25.5853,
    0.6953, 17.2644, 3.1094, -1.1610, 0.7394, 24.4877, -6.8300, 15.2196,
    -0.6417, -8.0189, 1.1964, 0.7137, -0.8129, -27.2998, 0.6973, -7.3595,
    -19.5491, -4.4930, 19.2337, -2.7605, 4.1430, -2.1717, -16.1282, -2.4699,
    26.9026, -7.0295, 1.0635, 2.3810, -1.4668, 1.8120, 26.3143, 1.9589,
    5.2030, 8.6993, -5.1563, 2.5418, 17.4750, 3.1481, 1.3304, 5.8284,
    -13.9748, -5.9107, 1.6335, -19.2931, -12.9377, -0.7080, -2.3989, -8.0673,
    4.7663, -1.9724, -0.8855, -0.8606, 5.9057, 1.6457, 0.7453, -1.1632,
    18.5647, 4.5944, -9.6461, -24.1481, 22.2923, -3.7295, 10.7415, 2.1449,
    10.1574, 22.2801, -1.2424, 0.8603, 16.5728, -8.0535, 1.6617, 20.7821,
    -0.6032, -20.6446, 9.2151, 13.2674, -0.9363, -0.7150, -3.2775, -23.9762,
    0.9126, 0.9961, 2.8864, 6.8235, 3.2654, -4.0747, 1.1750, -0.5574,
    -24.5623, -4.3601, 2.4939, 20.8880, -6.5121, -5.8038, -19.2405, -2.6903,
    -24.1382, 14.6245, -7.5138, 7.7968, -1.1557, 3.6538, -24.1834, 3.1042,
    16.4940, -1.3227, 5.6807, 0.5779, 1.1284, 1.4399, 28.3183, -2.1765,
    -25.5219, -27.0736, -1.9761, 19.3467, -1.4366, 1.7056, -26.2188, 10.4757,
    -0.8645, -3.7160, 2.3033, 2.7623, 1.0967, -14.3610, -0.5564, -2.4099,
    -6.6941, 2.9717, 0.7708, 27.5992, -0.5530, 1.6793, -6.2450, 28.5634,
    -11.4248, 5.0524, -3.3090, -1.2257, -1.1011, 3.7921, -4.0257, 0.8123,
    0.7633, -27.3465, 1.8566, 10.1022, 0.6948, -6.5510, 20.5890, -27.1733,
    8.1220, -4.7947, 2.1218, -0.9780, 7.8793, -24.9486, 10.2308, -15.0339,
    1.1214, 15.6808, -7.4421, -19.2561, -22.2868, -4.6614, -4.1037, 14.8249,
    2.0710, 5.8972, 3.1904, 1.7091, -3.0155, 16.7815, 8.5594, 4.8878,
    -4.7673, -5.1979, -9.9390, -11.5917, 1.4539, 28.6247, -14.1153, -8.1910,
    4.8430, -4.0881, -0.5290, -14.6613, 0.8303, -2.7434, 16.4051, -4.3567,
    -14.0684, 11.6402, -0.9977, -1.0367, -21.4686, 4.0757, -15.8826, -1.7677,
    10.6537, -7.8108, 0.7332, 20.7099, 25.9132, 22.6135, 17.8488, -18.3100,
    2.1619, 8.0053, 1.0797, 5.9433, 3.3376, -1.0676, 9.9960, 10.2808,
    -21.6579, 0.7735, -6.6384, -1.5107, 22.6926, 3.8829, -7.6422, -1.7455,
    -0.7118, 9.1848, 26.4946, 5.2437, -9.2976, -2.5763, 0.6705, 10.6876,
    -24.2378, -3.2548, 6.7887, -7.0120, 1.0994, 2.0848, -21.4424, -1.5069,
    0.5563, -1.0066, -2.8554, -9.4918, -0.7837, -2.0235, 1.9688, -0.5703,
    -13.1425, 12.1942, -22.1095, -3.9891, -1.0322, 22.0829, -0.6014, 12.3296,
    23.2840, -2.5287, 1.4865, -3.6635, 1.6996, 7.9421, 19.5295, -1.0922,
    -5.1758, -1.5746, 1.1832, -5.0912, -26.0853, 0.9791, -1.8590, -3.4284,
    -8.2276, -20.3371, 4.1573, 3.3663, 0.8453, -0.6270, 1.3582, -18.6015,
    -3.9938, 4.6213, 4.9269, -0.6359, -1.2938, 2.1254, -7.6976, 4.9034,
    0.6471, 6.6678, -0.6094, -23.4277, -15.6794, 22.6829, -0.6822, 7.4985,
    -21.4170, 6.4531, -1.5129, -0.7321, 12.1881, -22.1646, -24.7580, 0.6494,
    1.6385, -1.3269, -0.7572, 6.2842, 0.8886, -0.8234, 2.6586, -0.8998,
    3.4964, 5.8149, 2.2132, 13.2935, -5.4642, 2.0418, 2.0526, 4.2844,
    -8.3851, 1.2086, 1.2186, 20.5586, -13.0667, -16.2969, -8.9793, 4.3532,
    2.2383, -14.7422, -1.8654, -1.5280, 0.8487, -2.9528, 11.8895, -24.3193,
    8.7333, -4.7571, -9.0452, -10.0046, 2.2535, -27.5819, -12.4520, -23.7027,
    -3.0946, -6.9957, 7.4381, -1.2029, 3.6870, 0.6020, -5.7597, -7.7269,
    23.4002, -1.9300, 12.5652, 22.6163, -0.9435, 1.1941, 13.8644, 14.4268,
    10.6666, 10.5920, 12.8437, 0.5799, 0.8945, 16.0679, -0.8285, 13.0261,
    -17.0432, 3.2497, -6.3979, -4.3563, 1.8368, -0.9866, -25.4268, 8.1047,
    -2.2108, 3.0369, -15.2502, -0.6719, 2.1303, -1.4779, -3.9335, 3.0776,
    8.9374, -0.8137, -0.5397, -1.3165, 21.3923, 12.5932, 5.3641, 0.7104,
    -20.3976, -0.6718, 10.8230, -9.2080, -6.5505, -1.3102, -14.1818, -15.8660,
    -2.5483, -21.8704, -0.9103, -2.2696, -11.7107, -9.5564, 22.4411, -17.0509,
    -28.6997, 12.4416, -0.6230, 13.7356, 1.0831, -2.0508, -26.0319, -0.5408,
    -7.6530, -6.4482, -8.8866, 5.7249, 22.9373, 27.2008, -12.2877, -2.4370,
    16.0251, 0.5370, -1.3034, -23.5394, 0.7631, 0.6226, 25.6148, -0.6560,
    -22.0017, -14.3025, -22.7734, 15.4187, 1.4206, 1.3905, -16.5604, 11.9783,
    -1.1361, 1.0336, -0.7512, -4.0936, 0.8099, 6.0228, -0.5658, 1.0276,
    -3.4688, 25.7720, 4.9312, -1.2958, -5.4301, -8.3272, 8.5764, 19.3131,
    0.6288, 7.3264, 7.0002, 2.5519, 0.5607, 3.3479, -0.5888, 11.8109,
    -18.8138, -11.0887, 2.3240, 1.5650, 0.7881, 10.4362, -28.5805, -1.4264,
    19.2542, 11.2505, 11.1192, -3.0405, 8.0881, 15.1944, 4.5291, 4.5953,
    -1.5458, -9.1370, -3.0314, -24.5408, 0.5843, -11.9200, 1.3311, -22.4217,
    -2.6890, 9.4157, 6.2835, 5.6100, -0.6397, 6.6337, -23.0792, 27.3047,
    -5.4091, 0.7330, -2.7871, -5.3321, 1.0823, 2.3285, -3.6304, 1.1537,
    9.6670, 7.2547, 15.1155, -3.0982, 15.2555, -20.3882, 10.4726, 0.7936,
    5.6263, 28.5237, 26.2345, -0.6747, 0.9699, 18.7696, -22.7125, -2.7505,
    7.9039, -5.3184, -9.9651, 0.6658, -7.1770, -16.7512, 9.2380, -8.4434,
    3.5959, -18.6001, -4.6525, 2.5369, 26.2847, -0.5530, -1.8176, -13.3109,
    -18.2053, 5.0495, 4.5005, 14.6295, -1.6106, 2.0649, -6.6925, 4.7596,
    15.0729, 0.6215, 4.5791, 1.3360, -9.8652, -13.4607, 19.8149, 24.4667,
    -9.1181, 22.7167, -0.6596, -5.7014, -27.1984, -0.7267, -26.0333, -5.1227,
    15.4291, 5.2051, 1.1144, -14.4009, 0.9122, 13.6173, 25.7854, -6.0009,
    -1.0659, 1.4230, 5.5809, 1.5372, 1.2938, -1.1347, 0.9835, 0.7137,
    -3.5522, -1.6509, -1.8474, -1.1571, -4.9049, -0.7222, -14.5048, -8.5662,
    -5.7474, -23.3891, 0.6517, 0.8750, 16.3735, -20.4790, -26.7050, 27.2410,
    3.5343, 5.7580, 1.0422, 7.6504, 0.9354, 1.1249, 0.5457, 2.3171,
    24.4313, -28.6297, -2.7000, 16.9604, 24.9199, -5.3841, -4.3260, 7.3560,
    -2.1492, -2.4417, -8.1172, 3.1120, -7.6056, -7.7468, -1.6780, -0.5824,
    4.9491, -2.4767, -1.3757, -17.4711, 0.6343, -3.3184, 12.0227, 0.8984,
    27.2816, 21.3325, 13.5136, 1.6540, -4.4992, -5.9793, 20.5571, -3.0963,
    -10.4649, -16.8019, -1.6349, -18.5484, 1.3837, 20.6102, 5.5408, -13.3736,
    -2.0304, 25.9246, -1.9163, -2.0180, -25.1447, -3.2462, -4.9759, 12.6840,
    -7.9789, -1.4037, 5.5930, 0.7889, 0.8182, 3.3630, -6.6433, -1.3668,
    -1.2177, 7.2233, -9.9550, -0.6534, 2.1422, 24.2494, -14.1814, 11.0497,
    -27.4866, -1.9848, -0.6449, -4.9012, -3.5306, -2.9904, -19.4764, 0.8602,
    1.4769, -26.4731, 3.2882, -2.7719, -15.7419, -4.2430, 22.5115, -17.3625,
    13.0883, -21.8352, 19.9635, 12.8349, -0.8745, -2.6405, 13.1172, -5.3505,
    0.8753, 13.5812, -3.7409, 11.1318, 3.7625, 0.8422, 27.2741, 17.3556,
    -27.6148, 25.2592, 27.8594, -3.3252, 0.7524, -3.9223, 15.6201, -12.6376,
    2.5892, -3.6443, -1.2170, -4.0195, 9.1603, 0.6070, 13.6963, -0.6476,
    -7.6742, 15.3480, 0.8688, 1.7408, -1.4750, 0.8286, 4.5565, 17.1261,
    -0.8028, -21.3905, -9.2814, -1.6295, -21.7818, 1.7742, -14.2444, -5.8850,
    1.6356, 5.4398, 18.0622, -1.2515, -0.6915, 9.2676, 0.9165, 3.5895,
    -18.2267, 0.7555, 1.8531, 1.6877, -0.5615, 5.0378, -23.2366, 0.5756,
    -0.6692, -1.7182, 2.5689, 1.3930, 14.2893, -3.1131, 1.1610, -1.0248,
    -13.7302, 8.7343, 16.2726, -25.8626, 18.9765, -10.7870, -4.5959, 18.0021,
    1.8258, -7.2047, -6.2456, 0.7096, -6.6915, -18.0857, 4.9414, -14.9980,
    -9.5371, -4.8288, -4.7707, 0.7545, 2.8209, -5.7685, -20.5883, -0.6473,
    -2.5568, 11.9415, -1.0949, -2.5709, -1.4862, 17.5107, -1.5264, 1.1370,
    16.4025, -0.7047, -6.0579, -2.1314, -3.8747, 3.0084, 12.1533, -2.3362,
    8.7824, 7.9742, 10.4517, -0.6554, -1.8133, -0.6497, 2.2165, 6.6317,
    1.4032, -11.7825, 0.7797, 2.6042, 24.6200, 22.0163, -18.5519, -25.1727,
    -8.4537, 0.6411, -0.5840, 2.5551, -1.3490, -0.7592, -1.3109, 1.2840,
    -3.3465, 3.0175, -6.8756, 2.7558, 1.5391, 0.7588, 0.5408, 4.5333,
    13.7350, 13.3968, 0.9736, 1.0169, 17.0261, 18.4620, 5.9227, 0.6105,
    -24.8058, 4.3306, 3.1847, 8.6266, -1.7229, 5.2368, -18.5661, -3.6793,
    19.3635, 1.9176, -25.1882, -12.4203, -2.9024, -0.8490, 18.2609, 17.9533,
    0.9238, -0.8868, -2.4200, -0.6944, 5.4977, -2.7233, -1.1115, 1.2410,
    -9.5849, 12.1208, -7.4243, -1.1640, 18.1531, 6.9587, -6.2154, 1.8868,
    23.5078, 7.6338, -20.6163, 25.9364, 3.4005, -1.8396, 6.6453, 1.6878,
    3.9704, -13.9959, 1.5016, -0.8344, -0.5767, -0.8105, 9.8391, -9.2142,
    -11.4010, -0.7548, -11.0214, -4.8761, -0.9335, -0.7990, -26.2733, -5.4346,
    12.1395, -15.5861, 7.3328, 1.7026, -2.1682, -9.6340, 6.8543, -1.4206,
    -21.2962, -21.8380, -14.2075, -9.3660, 7.9168, 4.2952, -23.3708, 8.0523,
    -22.0213, 21.5680, -23.9723, -3.5001, 1.1541, 24.3705, -0.8675, 7.5495,
    -0.8948, -6.5093, -0.7535, -2.2960, 1.8573, -19.9932, -0.5820, -27.5437,
    -2.6065, -4.6112, 7.9014, 4.9236, -5.4115, -3.9555, 1.3196, -6.7196,
    22.5101, -2.6495, 3.8280, 5.3068, 6.7888, -7.7862, 4.2007, 25.2429,
    -10.9003, 0.7969, -1.9029, 1.4666, 0.5412, 0.6157, -10.7477, 2.5014,
    -13.8997, -10.7094, 6.6523, 10.9205, -2.5877, -1.4317, -3.7150, -14.0908,
    -27.9147, 11.7848, 20.8573, -5.7305, 19.4543, -0.5435, 15.1792, 21.1816,
    0.6863, 6.4891, 1.6083, 1.1182, 0.5496, 4.5236, 0.5822, 8.8386,
    -11.8516, -17.5044, -3.0077, -13.7850, -1.5308, -25.9746, -5.2455, -3.5061,
    -5.6466, 2.3640, -6.6187, -4.3141, 2.7030, 2.4876, -3.5086, 0.5468,
    13.1141, 1.8710, 0.8806, -19.0474, -2.4368, 14.0830, 24.3798, 0.9621,
    4.5932, 12.5924, 0.6494, -18.0140, 2.6875, 22.8027, -7.6010, -22.9714,
    -16.6296, 10.3857, 1.8646, -1.6343, 9.7218, -25.5113, -11.2622, 3.8952,
    12.3767, -5.4934, 15.0794, 5.6328, 4.1228, 26.9628, 4.8968, -11.1904,
    -16.8790, -5.2738, -0.6742, 7.1028, 23.6200, -0.5942, -9.6571, -6.9732,
    1.3939, -26.8722, 1.3967, -9.6487, -2.0942, -10.0641, 28.7096, 24.5699,
    -24.0104, 1.3298, 2.8349, 26.1344, -22.9528, -2.1851, -11.9692, -0.8489,
    -0.6765, -2.4560, -1.0065, -4.4553, 5.0384, 0.8499, -2.1563, -1.3250,
    1.7692, 8.5533, 4.3507, -0.7672, -6.3655, 7.4126, -2.8881, 24.5392,
    7.7515, 20.2669, 0.5819, 16.1913, -18.7478, 0.8903, -28.2213, -0.7852,
    0.5510, -1.2716, -0.5778, 1.1793, 0.7618, -8.8699, 2.2276, -1.4220,
    2.9466, -2.2799, 24.4633, -23.2700, 28.7137, -1.8616, 12.0768, -19.2019,
    -15.5434, 0.8231, 4.8971, 1.9097, -4.4207, 19.0418, -25.2156, 26.5711,
    1.4347, 1.2079, 1.9897, 0.6587, -13.8476, -12.2469, 1.6718, 1.3490,
    1.7316, -9.4805, -11.9465, -0.9224, 8.3446, -27.5002, -1.1722, -4.8400,
    -9.5700, 5.4525, -1.9245, -1.0378, -1.0868, 3.9126, -1.4584, 4.3877,
    2.2532, 1.5048, -4.6913, 7.9522, 1.0953, -1.9446, 10.1977, 6.4005,
    1.1243, 9.1837, -1.1072, 1.9392, 18.4243, 1.7211, -1.4815, 3.4399,
    -10.4636, -17.8338, -2.4986, 5.1777, 0.9899, 23.0603, 4.2667, -25.6342,
    -4.6525, -0.5522, 0.9060, 1.8578, -7.4524, -2.1514, -4.0013, 4.1061,
    28.0961, -12.2271, -3.4999, -4.1753, 6.2651, 0.6398, 9.6478, -0.7918,
    19.1544, -6.6048, 28.4256, -6.8614, -0.6523, -1.1463, 0.9259, -8.2797,
    -8.7028, 0.5599, 0.9234, -1.2314, -9.6310, -1.2219, -17.7942, -0.7450,
    -14.1507, -7.3934, -10.8962, -5.9251, -9.3019, 0.7054, -20.3864, 6.1765,
    13.7151, -14.7877, -6.9383, -0.5835, 5.3805, 0.7172, 25.6426, -5.8965,
    5.5057, -10.3114, 0.5582, -5.9838, -6.3399, -5.6529, -3.1955, -24.1652,
    -8.0462, 7.2670, 2.6979, 2.3457, -0.9762, 0.9720, -6.6771, 8.7072,
    -1.4834, 5.3101, -3.6885, 2.9270, 19.9085, 7.3644, -9.1408, 9.8473,
    -4.8931, -0.5963, -0.8801, -25.6058, -4.1361, -3.8379, -5.0379, -3.8698,
    14.6199, 20.4675, -5.7829, -0.7330, 3.2872, -21.1579, 19.9791, -24.1397,
    -20.9673, 14.3743, -5.4005, 0.8851, -22.0390, -1.3419, 24.7201, 12.8966,
    -3.1910, 7.9638, 5.2280, 14.6405, 0.6641, -1.8644, 4.1115, 15.5001,
    -13.5275, -1.0255, 8.8234, -4.9535, -0.9190, -2.9510, -18.9861, 1.4350,
    -20.5943, -14.5614, 18.0673, -2.5331, 4.5403, 1.5218, -17.1027, -0.9305,
    21.3014, 4.9368, 0.9800, -3.9415, 13.8667, -0.8871, 2.7444, -3.3020,
    -5.7340, -1.8753, -23.9790, 0.8980, -15.3305, 4.5329, -25.1918, -1.1618,
    1.8954, 28.6547, -2.3057, 0.6177, -3.3070, 19.2579, -1.4899, 0.6105,
    28.5641, 24.2401, 0.6585, 0.9643, 0.8215, -6.8028, 14.9901, 1.8676,
    -2.3890, -13.5748, -4.2764, -0.5586, -0.7140, 3.1900, 4.8324, -11.6699,
    -2.0011, 10.3033, 0.9166, 11.1311, -20.0904, -5.6640, -1.5217, 6.0168,
    -7.0911, -6.3415, 2.0966, 3.8662, 1.5274, -4.4043, -1.6426, -4.7988,
    -13.2473, 21.9211, 10.3375, 2.2845, 2.0755, -5.2394, -28.4726, -22.2310,
    -9.3205, -18.8858, 1.5322, 0.9939, -25.0841, 16.6582, -22.7205, -5.5573,
    7.1824, 0.7325, -2.1556, 2.9382, 3.0389, -4.8601, 13.1981, -0.7041,
    5.0952, -17.5668, 0.7198, -15.0471, -1.0358, -0.5604, -0.5556, -6.9412,
    -6.9921, 1.1458, -5.1309, -1.0817, 1.2664, 6.2470, -0.9386, 3.8189,
    -0.6272, 23.2994, 5.7422, -7.0477, 24.2813, 16.9299, 22.6423, -1.7038,
    1.2928, 3.4474, -0.5664, -1.1988, 0.7063, -16.8344, 1.4539, 3.7347,
    9.9129, -0.5454, 13.0007, -9.9899, 11.7611, 1.3452, -18.1049, -21.2530,
    2.2689, 27.3568, 0.5527, -21.9979, 1.3761, 15.0857, -21.8329, -0.6018,
    8.5443, -5.8381, 0.7948, 0.6025, 8.8551, -4.7910, 10.9335, -0.7117,
    -13.7419, 1.3868, 24.7965, -5.6512, -1.1672, -0.5918, -7.6020, 3.2783,
    -4.5640, -3.1438, 0.7632, 3.6155, -24.3952, 1.3165, -2.3326, 0.5814,
    1.7269, 5.6566, 1.7659, -4.9578, -0.6818, 23.1886, -2.1615, 21.6729,
    2.0338, 11.8639, 11.9350, -0.8037, -0.7145, 0.7806, 17.3822, 14.2962,
    -0.8832, -4.7843, 3.9567, -9.5111, -0.7304, -21.9050, 1.6446, -13.3374,
    -23.9595, 2.2656, -14.1714, 16.6387, -19.8267, -7.4888, 22.6573, -24.4633,
    0.8970, -1.5625, -13.5583, 3.0405, 0.8368, -2.0108, -1.9135, 3.8111,
    3.4371, -2.8349, -2.6182, -0.9033, -0.9399, 0.7052, 0.8221, -0.9270,
    -2.0133, -13.7356, -2.2159, 1.3780, -2.1347, 4.0446, -4.0566, -18.6056,
    -12.9350, -0.7039, -0.6392, 0.5301, -0.5655, -0.8865, -16.0835, 5.5484,
    6.6248, 0.8837, -0.8538, 2.3274, -2.2451, -20.1489, 2.2632, 1.4596,
    -0.8697, 3.1821, 3.8136, 3.8826, -13.3375, -8.1537, -1.5867, 4.0727,
    -2.0579, 5.7897, 26.9150, 10.0562, -24.3994, 0.7010, 11.7872, 24.3732,
    -0.9386, -10.9678, 2.4773, 0.9720, 6.8155, -2.8396, 4.3560, -25.4589,
    2.6064, -0.6280, -3.4517, 1.1016, 3.2363, 3.6038, 1.2038, 1.5535,
    -4.4756, 5.9258, 3.2735, 5.8922, -1.6740, 22.5317, -2.1050, 4.0442,
    -6.5835, -13.6013, -1.3864, 21.2578, -1.6796, -9.8171, -19.2857, 9.9843,
    23.4304, 12.3106, 15.3684, 14.8537, 15.2658, -1.5914, -2.9844, 5.9268,
    -1.5415, 1.2381, 2.2467, -4.4131, -11.0815, 18.3719, -2.9124, 0.8918,
    2.5932, -0.5382, -22.1236, -0.9255, 0.6020, -23.4351, -1.1339, 10.2228,
    -8.6797, 17.0411, -8.6886, -27.4302, 0.5855, 10.4511, 11.4215, -23.1899,
    2.1254, -10.4962, 13.1647, -1.6182, 22.8317, -14.3346, 1.1511, 2.1103,
    -6.3613, 0.5477, 2.7318, 1.5462, -7.1948, -1.9416, -3.2842, 0.5669,
    -20.6897, 7.5661, -2.2925, -26.9559, 7.0617, 22.0854, -24.3694, 26.2958,
    -26.6219, -7.2876, 1.9526, 0.7003, 3.6442, -2.3814, -3.7036, -0.5418,
    28.3996, 3.7139, -5.0765, -9.5666, 16.6667, 0.8282, 25.1631, 2.7100,
    -2.1127, 10.1592, -6.6859, 2.3571, -1.1202, 23.6425, -4.4087, 6.5463
};

/**
 * Function that computes the Harris responses in a
 * blockSize x blockSize patch at given points in the image
 */
static void
HarrisResponses(const Mat& diff_x, const Mat& diff_y,
                std::vector<KeyPoint>& pts,
                Mat& response, int blockSize, float harris_k)
{
    size_t ptidx, ptsize = pts.size();

    const int* dx00 = diff_x.ptr<int>();
    const int* dy00 = diff_y.ptr<int>();
    int step = diff_x.step1();
    int r = blockSize/2;

    float scale = 1.f/(3 * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        float kp_x = pts[ptidx].pt.x;
        float kp_y = pts[ptidx].pt.y;
        int x0 = (int)kp_x;
        int y0 = (int)kp_y;

        const int* dx0 = dx00 + (y0 - r)*step + x0 - r;
        const int* dy0 = dy00 + (y0 - r)*step + x0 - r;
        float a = 0, b = 0, c = 0; // e is for debug

        for( int i = 0; i < blockSize; i++ )
        {
            for( int j = 0; j < blockSize; j++ )
            {
                const int* dx = dx0 + (i*step + j);
                const int* dy = dy0 + (i*step + j);
                int Ix = dx[-step] + 1*dx[0] + dx[step];
                int Iy = dy[-1] + 1*dy[0] + dy[1];
                a += (float)(Ix*Ix);
                b += (float)(Iy*Iy);
                c += (float)(Ix*Iy);
            }
        }
        pts[ptidx].response = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;

        response.at<float>(ptidx,0) = (float)a;
        response.at<float>(ptidx,1) = (float)b;
        response.at<float>(ptidx,2) = (float)c;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline float pick( const Mat& img, float x, float y, bool debug )
{
    int step = img.step; // The same as step1() for single channel image
    int x0 = (int)x;
    int y0 = (int)y;
    float xd = (x - (float)x0);
    float yd = (y - (float)y0);
    const uchar* ptr = img.ptr<uchar>() + y0*step + x0;
    return ptr[-1]*(1 - xd) + ptr[0] + ptr[1]*xd + ptr[-step]*(1-yd) + ptr[step]*yd;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

static void
computeSkew( const Mat& responses, Mat& skew)
{
    Mat corr(2, 2, CV_32F), eig_vec, eig_val;
    float* corr_p = corr.ptr<float>();
    for ( int i = 0; i < responses.rows; i++ )
    {
        // Declare pointers to the rows corresponding to keypoint `i`
        const float* resp_row = responses.ptr<float>(i);
        float* skew_row = skew.ptr<float>(i);
        //int skew_sgn = sgn(resp_row[3]);

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
        val_p[0] = (val_p[0] / val_sq);// * skew_sgn;
        val_p[1] = (val_p[1] / val_sq);// * skew_sgn;
        //val_p[0] = std::sqrt(val_p[0] / val_sq);// * skew_sgn;
        //val_p[1] = std::sqrt(val_p[1] / val_sq);// * skew_sgn;

        // Calculate transformation matrix based on the matrix multiplication
        // of skew `diag(eig_val)` and rotate [-1*vec[1] vec[0]; vec[0] vec[1]]
        skew_row[0] = vec_p[1]*val_p[0];
        skew_row[1] = -1*vec_p[0]*val_p[1];
        skew_row[2] = -1*vec_p[0]*val_p[0];
        skew_row[3] = -1*vec_p[1]*val_p[1];
    }
}

static void
computeDAFTDescriptors( const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
                       const std::vector<float>& layerScale, const Mat& harrisResponse, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, int dsize, float patchSize, float sizeScale)
{
    // Compute skew matrix for each keypoint
    int nkeypoints = (int)keypoints.size();
    Mat skew(harrisResponse.rows, 4, CV_32F), points_kp, s, img_roi;
    computeSkew(harrisResponse, skew);

    // Now for each keypoint, collect data for each point and construct the descriptor
    KeyPoint kp;
    for (int i = 0; i < nkeypoints; i++)
    {
        // Find image region of interest in image pyramid
        kp = keypoints[i];
        float x = kp.pt.x / layerScale[kp.octave];
        float y = kp.pt.y / layerScale[kp.octave];
        img_roi = imagePyramid(layerInfo[kp.octave]); // TODO: this can break for unsupported keypoints
        uchar* desc = descriptors.ptr<uchar>(i);

        const float* p = (const float*)points; // points are defined at line 57
        const float* s = skew.ptr<float>(i);

        float min_val = 9999, max_val = 0, picked = 0;
        unsigned int min_idx = 0, max_idx = 0, byte_val = 0, l = (unsigned int)dsize*8;
        for (unsigned int j = 0; j < l; j++)
        {
            float x0 = (p[2*j]*s[0] + p[2*j+1]*s[1])*sizeScale + x;
            float y0 = (p[2*j]*s[2] + p[2*j+1]*s[3])*sizeScale + y;

            picked = pick(img_roi, x0, y0, false);
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
                    desc[(j >> 3)] = (uchar)((byte_val << 4) + ((max_idx << 2) + (min_idx)));
                else
                    byte_val = ((max_idx << 2) + min_idx);
                min_val = 9999; max_val = 0; // Reset
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
    explicit DAFT_Impl(int _nfeatures, int _size, float _patchSize, float _scaleFactor, int _nlevels, int _edgeThreshold,
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

    void setPatchSize(float patchSize_) { patchSize = patchSize_; }
    float getPatchSize() const { return patchSize; }

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
    float patchSize;
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
static void computeKeyPoints(const Mat& imagePyramid,
                             const Mat& maskPyramid,
                             const Mat& diff_x, const Mat& diff_y,
                             const std::vector<Rect>& layerInfo,
                             const std::vector<float>& layerScale,
                             std::vector<KeyPoint>& allKeypoints,
                             Mat& response,
                             int nfeatures, double scaleFactor,
                             int edgeThreshold, float patchSize,
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
        Mat dx = diff_x(layerInfo[level]);
        Mat dy = diff_y(layerInfo[level]);
        Mat mask = maskPyramid.empty() ? Mat() : maskPyramid(layerInfo[level]);

        // Detect FAST features, 20 is a good threshold
        Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(fastThreshold, true);
        fd->detect(img, keypoints, mask);

        // Keep more points than necessary as FAST does not give amazing corners
        KeyPointsFilter::retainBest(keypoints, 2 * featuresNum);
        // Remove keypoints very close to the border
        KeyPointsFilter::runByImageBorder(keypoints, img.size(), edgeThreshold);

        // Filter remaining points based on their Harris Response
        HarrisResponses(dx, dy, keypoints, cur_response, 7, HARRIS_K);
        KeyPointsFilter::retainBest(keypoints, featuresNum);

        nkeypoints = (int)keypoints.size();
        Mat resp = response(Range(responseOffset,responseOffset+nkeypoints), Range::all());
        HarrisResponses(dx, dy, keypoints, resp, 15, HARRIS_K);
        for( i = 0; i < nkeypoints; i++ )
        {
            keypoints[i].octave = level;
            keypoints[i].size = patchSize*layerScale[level];
            keypoints[i].pt *= layerScale[level];
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

    //int border = std::max(edgeThreshold, patchSize);
    int border = 10;

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
    //int kp_x = 670, kp_y = 440;

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
    Mat harrisResponse(nfeatures, 3, CV_32F);
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
        computeKeyPoints(imagePyramid, maskPyramid, diff_x, diff_y,
                         layerInfo, layerScale, keypoints, harrisResponse,
                         nfeatures, scaleFactor, edgeThreshold, patchSize, fastThreshold);
    }
    else // supplied keypoints
    {
        KeyPointsFilter::runByImageBorder(keypoints, image.size(), edgeThreshold);
        HarrisResponses(diff_x(layerInfo[0]), diff_y(layerInfo[0]), keypoints, harrisResponse, 21, HARRIS_K);
    }

    if( do_descriptors )
    {
        for( level = 0; level < nLevels; level++ )
        {
            // preprocess the resized image
            Mat img = imagePyramid(layerInfo[level]);
            GaussianBlur(img, img, Size(7, 7), 2, 2, BORDER_REFLECT_101);
        }
        int dsize = descriptorSize();
        float sizeScale = patchSize / 29;
        nkeypoints = (int)keypoints.size();
        _descriptors.create(nkeypoints, dsize, CV_8U);
        Mat descriptors = _descriptors.getMat();
        computeDAFTDescriptors(imagePyramid, layerInfo, layerScale, harrisResponse,
                               keypoints, descriptors, dsize, patchSize, sizeScale);
    }
}

Ptr<DAFT> DAFT::create(int nfeatures, int size, float patchSize, float scaleFactor, int nlevels, int edgeThreshold,
         int fastThreshold)
{
    return makePtr<DAFT_Impl>(nfeatures, size, patchSize, scaleFactor, nlevels, edgeThreshold,
                              fastThreshold);
}

