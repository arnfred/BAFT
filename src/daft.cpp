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
    -21.4634, -1.0057, -1.1487, -11.8690, -1.3249, -2.3913, -13.5496, -9.1797,
    2.2054, -1.9503, 3.1662, -1.4500, 7.5864, -2.6167, -5.7485, -9.2780,
    2.1567, 26.8883, -1.2886, -2.2615, -4.3248, -1.5301, -6.9683, -40.3469,
    2.6023, 14.7169, -3.4652, 1.3070, -42.6123, -3.9935, -1.4048, 0.9430,
    3.0644, -0.9559, 2.1213, 15.7720, -10.5471, 1.9417, -5.8791, -7.4575,
    2.1562, 2.1877, -28.4774, -2.4734, 40.0124, -1.4300, -9.7054, 8.6736,
    -28.7606, 0.9388, -10.2151, -4.5940, 2.4351, 6.9999, -19.6847, -1.1539,
    7.3440, 6.3103, -40.8744, -3.5131, 2.5215, -4.2328, 5.5412, 1.7631,
    32.1780, -9.2951, 6.7123, -1.4161, 2.1187, 2.7653, -29.8870, 2.9510,
    -17.8105, 0.9539, 39.5864, 2.0723, 12.3777, 25.1262, -2.5759, -19.0141,
    -1.2374, -31.3440, -17.5264, -20.2228, -23.1351, 48.1974, 24.9852, -1.2054,
    -27.7071, 11.9766, 1.7425, -7.5245, -42.1561, 1.4160, 3.1205, 48.5143,
    2.3773, 8.2239, -12.6615, -1.4899, -14.2025, 12.7283, 1.9404, 15.8846,
    4.8981, -6.4594, -30.6646, 21.4445, 16.0063, 15.6095, 25.6073, 2.1960,
    -5.2631, 10.0128, 2.2585, -3.3111, -6.0647, 0.9938, 3.9698, -2.2096,
    11.9707, -2.8776, -29.4109, -2.5274, 43.4768, 18.9195, -31.0733, -38.9559,
    6.0219, -19.6455, -2.2552, -2.1577, 9.6544, -0.9867, -25.7481, -3.7087,
    5.1336, 25.7346, 18.3085, -2.0346, 1.5714, -8.3269, 10.7095, 12.3224,
    5.2074, 2.1995, -2.6605, -4.1192, -8.9270, -6.2800, -3.1197, 5.6977,
    36.6642, -7.7429, -5.4136, -1.1677, 5.7719, -13.7825, -1.8695, -7.7474,
    -2.7560, -45.7178, -39.8005, -40.6290, -18.0311, 1.8712, 3.5651, -3.9935,
    -1.1794, 3.2162, 3.4470, -1.6262, -1.2014, -38.8265, 2.8800, 21.9409,
    -12.8668, -1.8595, -27.2992, 43.7407, -15.2105, -0.9190, 15.3024, 1.7630,
    26.9097, 9.9364, -1.3039, 0.9354, -18.3619, -27.2582, -3.7736, -12.3408,
    2.6530, -1.8702, -1.2199, -48.6306, 3.3844, 17.8697, -27.5961, -7.7989,
    -37.6704, -3.8003, -49.8373, -1.0412, 1.8956, -9.3531, -0.9600, 4.6087,
    -14.5949, 35.8959, -0.9942, -5.8662, -19.2147, -1.8858, -0.9993, -1.0336,
    -42.1933, 8.0440, 20.1886, -27.4073, 11.2489, 2.1714, -7.4108, -20.2828,
    -1.8937, -16.2641, 1.3434, -36.0841, 17.8051, -0.9943, 26.4403, -2.0283,
    2.1552, 10.0920, -2.5819, 4.5358, 3.3353, -8.9846, 1.3983, -28.4361,
    -5.0033, -49.0011, 10.8583, 11.6544, -2.4753, 4.1652, -1.7478, 18.7984,
    19.5583, 1.5841, 2.7114, 20.7179, -1.1072, 5.3989, 6.7813, -3.8322,
    5.0586, -8.8668, 2.0036, -18.9820, 1.1338, 1.6459, -10.3619, 36.0802,
    -23.1035, 1.7749, 39.9215, -0.9459, 29.7154, 5.6072, 11.5484, 45.0075,
    13.9346, 24.0266, -7.8631, 2.3580, 6.1822, 4.9847, -2.0906, -1.4132,
    5.8020, 2.9407, -1.1924, -4.9672, 3.8664, -22.5549, 9.0918, -14.5889,
    18.2523, -8.6769, 4.0441, 23.3775, 9.4949, -11.1924, -0.9624, 3.9439,
    14.8682, 3.0767, -8.5100, -1.3944, 7.4073, -4.5458, -5.3410, 0.9904,
    42.0431, -41.2111, -46.6683, -2.3551, 2.0593, -2.2749, -7.4046, -22.8206,
    -4.6214, -4.6462, -16.4884, 13.4730, -3.2804, -46.8194, 9.5886, 2.3552,
    3.7403, -5.6093, -43.0578, 3.4530, -1.4278, 4.7731, 1.4874, -3.6124,
    1.7365, 4.4729, 2.8954, -2.1544, -3.9061, -9.7835, -32.6422, -1.5963,
    12.9881, -6.3641, -14.7028, -1.5267, 37.4530, 9.8925, 2.0810, 41.7311,
    23.3845, 7.2847, -46.4420, -16.2199, 0.9170, -6.8981, -10.9934, 25.9663,
    3.2981, -4.4283, -11.7616, -1.6418, 8.8566, 12.1747, -8.9770, 8.3512,
    -5.3721, -48.1586, 2.4311, -28.3435, -2.1113, 6.0788, 1.4889, -1.4567,
    39.8580, -6.6551, 5.1229, -26.5292, 5.4903, -16.9849, 20.6070, 5.3525,
    26.3860, -11.1709, -2.7432, 5.0444, 1.5307, -42.3654, 13.6318, 8.8764,
    -1.4784, 4.6908, -2.3232, 44.5129, 31.6960, -1.4417, 12.7307, -4.7554,
    43.8345, 4.6192, -15.9048, -36.7457, 14.3447, 37.5613, 12.0033, 19.5404,
    -2.0054, -2.2892, 3.3646, -5.7036, 14.0021, 3.1786, -1.1688, -2.1267,
    -49.1967, -1.1707, -1.8357, 1.8443, 1.8115, -1.7244, -27.8560, 5.2100,
    1.7184, 0.9338, 13.7880, 7.0273, -2.2478, 18.9701, -19.1702, -5.7333,
    19.0613, 34.5746, 2.7112, -4.9399, -11.0724, 26.4678, -1.1836, -23.2811,
    11.4202, 19.4139, 49.4140, -1.9108, -10.8314, 8.6634, 9.2423, 2.3677,
    2.5706, -1.4160, 14.2657, 0.9332, -1.5893, 27.2375, 2.6975, 2.6426,
    42.4533, -10.2301, 3.9860, -1.9620, 1.5345, -25.5935, 9.4122, -2.5934,
    12.4753, -1.0485, -4.0613, 19.3755, -3.7996, -6.2312, 12.4328, 10.4364,
    1.0098, 2.2716, -25.5452, 3.2616, -15.4659, -1.1248, -14.7341, -13.7386,
    -23.8943, 4.6207, -14.2026, -0.9830, -41.8042, -4.4519, 2.0584, 2.8917,
    -9.3314, 2.3756, 13.4146, 1.0530, 14.8546, -31.8275, 16.2064, 1.7812,
    12.9355, -1.1567, -10.6716, 6.0672, 2.8248, 3.5728, -3.3004, 5.6919,
    -2.3558, -17.1471, -3.4035, 14.8142, -1.4821, 2.2875, 1.7179, -14.3889,
    -49.7958, 2.9248, 2.2226, 0.9936, -1.3621, 1.0447, 9.5415, 5.6212,
    3.1789, 3.1231, 44.2680, 1.1141, -3.2505, -9.0149, 8.8880, -10.5050,
    10.9358, -26.4054, 2.0278, 2.5285, -1.5502, 5.5631, 5.4949, -24.8097,
    49.9013, 1.5379, 11.5213, -2.0561, 14.7784, 5.6870, 4.0140, 1.3456,
    7.0616, 2.9207, 42.6522, 1.2895, 4.2773, 8.3811, -1.5549, -6.4683,
    -32.1964, -5.8356, -15.6901, -20.1607, 34.9202, -14.5274, 1.3591, -8.5867,
    1.1291, -16.9190, 9.8373, 4.1809, -8.6097, 19.4592, 9.2249, 2.9425,
    29.5038, -22.2920, -40.5504, 3.3382, 1.2986, 1.6882, -34.4610, -1.9425,
    -7.8079, 7.8297, 15.8082, -20.6447, -12.9067, -1.8903, 38.1394, 9.1238,
    10.2253, -3.6930, -20.6668, -5.9459, -6.3877, -11.8287, 6.6541, 1.1670,
    2.7348, -3.0398, 1.1147, 8.0694, -48.0193, -26.6428, 0.9644, 1.0451,
    26.1116, 12.5137, -1.5199, 27.9555, 1.3519, 28.0745, 28.3184, 14.2754,
    -2.8472, -20.8997, -2.0455, 29.1257, 17.5683, -35.4112, 16.7344, 9.6308,
    1.3563, -11.9848, -2.8901, 13.7167, -2.8192, -1.8910, -7.6516, 1.1218,
    1.3477, -9.9587, -33.8900, 5.1227, -15.6236, -11.6647, 17.6413, -25.8368,
    35.5654, 1.9650, -36.5844, 47.4385, 5.6913, 3.1298, -19.9613, -9.5900,
    -16.4549, -22.9847, 6.0891, 7.6657, 1.0792, -42.6269, 1.9430, 47.2696,
    26.2177, 5.3177, 20.3629, 3.9385, 8.9299, 9.9966, 10.9301, -10.2753,
    -3.1113, 1.6726, -25.7738, -1.2430, 2.7054, -4.5216, 1.0646, -3.5306,
    -48.1389, 1.9771, -13.1827, -8.3197, -7.7355, 40.5465, 9.0946, -4.9212,
    8.4821, 5.4456, 1.3454, -2.8933, -37.5129, 13.7015, 41.5638, -1.2044,
    -20.9524, -29.5355, -24.7845, -49.3837, 1.2823, 1.6566, -41.3523, 4.3961,
    2.6546, 13.9652, -7.6640, 8.5055, -4.0038, 3.5441, 1.0994, 8.2825,
    21.9622, 6.8730, 2.6741, 21.4205, -1.0075, 2.0048, 15.0236, -11.0526,
    1.6932, -8.6794, -2.7515, 4.9219, -37.9452, 1.2170, -0.9969, -3.4436,
    -28.1159, 4.7564, 2.2937, -2.1440, -25.3606, 25.1974, 9.9679, -24.6541,
    -0.9326, -1.9565, 23.1158, -15.8928, -7.3779, -1.8507, -4.3662, 36.6843,
    -2.8778, 32.8721, -21.8479, 4.6882, 38.2552, 2.2656, 45.5484, -3.5712,
    1.9347, -47.3564, 2.1749, -28.1974, -1.3806, 1.2756, -2.4102, 4.4187,
    -3.7499, 1.1261, -0.9219, 3.2067, 1.2049, 1.4001, -2.0620, -1.1094,
    22.3142, -45.2030, 18.1892, 9.4215, 11.8469, -19.6397, -12.8144, 1.0118,
    -9.9284, 2.4872, -1.4118, 1.6390, 8.9661, -2.7112, 2.0358, 9.4664,
    -2.9630, -1.0549, 38.3744, 3.4375, -2.8632, -2.2985, -17.0987, 3.1810,
    -1.2031, -46.9107, 14.8153, 15.2390, 39.2046, -17.4132, 3.9154, -8.5331,
    8.4236, -5.4738, -10.9572, -15.7593, 1.6786, -19.6807, -15.6253, -0.9791,
    20.1195, -8.2951, 27.7766, 7.3846, -1.9001, -3.0834, -20.9530, 16.8922,
    24.7934, -6.0037, 5.0803, 4.2668, 1.1519, -48.5194, 32.6530, -16.8617,
    4.0276, -7.5880, 0.9377, 7.1279, -18.1519, 22.1418, -16.0696, -5.7711,
    11.4275, -5.8728, -0.9398, -1.6852, 1.2382, -15.3285, -3.5791, -4.3195,
    -3.5895, 11.2038, -2.7821, 3.8072, 11.0306, 7.2156, -18.3541, -39.5882,
    8.4891, 1.8886, 1.2269, -8.2774, 4.7387, 24.0433, -1.7807, 3.6377,
    -2.1077, 6.6501, 5.1562, 19.4755, 11.9841, 7.8227, -2.9071, -5.8268,
    -4.5627, -2.2465, -19.0012, -1.8983, -4.6624, 16.6643, -11.2104, -49.2616,
    -47.2539, 35.8940, -10.5318, 5.3692, 17.2299, 1.8150, -3.2817, -1.4141,
    -8.0706, -6.3055, 9.0400, -3.9047, -2.3911, -1.5692, -3.4523, -15.0418,
    -1.6273, 7.8740, 2.0504, -6.0808, -6.4641, 12.9058, 0.9970, -3.1321,
    -2.8059, 6.3270, -4.1607, -27.8039, 3.2713, -48.0416, -1.3461, 33.4601,
    21.4661, 2.8696, -31.7925, 3.4769, -9.3866, -20.5938, 37.7286, -2.5979,
    19.2926, -16.7324, 7.3757, -3.6083, -9.6580, 6.2328, 36.9339, 0.9661,
    -10.1670, -7.0395, 4.4922, -15.9627, -45.5748, -4.5222, -5.6684, 24.7749,
    5.3424, -7.0554, -0.9639, -49.9542, 27.6188, -4.0359, -1.2915, -41.7873,
    2.5451, 3.4978, -2.5247, -15.1952, -8.4746, -21.3617, 3.1905, -45.7902,
    1.2955, 10.9998, 18.9086, -15.1520, -18.4854, -3.4649, -7.3241, -5.5211,
    -11.5499, 47.4641, 43.0042, -2.5661, -2.7018, -4.3996, -3.0968, 21.6775,
    -13.0410, 3.5824, 41.8088, -1.2213, 1.1846, 3.7092, -1.3377, -7.5629,
    13.8622, -5.2301, -1.9808, 14.9997, -2.7334, -24.9787, 2.3519, -14.2645,
    -9.0575, 6.9718, 30.2342, 2.1492, -1.2283, -1.2408, -5.6623, -5.0959,
    15.6768, -47.5149, -2.8169, 16.4878, 12.6545, -1.7409, -26.2240, -3.3563,
    0.9769, -37.6986, -1.6022, -3.2648, 11.6915, -25.4661, 1.7663, -41.8856,
    -5.0581, 4.0689, 45.5890, 20.1530, 2.3300, -42.7428, -16.8224, 11.3059,
    9.6772, 36.2662, -14.7972, 2.4136, -14.2953, 2.3138, -1.9435, 1.4319,
    -2.3759, 5.9736, -7.1608, 25.0397, -6.9094, 6.5770, -5.2292, -1.3631,
    -9.3149, 6.5591, 1.9263, -15.4522, 1.1427, 4.9009, -3.6909, 12.9628,
    17.7289, -43.0044, -40.6048, 25.8877, -5.2548, -1.1574, -11.6266, -37.0882,
    -2.7308, -45.5225, 39.5359, -3.4103, 9.7857, 37.4708, 2.1236, 9.1117,
    -8.0260, -6.0021, -1.6239, -14.8671, -43.3625, -24.1641, -1.2217, 4.0641,
    -5.6955, 2.1714, -11.3932, 28.5035, -7.1325, 35.0524, -10.1559, -9.0617,
    25.9226, 11.7760, 3.9806, 9.2330, 29.0481, -10.2080, 2.9094, -17.9926,
    11.0936, 4.7962, 10.6294, -6.6353, 6.1523, 11.6667, -3.6868, 47.3173,
    47.8165, -2.7237, 8.7699, -23.2994, -5.3570, -30.4103, 1.6274, 1.2063,
    1.7875, 13.7378, 23.9176, -1.8890, -20.0567, 24.6471, 28.6235, 24.6938,
    -1.1242, -6.6049, 19.4692, -10.7454, -10.9565, -36.4861, 4.3480, -8.6002,
    16.9862, 9.1175, -1.5625, 7.1123, 34.2081, 19.4501, 12.9585, -8.5064,
    0.9983, -5.0063, 27.4241, 4.3886, -36.2772, -31.5269, 37.2271, -2.7056,
    4.2029, 14.1483, -7.4851, 6.4267, -18.2518, -5.0411, 3.5062, -6.2740,
    -0.9474, -2.6807, -2.0406, 5.8956, 35.0286, 16.6358, -1.1769, -3.8875,
    -48.5058, 2.0405, 1.4420, 1.9838, -9.7249, 11.0406, 4.9224, 6.8611,
    1.1109, 3.7847, 1.0719, -17.2202, 1.9775, 19.5803, 1.8748, -24.9196,
    3.7881, 3.7706, -2.9588, 30.8874, -17.8229, -21.2280, -4.0335, -2.1306,
    1.6081, -4.4179, 1.2064, -1.3375, -35.5541, -6.2310, -5.0203, -13.6178,
    9.7370, -23.7811, 34.6115, 1.1146, 10.6118, -6.8279, -18.2252, -48.9559,
    -36.2802, -0.9261, -16.0325, 1.5115, 25.3203, -6.4425, 2.4864, 1.8591,
    -35.7173, -27.6222, -16.6373, 12.6206, -4.6723, 5.0078, -2.3689, 17.4897,
    40.4439, 10.7859, -13.3811, 1.8920, 1.3734, -7.3551, 5.1764, -10.5536,
    -1.0004, -2.5212, 0.9516, 8.4903, 13.9332, 2.1122, 7.7542, 8.5778,
    5.5731, -17.3383, -9.5370, -1.8167, 26.2821, -37.4629, -3.7787, 44.2156,
    2.0593, 46.4250, -12.6588, -0.9890, -18.0641, 22.9199, 13.7296, -2.3441,
    -28.9998, -28.5385, -3.7266, -1.0250, 4.1861, -9.1189, 2.8426, 4.7053,
    2.9526, 43.1981, 1.1387, -4.5942, -1.1364, -9.7143, -48.6409, -24.9384,
    -17.4505, 2.0841, 15.5039, -2.5802, 4.5709, -1.3637, -29.0877, -47.1835,
    -31.3955, 10.2590, -28.0134, -13.1404, 3.5317, -34.8654, 4.4605, -7.8628,
    23.6831, -10.5393, 32.3541, -1.3761, -10.7825, -1.6416, -37.3159, 1.3234,
    15.0768, 38.2002, -2.3851, -15.0744, -23.9861, 2.3218, -1.1906, -5.8877,
    1.5735, 7.1591, 1.2678, 3.4463, -33.6356, 36.1896, 37.6398, 1.2661,
    -21.1492, -37.6220, -1.2112, -2.9646, 1.2260, -18.5758, 19.2649, -37.9386,
    -12.5679, 2.7555, -39.0252, 12.8908, -6.1109, 19.2016, 1.1325, -3.1658,
    9.6808, -1.4801, 37.2482, -7.9198, -0.9829, 17.6365, 13.5451, 4.9857,
    -1.1388, 4.1213, 15.5307, 18.5433, 7.3168, -7.7401, -44.0738, 0.9944,
    -12.5323, -3.4699, -8.7720, -38.6054, -24.6707, -0.9554, -9.3424, -7.0527,
    6.7425, 2.8769, 21.7749, 5.9802, -1.7310, -4.0221, -43.8372, 3.6824,
    12.1462, -20.6661, 3.1543, 2.2520, 2.4555, 18.2366, 22.7160, -2.4539,
    36.2322, -4.2398, 1.8406, -3.4788, -9.6009, 18.8471, -4.1254, -1.0026,
    3.9781, 4.6142, -4.2262, -30.4011, -3.1236, -4.0425, 34.4456, 26.9598,
    29.1794, -1.1350, 31.3453, -26.7304, -1.3642, 1.8820, -5.6198, 4.1193,
    33.5205, -4.1485, -6.8084, 39.1505, -20.9214, 13.6335, 34.1596, 4.5739,
    2.7309, -15.8393, 39.8960, -4.3419, -13.9418, -46.3633, -43.9262, -8.1624,
    25.4189, -3.6541, -27.4236, -46.8671, 1.8720, 8.5993, 1.5920, 37.2906,
    1.4464, 34.2673, 22.2561, -2.8518, 2.4896, -0.9503, 7.9778, -22.5340,
    -47.5263, 5.1899, 2.2243, 27.4695, -19.0310, 5.1659, -3.7964, 1.1160,
    14.0223, -5.5132, -34.0435, 1.4295, -38.2356, -48.6756, -22.4428, -1.0487,
    3.0475, 1.6482, -32.0497, -8.5896, 5.8800, -4.2173, -1.9135, -1.5529,
    -3.9428, 2.4310, -1.5421, 42.8890, -20.9680, -9.3296, -26.8920, -29.3249,
    -0.9333, 30.1986, 26.0358, -1.6986, 9.1418, 1.3810, 3.2670, 8.8969,
    48.5981, 1.1773, 4.6932, 24.4932, 1.0947, -4.2662, 48.9170, -44.8977,
    1.5252, -1.2453, 5.6671, -1.7818, -9.3185, -41.5907, 1.3538, 17.1752,
    0.9842, -5.0022, -11.7176, 1.5658, -1.6364, -48.7072, 19.5090, 7.8993,
    -1.1536, 3.4626, -1.7442, 7.6800, -4.9342, 22.9813, -1.6352, 29.9031,
    0.9225, 4.3670, 1.7208, -8.1831, 39.8655, 9.6604, -17.3743, -18.4090,
    15.1595, -4.8744, -3.1580, 37.1176, 35.3226, 3.0306, -4.6813, -6.8796,
    -1.0772, -3.0031, -5.0175, 31.0137, 5.4979, 6.0609, 1.4792, 1.0758,
    -11.0663, -6.2950, -13.9325, 25.5264, 1.0622, 1.4587, -26.0357, -7.9030,
    24.0868, 1.5988, 23.3906, -33.4556, -28.8383, -24.9200, 5.7476, 24.8684,
    20.1756, 6.6976, 5.0865, -22.5082, -22.6445, -26.7094, -2.8839, 15.8263,
    -43.6000, -29.4829, 6.3848, -1.3340, 13.2474, 6.4168, -36.0757, 2.0240,
    -1.5522, -1.8348, 1.0389, -14.2969, 38.7808, 34.7243, 8.5302, -11.2195,
    -42.0090, -28.5978, 40.1886, -1.2133, -14.0918, -3.1650, -1.0658, 17.5271,
    5.6301, -1.8834, -4.8845, -47.3869, -9.3355, -3.3688, 25.9325, 11.0846,
    -4.6295, -14.7034, 2.3421, 4.1222, 2.1800, -31.1280, -8.9449, 3.7733,
    8.2333, -2.2765, 9.6543, 28.5204, 2.3255, -1.2339, 1.2263, 1.7022,
    21.3738, -4.8139, -48.7386, -3.5719, 3.3787, -36.5731, 3.1921, 44.7112,
    2.4545, 2.3931, 26.5281, 47.7043, 33.2119, 4.2899, -1.8026, 4.7273,
    9.4494, 31.2414, -20.6479, -34.9671, -1.2603, 8.0215, 6.6263, 1.6893,
    10.0615, -12.8409, -11.0755, 6.9037, 6.8950, -9.7490, -31.6464, 10.8778,
    5.3988, -3.9930, 1.3480, 4.3416, 1.5335, -6.6070, 1.1461, 17.1223,
    -12.6689, -23.0929, -2.5056, 4.5814, -2.0805, -1.0038, 18.7031, -1.8473,
    -2.1904, -23.2791, -2.3275, 12.2513, -10.8393, -4.4455, -3.1150, -1.8544,
    -1.3295, 8.7500, 7.7303, 9.2700, 35.3128, -47.2592, -47.4215, -8.4214,
    0.9737, 2.6479, -12.4223, 8.5885, -26.1729, 7.1715, -17.6474, -0.9853,
    -5.8132, 5.7058, 7.7730, -25.0822, -4.6128, -0.9864, 45.2363, -16.1293,
    28.1634, 1.0249, 17.3188, 1.3013, 37.4728, -3.7978, -2.1523, -0.9974,
    1.6254, -5.4366, 0.9553, -18.2909, 2.0317, -28.7341, -11.4978, -35.9206,
    -1.1799, 1.7093, 4.7020, -6.2640, 11.4977, 3.0406, -2.1183, -2.9236,
    2.5510, -4.7739, 0.9452, -6.1075, 4.3234, 2.4037, 21.3957, 4.0702,
    -1.0209, -27.5438, -1.6576, -1.0213, 32.2058, -1.1906, -2.0928, 18.4961,
    28.1425, -15.7905, -21.1641, 33.3190, -39.4393, -47.7486, -6.4539, 5.1333,
    0.9580, 1.5544, 3.3970, 25.2823, 13.9287, 29.6906, -1.5429, -2.6750,
    1.4195, 13.5111, -2.4043, 20.1053, 18.9567, 3.9918, 9.5817, 21.4415,
    6.2636, 3.4966, 21.1712, -8.5267, -1.1003, -11.9529, 17.1972, 4.8080,
    -9.9395, 1.5535, -6.6243, -4.4878, -15.4512, -12.1926, 10.1549, -4.7190,
    39.3590, 1.3203, -2.6292, -10.0875, 1.2905, 34.0447, 9.9898, -1.3508,
    10.7793, -7.5986, -10.3384, -25.8492, -14.8410, -4.4564, -46.4435, 1.6566,
    -28.6001, 5.5057, 1.5668, -48.4648, 5.0480, 1.4421, 10.0380, 5.0619,
    45.3047, -49.5958, 17.0814, 8.6259, 21.9478, 37.1654, -2.0422, -31.1235,
    -0.9378, 3.7049, 12.0814, 5.6985, -36.2960, -3.0950, -2.5039, -2.0121,
    -12.6720, 4.7809, 12.9712, -7.7689, 19.5181, 21.0608, -12.4102, 5.3703,
    -1.0865, -11.8696, 3.8715, -42.5554, -6.9651, -1.5950, 6.0712, 4.4677,
    26.6889, -11.7123, -23.0569, 1.0306, 33.3479, 23.9119, -1.0645, 0.9394,
    5.2187, -2.7419, -14.3495, -4.2488, -4.7848, -1.2743, -44.1323, -9.8778,
    18.5491, -4.7358, -16.8618, 39.3246, 1.6600, 1.9242, -26.1493, -1.5381,
    -2.9611, 20.6005, 1.6239, -2.5835, 17.2601, 6.1978, 30.8914, -7.2841,
    -10.6001, 4.1026, -1.0072, -41.8379, -3.7960, 1.1343, 3.5272, 7.3165,
    -0.9314, 2.1642, -3.2509, 4.2359, 3.3098, 1.2641, 30.7241, -5.6116,
    25.3424, 0.9256, 46.8250, -8.2270, 1.9223, -2.2195, -1.2277, -6.2197,
    -18.1775, 4.7487, 2.7876, 1.4096, -10.1615, -13.3282, 3.0603, -23.6401,
    -13.3077, -1.2126, -16.2358, 2.2334, -7.5883, 28.5211, -25.0108, 1.6014,
    29.8974, -40.7513, 19.3784, 13.9183, 5.7478, 18.6814, 1.0599, 27.2923,
    -0.9403, 17.5195, -10.3028, 7.5983, 22.8630, -3.1710, 40.6476, -4.0738,
    5.7670, -1.0510, 8.5572, -2.4325, 3.4872, 0.9366, 35.9498, 41.6793,
    9.4913, -11.3258, 1.2089, -1.1168, -11.4981, 18.0455, 47.6192, -2.2364,
    -2.4240, 3.2350, -17.4038, 27.2642, 3.5714, -1.9971, 34.1632, 2.1104,
    -1.1455, -1.4392, 1.0067, -30.0058, -6.5761, -11.0497, 3.2735, -38.2721,
    8.3025, 2.2369, -6.0087, -8.4363, -34.5960, -5.0994, 3.6923, 3.3782,
    -1.7392, 12.1317, -8.7394, -21.1597, -8.6527, -2.2769, -1.0293, 4.1524,
    -13.1968, -28.6353, -16.5547, -34.9639, -3.4513, -1.9851, -1.1408, 1.6477,
    24.2376, -2.8205, 6.6531, 1.5490, 2.2323, 25.0042, -2.8047, -5.1252,
    -31.2146, 31.8303, 5.2545, 5.9421, 2.9346, -1.2546, -47.9414, 2.9327,
    -4.5856, -7.5506, 21.4538, 18.4177, 45.1411, -4.2250, 33.5513, 2.8318,
    -18.2809, 16.5276, 3.5718, 39.7935, -14.2658, 3.8987, -1.6764, -3.7105,
    -2.0071, -5.4497, -3.0463, -3.7301, 43.2430, -5.9694, 3.2357, 42.6574,
    1.9358, 18.8412, 5.3907, 25.1344, 24.6932, -1.5429, 20.0910, -16.3627,
    -32.4954, 10.1594, -5.1965, 11.5475, 35.6984, 31.8866, -15.6618, -28.8143,
    -1.8143, 1.3931, -6.2320, 16.3477, 3.5288, -30.2517, -22.2355, -2.0977,
    2.4107, 11.9762, -35.6476, 4.4425, 3.1097, -4.5353, 47.8218, 24.6213,
    1.5568, -5.7098, -1.9708, 16.8310, 1.9027, -6.6509, 1.6330, 4.5488,
    10.0139, -4.8824, -1.2961, 1.2657, -30.6428, 10.6405, -43.2143, 9.5094,
    -1.3016, 36.9896, 6.7783, -35.7406, -1.4190, 13.3688, -1.5853, -1.6295,
    8.1053, 6.7379, 13.9430, -42.7993, -6.2842, 12.8694, 1.4998, 2.5566,
    -4.4321, 42.7580, -1.6871, -1.0861, -34.1691, 27.4997, -35.3054, -32.1620,
    4.3299, 0.9198, -49.6714, 31.1939, -3.9647, 27.8981, -4.4351, 19.4622,
    -34.5313, -8.8961, 1.5933, 2.1647, 0.9365, -25.4961, -2.0500, -38.4884,
    2.0864, -3.1729, -1.6488, 14.1618, -6.1784, -6.2682, 5.5980, 24.8509,
    8.8402, 1.4538, 3.3368, 16.8313, 39.7790, 1.2530, 36.6273, 17.3275,
    0.9593, 0.9886, 21.3898, -9.1695, -1.5811, -19.5814, -16.4721, -6.8891
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
computeSkew( const Mat& responses, Mat& skew, int nkeypoints)
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

        // Normalize eigen values
        const float val_sq = std::sqrt(val_p[0]*val_p[1]);
        val_p[0] = val_p[0] / val_sq * skew_sgn;
        val_p[1] = val_p[1] / val_sq * skew_sgn;
        //val_p[0] = std::sqrt(val_p[0] / val_sq);// * skew_sgn;
        //val_p[1] = std::sqrt(val_p[1] / val_sq);// * skew_sgn;

        // Calculate transformation matrix based on the matrix multiplication
        // of skew `diag(eig_val)` and rotate [-1*vec[1] vec[0]; vec[0] vec[1]]
        skew_row[0] = -1*vec_p[1]*val_p[0];
        skew_row[1] = vec_p[0]*val_p[1];
        skew_row[2] = vec_p[0]*val_p[0];
        skew_row[3] = vec_p[1]*val_p[1];
    }
}

static void
computeDAFTDescriptors( const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
                       const std::vector<float>& layerScale, const Mat& harrisResponse, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, int dsize, int patchSize)
{
    // Compute skew matrix for each keypoint
    int nkeypoints = (int)keypoints.size();
    Mat skew(nkeypoints, 4, CV_32F), points_kp, s, img_roi;
    computeSkew(harrisResponse, skew, nkeypoints);

    // Now for each keypoint, collect data for each point and construct the descriptor
    KeyPoint kp;
    for (int i = 0; i < nkeypoints; i++)
    {
        // Find image region of interest in image pyramid
        kp = keypoints[i];
        //float scale = 1.f / layerScale[kp.octave];
        // TODO: Instead of scaling here, we might as well scale when generating the points
        //float scale = layerScale[kp.octave] * 0.57554;
        //float x = kp.pt.x;
        //float y = kp.pt.y;
        float scale = 0.57554*(19.f/29.f);
        float x = kp.pt.x / layerScale[kp.octave];
        float y = kp.pt.y / layerScale[kp.octave];
        img_roi = imagePyramid(layerInfo[kp.octave]); // TODO: this can break for unsupported keypoints
        //img_roi = imagePyramid(layerInfo[0]);
        //int step = img_roi.step;
        uchar* desc = descriptors.ptr<uchar>(i);
        const float* p = (const float*)points; // points are defined at line 57
        const float* s = skew.ptr<float>(i);

        float min_val = 9999, max_val = 0, picked = 0;
        unsigned int min_idx = 0, max_idx = 0, byte_val = 0, l = (unsigned int)dsize*8;
        for (unsigned int j = 0; j < l; j++)
        {
            float x0 = (p[2*j]*s[0] + p[2*j+1]*s[1])*scale + x;
            float y0 = (p[2*j]*s[2] + p[2*j+1]*s[3])*scale + y;

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
static void computeKeyPoints(const Mat& imagePyramid,
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


        nkeypoints = (int)keypoints.size();
        //int index;

        HarrisResponses(img, dx, dy, keypoints, cur_response, 15, HARRIS_K);
        //HarrisResponses(img_0, dx_0, dy_0, keypoints, cur_response, 21, HARRIS_K);
        for( i = 0; i < nkeypoints; i++ )
        {
            keypoints[i].octave = level;
            keypoints[i].size = patchSize*layerScale[level];
            keypoints[i].pt *= layerScale[level];
            float* response_row = response.ptr<float>(i + responseOffset);
            float* cur_row = cur_response.ptr<float>(i);
            for ( int j = 0; j < 5; j++ )
                response_row[j] = cur_row[j];
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
        computeKeyPoints(imagePyramid, maskPyramid, diff_x, diff_y,
                         layerInfo, layerScale, keypoints, harrisResponse,
                         nfeatures, scaleFactor, edgeThreshold, patchSize, fastThreshold);
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
                               keypoints, descriptors, dsize, patchSize);
    }
}

Ptr<DAFT> DAFT::create(int nfeatures, int size, int patchSize, float scaleFactor, int nlevels, int edgeThreshold,
         int fastThreshold)
{
    return makePtr<DAFT_Impl>(nfeatures, size, patchSize, scaleFactor, nlevels, edgeThreshold,
                              fastThreshold);
}

