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
/** Modified to create BAFT by: Jonas Arnfred */

#include "baft.h"

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

static float points[256*4*2] =
{
    6.0285, 2.2085, 10.0159, -2.8512, -13.1761, -0.6847, -16.1396, -10.1978,
    1.0495, 2.9465, -16.7350, 10.7787, 4.2515, -15.0652, -0.6586, -0.6450,
    0.3825, 3.7047, -0.3398, 17.2410, 2.4161, -2.0582, 0.5195, 8.0279,
    -0.8574, 2.5954, 4.1138, 5.6176, 0.4359, 1.4121, -0.5483, 10.1345,
    -6.5425, -13.0005, -3.3931, 6.2080, -0.6888, -25.6323, 0.6131, -0.9757,
    -4.9930, -8.4040, -0.9469, -0.5876, -16.8552, 0.4196, 10.0416, -1.8751,
    7.9727, -0.9710, -2.3309, 3.4518, 2.3395, 2.2393, -21.7321, 1.1713,
    -0.6777, -10.9243, -33.3191, 41.6351, -0.4547, 0.3382, -0.4783, -4.6219,
    -9.7217, -9.0935, 2.5604, 0.5134, -6.4791, 0.4543, 3.1947, 36.8382,
    3.4170, -1.3606, -2.2384, 0.4843, -35.2887, -33.1100, 35.6710, 0.5223,
    0.7073, 13.8159, -1.0544, -9.4762, -39.1406, 34.1536, -7.2981, -30.7687,
    -5.1808, -1.2839, 0.6585, 3.5257, -6.7681, 0.6003, 1.0615, -3.3700,
    0.9211, -2.9149, -1.4757, -3.2818, 0.6090, 21.0196, 28.1988, -1.3620,
    0.3489, 5.3241, -8.2435, 1.6159, -0.5425, 29.0155, 1.5422, -10.2814,
    -27.5204, -0.6816, 10.0116, 0.3432, -37.3453, -2.2241, 0.4399, -16.2387,
    0.5416, -46.6390, -3.3076, -0.4185, -1.0276, -4.9050, -0.8408, -0.6520,
    1.4572, -12.1775, -0.4016, 2.6388, 0.3658, 8.4100, 30.7274, 0.7250,
    -6.8093, -0.4332, 8.9396, 3.1327, -21.5531, -6.9479, -0.5160, -2.8134,
    -3.4216, -2.5326, 1.7949, 3.3127, 36.1550, -3.5870, 0.5129, -10.5408,
    -3.7900, 0.3617, 38.8079, 6.2441, 2.4938, 11.1853, -1.3834, 1.1631,
    23.5146, 0.7309, -39.3609, 0.6440, 46.0235, 13.6425, -38.0217, -1.5136,
    10.8492, 8.0276, -0.6005, -3.5765, 1.2153, -11.8399, -20.7614, -9.3544,
    -0.9954, -16.8310, -0.9675, 7.5113, -0.5852, 4.2701, -36.6012, 0.4186,
    -1.8456, 4.5554, 29.6966, 1.8054, 0.3810, 7.1495, 1.5441, -1.0106,
    -0.4419, -3.3360, 0.6053, -0.4849, 1.0892, -7.1051, 38.6277, 0.9361,
    6.4723, -8.2994, 0.5750, -16.9661, -26.0766, -0.4030, 9.8910, -25.3365,
    -21.9027, -21.4977, 15.1282, 49.9732, 1.4929, -13.5759, -8.9721, -2.5525,
    2.7447, 12.3219, -5.5902, 32.4165, 0.8068, 41.6139, 2.0080, -0.3455,
    -0.5961, -27.9562, 2.8045, 3.5674, -2.2076, 0.7092, -3.3018, -0.4392,
    46.5873, 42.7010, 1.7967, -18.4628, -3.6154, 29.7799, 3.8322, 7.1134,
    4.9760, -19.5761, -1.8923, 39.9259, 2.0121, -8.2288, 25.7789, 3.5923,
    0.7025, 1.7178, 0.9298, 23.4181, -0.5762, 0.4496, -0.5010, 4.9991,
    39.6891, -2.9961, 0.4382, 0.3623, 1.9070, -22.3773, 1.3243, -1.7341,
    15.6212, -19.3525, -0.5865, -2.8977, 24.2814, 2.4135, -18.3051, -0.4129,
    25.4177, -0.3713, 0.4972, 8.1360, 2.9686, -1.5303, 1.1765, 0.4592,
    0.4375, 7.0712, 0.8081, 2.7964, -15.2715, -41.5890, 0.4462, 15.9753,
    2.9226, -0.8972, -15.5852, 0.7255, -0.7854, -9.5269, 3.1376, 1.0628,
    7.4218, -34.5760, -3.8489, 46.1877, -9.5879, -12.3585, -15.0466, 7.4483,
    -18.3729, -49.9051, -6.2248, 5.4696, -6.5605, 2.2517, -16.9183, -39.5179,
    -0.6387, -18.3028, -3.3329, -18.5169, 0.9110, -5.4210, 1.6900, 12.3477,
    14.1827, -5.2086, -10.5499, -47.1335, -25.1787, -11.8245, 1.4878, 22.5404,
    28.3363, -11.3616, 10.7344, -5.6273, -4.2505, -7.8662, -10.1299, 38.8976,
    -0.3585, -0.3854, -0.6360, -0.9701, -3.3776, -4.8861, -44.1708, 0.7659,
    -3.6267, 47.0414, -24.0039, 19.8477, 1.3197, 26.5774, 9.1144, -0.4446,
    21.3098, -0.5392, -3.3028, -2.1305, -3.1193, -0.5199, 0.3744, -0.6055,
    11.4271, 0.4328, 3.0376, 35.4604, 15.7712, 18.2627, 2.1805, 0.5806,
    -0.6552, -11.5556, 0.4411, -4.0191, -4.2730, -0.7560, 23.1441, -2.4031,
    16.4466, -10.0904, 1.9717, -7.6162, 0.3914, -0.7339, -7.6039, 14.8897,
    3.4503, 42.7269, 0.4877, -40.4607, -26.4044, -2.2900, -1.2153, -0.7467,
    1.6543, 3.2242, -1.8358, -10.3425, -1.2958, 6.7208, 10.1551, 14.7911,
    31.1406, -1.5593, -0.9191, 0.3828, 3.0670, -0.6989, 20.3100, -18.8035,
    -4.4850, 0.5416, -22.8721, 5.2298, 7.2783, -2.2542, 4.3401, -21.2189,
    1.9366, -0.5959, -8.0730, 7.0131, -0.8158, -0.8617, 3.2526, -3.5601,
    4.8400, 3.1834, 24.1490, -30.8733, -8.1493, 13.3081, 0.7812, 2.8354,
    -6.6793, 7.7857, 15.6835, -1.1652, 14.0575, -14.7646, 5.1693, 3.9718,
    27.2496, -2.9104, -28.0596, -45.1372, 0.7136, -9.6272, -29.2606, 0.6417,
    -1.3767, 1.3706, -15.6963, -17.3049, -2.9812, 2.7123, 0.6035, 0.5564,
    15.1331, 35.6950, -10.8262, -10.6810, 3.0107, 0.4278, 9.0300, -44.8796,
    7.8845, 0.4242, -2.8581, 19.1926, 3.6007, 0.7727, -6.1646, 2.7534,
    -49.3488, -5.3587, 7.4380, 0.9574, 0.8837, -9.4902, 8.4018, -4.7699,
    1.0110, 0.4002, -0.6125, -11.0070, -48.0223, -1.7275, -20.2314, 0.8553,
    -5.7880, -42.5587, 6.1031, -6.8766, -2.0079, -0.5599, -7.2157, 0.6723,
    -6.2245, 9.3947, 14.3985, -1.8950, 41.7209, 4.2084, -0.7328, -0.7872,
    -29.8191, -0.3928, -41.0322, 25.5607, 11.7223, -16.7228, -6.3219, -14.5516,
    -5.9628, -0.4899, 0.3719, -0.7701, -2.4158, -21.2868, 0.5890, 5.9902,
    -0.6227, -2.0600, -6.8685, -1.2233, 1.7899, -0.3974, 2.3429, -5.7017,
    0.4284, 11.7195, -0.4567, -28.1294, -34.5054, 36.8975, 1.8926, 1.2433,
    -2.0929, 4.9869, -0.7455, -7.0432, -0.6402, -3.0426, 0.3484, -0.4188,
    -0.5101, -3.2095, 3.4275, -3.6912, 0.8890, -1.3613, 0.5980, -24.1569,
    -0.8571, 11.0123, 1.8574, 3.4902, -4.2291, -9.4163, 25.2445, -12.2471,
    -7.9000, -5.1002, -0.5673, -5.3688, -20.7775, -1.4971, 0.8325, -21.9982,
    13.7491, -1.2529, -0.6887, 0.4402, 2.7395, -49.1410, -17.5886, 1.1251,
    -3.3992, -3.6983, -21.0324, -9.2500, 0.4243, 21.4919, 35.5025, -1.7385,
    -7.1702, -1.8647, -31.7913, 3.2972, 22.5062, -33.8954, 19.9248, 15.2778,
    1.2341, -0.4782, 29.2818, -26.3183, -21.5343, 6.0856, 22.7864, 6.5349,
    0.3452, -0.3370, -11.2306, -1.7270, -1.0710, 1.1329, 29.6442, -1.2638,
    -14.9176, 7.3119, 27.3472, 41.6335, 7.8815, 1.5763, -35.4726, 4.5494,
    1.3973, -25.3534, -5.3839, 0.3754, 4.0579, -6.2750, -3.0345, 18.9430,
    -8.9200, 7.6707, -1.6687, 26.9521, -2.0668, 2.7718, 0.8757, -43.8139,
    5.2416, 1.1736, -21.8855, -1.8804, 28.8606, -1.8298, -1.7727, -0.7094,
    -29.6576, 2.7520, 8.7047, 0.8775, -4.9401, 6.2667, -2.4564, 1.2303,
    0.7133, -1.2054, -3.1296, 4.4453, 2.0878, -44.3737, -19.2787, -7.0846,
    6.0650, -4.5608, 4.8399, -36.2709, -23.5796, 4.2290, -26.2755, -40.9700,
    27.2111, 17.3362, -0.5826, -0.4368, 0.6360, 2.3116, 35.8805, 2.0020,
    0.6827, -4.2444, 34.4356, 12.7742, 7.0135, -1.1241, 1.1343, -2.4796,
    0.7732, -38.5961, -1.9439, -5.3551, -5.0703, -22.9698, -1.4817, -9.4255,
    0.4983, -5.7697, -0.5282, -10.8639, -5.8152, 7.1640, -25.2703, -0.6028,
    -1.7902, -6.8459, 40.7584, -3.6953, -0.8195, 17.2857, -0.9823, 4.6777,
    -2.3188, -1.8020, 0.4682, -1.5086, 3.1827, 2.2570, -42.5113, -1.5325,
    -45.4289, -0.4609, -2.7495, 1.2468, 19.8072, 0.5424, -8.7444, -0.5112,
    -2.7506, 23.0110, 29.7326, 9.2413, -0.4172, 0.8367, 2.8728, -5.1292,
    45.1418, 1.0446, -49.7796, 18.4872, 4.6173, 1.5947, 8.6695, -1.4702,
    -42.2081, -5.7747, -11.6209, 23.6807, -0.5971, 0.4817, -0.5132, -46.1042,
    -23.2041, 2.9892, -12.8944, 4.9225, 0.8169, -11.7521, -24.1292, -30.4917,
    0.5363, 15.0033, 0.6502, -4.2371, 25.6415, -0.3887, -2.8456, 1.1565,
    -0.5303, -16.0402, -4.1023, 0.5091, 39.6320, -0.6407, -36.5819, 0.3988,
    -3.1429, -4.8628, -3.7627, 9.2966, -28.9089, -0.6068, 11.9160, -0.9212,
    -3.0695, -2.5214, 20.8167, 3.1778, 0.5380, 0.5286, 0.8556, -49.7302,
    1.2080, -0.8891, 0.6758, 44.8201, 0.3784, -5.6986, -30.9674, 4.3529,
    -23.5926, 28.2864, 4.0459, 2.4585, 3.8565, 27.5002, 0.5920, 5.0998,
    28.1063, 31.5969, 0.8966, 9.7669, -13.1992, 0.9218, -5.0131, -41.6293,
    36.9164, 4.7946, 18.4634, -0.3847, -20.7165, 0.9566, -16.4771, 3.7186,
    -40.0144, 32.4842, -1.0041, 19.6608, 12.8003, -4.8528, 0.4683, 35.0420,
    5.9381, 0.3469, -1.5845, -4.0494, 0.3998, -25.2235, -5.6016, -6.7993,
    -6.5887, -0.5919, -0.7645, -2.3763, 29.0225, 9.6977, -11.7233, 12.2695,
    -45.3791, 5.8954, -39.3540, -19.1601, -2.2988, 5.3556, 13.9595, -1.7424,
    -12.5184, 4.8673, -39.4724, -0.7468, 0.3750, 2.6000, 2.2866, 16.9677,
    33.9860, 1.5250, -1.8004, -0.8034, -24.7007, -1.0147, 24.5948, 10.1044,
    -4.0150, 1.8335, 3.2396, -2.0917, 3.9398, -18.9960, 8.6122, -15.5162,
    0.4281, 1.7858, -3.1610, -1.0326, -0.7713, -2.3150, 0.4912, -3.4543,
    -3.7991, -35.7077, 0.4635, -1.4877, -1.3887, -34.1554, 11.1835, 1.2501,
    -21.1085, 1.4364, 11.9910, 2.4500, 0.8536, 0.4431, -6.7530, -2.4614,
    -0.8236, 14.3816, 3.4657, 20.6824, 5.1975, -42.3834, -1.2385, 0.4702,
    -0.9887, -0.6471, -3.5012, -3.1083, -8.1523, 6.1598, 3.1036, 49.2107,
    -6.4475, -3.5877, 5.6780, 1.0076, -0.8003, -11.1929, 0.4806, -0.5755,
    21.5453, -0.9426, 0.8060, 0.8834, 7.8613, 0.3524, 2.1130, 2.2940,
    1.5889, -1.9392, -0.7588, 2.0414, 5.9623, 3.1223, -1.5303, 9.4776,
    -0.7365, -20.6790, -0.6275, -21.8967, 3.4678, 40.1669, 1.5533, -1.1471,
    32.4282, 4.0313, 16.8355, 47.9455, -29.4821, 1.0638, 8.1738, -1.0564,
    23.0694, -3.4043, -4.1821, 0.7037, -38.1601, -33.2718, 36.6897, -27.3355,
    0.3797, 1.6306, 1.1376, 8.2619, 47.3004, 0.9248, -21.5169, 6.1745,
    26.5231, -0.7186, -2.8155, -0.8049, 4.4019, 3.0122, 7.6202, -1.9210,
    -12.2565, 1.1109, -5.1983, -18.6537, -30.8857, 1.6359, 0.7989, -1.5317,
    5.0025, 10.4580, -24.1215, -30.0958, -7.4727, 6.8375, 0.4189, -0.9271,
    13.5892, 2.0819, 10.3442, 1.0793, 6.5402, -19.6256, 5.4661, 18.6067,
    43.1606, 2.2130, -2.6568, 3.8015, -45.3122, 5.3695, 42.3410, -0.5277,
    -2.4049, 0.3944, -7.4950, -0.6338, -38.9908, 0.4302, 38.4059, -4.7719,
    -26.6535, 8.2919, 0.5148, 1.9696, 4.7666, 0.5506, 22.3075, -1.7331,
    -22.1971, 14.1993, -5.2124, -0.8638, -25.4711, 1.9838, 12.7246, 4.6497,
    44.4351, 3.9836, -0.8565, -4.2227, 20.8155, 4.8719, 1.9279, 5.4520,
    -2.1024, 31.5870, 35.4661, 7.2774, -0.3536, 13.4182, 0.4558, -23.8570,
    -0.5699, 21.3135, -1.1218, -18.9676, -0.3938, -18.3734, 6.5949, -26.8047,
    -1.6503, -1.1477, 3.2476, -1.7409, 15.9689, 0.4245, -0.4063, -0.8872,
    13.5956, -12.9969, 1.9245, 8.5954, -0.7145, 0.4034, 4.4463, -1.3593,
    -27.9014, 1.9560, -23.1299, -15.8807, -25.7697, 6.9218, 1.8535, 0.4190,
    -1.2963, -1.9978, 5.9517, 3.6022, 1.6907, -1.5770, -5.4249, 0.8097,
    2.6527, -0.9772, -7.0317, 5.1043, 26.3974, 2.5116, 19.4997, 18.9064,
    33.5957, 1.2986, 7.8039, -2.0266, 5.3515, 1.3789, -4.0720, 8.0923,
    7.5811, 8.6476, 11.3198, 6.6718, 2.1692, -0.3596, 22.8054, 2.0811,
    -43.5162, 5.5911, -3.6923, -37.6217, -3.5175, -2.6325, -2.0240, 1.8567,
    -14.3820, 1.3959, 29.0363, -9.6673, -2.0705, 1.1549, 26.5082, -0.8790,
    -0.7354, -2.2526, 7.1991, -10.9966, -3.6454, 3.2161, -16.5152, -13.1367,
    -26.1756, 13.4048, 2.6854, -7.1629, 19.1169, 41.8937, 3.6593, -2.9599,
    -2.6334, 2.1107, -7.9282, 0.6680, -22.8842, 1.8630, 1.3735, -1.8556,
    1.4216, 5.0111, -0.7044, 12.7409, -45.0101, -8.8956, -1.5563, 0.6844,
    13.0088, -0.6059, 0.4692, -25.1855, 0.6819, -37.7942, 8.5030, 4.8163,
    5.7647, 37.4734, 0.7726, -0.7424, -48.5362, 0.8150, -6.6525, -1.4933,
    2.2981, -2.2634, -44.0830, 1.0233, -3.1825, 40.7868, 4.3666, -1.7436,
    9.9665, -0.8794, 19.6407, -1.5827, -26.1363, 3.7889, -0.7715, 0.9495,
    47.7031, -3.1406, -7.0587, -12.1365, -3.3184, -24.0332, 8.1571, 11.1882,
    0.6173, -45.1529, 7.1443, 8.7102, -3.4288, 15.9802, -0.7869, -25.6034,
    -0.4652, -3.1788, -2.5347, 0.5881, -5.4787, -3.9761, -34.9223, 23.7552,
    2.3808, 1.8261, -0.4849, -24.4955, 34.2981, 47.1433, 0.6746, 9.0688,
    -1.9635, -0.4653, -30.3865, -38.3615, 1.2744, -1.9680, -37.9759, 0.5599,
    -7.0050, 1.7194, 48.7140, -47.1340, -6.1836, -24.0075, -19.7049, 12.5772,
    44.2977, -1.8947, 7.4001, -1.1716, -0.6331, -0.3584, 12.4165, 0.4489,
    7.8047, -16.5561, -30.9523, 14.0117, -0.4793, 16.1063, -4.6074, -1.7472,
    -10.0926, 1.1258, -4.4680, -46.6639, 16.3122, 0.5887, 0.4840, -0.4468,
    0.4685, 8.6699, 2.4549, -0.9377, 1.5193, 1.2554, 0.7636, 16.0608,
    22.7224, 29.6637, -44.6360, -2.6350, -1.0396, 2.1428, 8.5860, 19.7406,
    43.2104, -8.7560, 44.3145, -1.9150, 12.6630, -1.3440, 1.5179, 49.0105,
    -1.5045, -1.3114, -1.5380, 25.1990, 4.0325, -2.1410, 0.6557, -35.2483,
    5.1356, -7.4375, -2.6712, 4.4843, 1.5655, -25.5623, -3.4291, 12.2437,
    -13.9534, 1.5637, 9.2233, 21.9163, 1.3943, 8.1781, -0.8256, -9.8400,
    -13.0506, 0.9794, -1.1987, 1.2688, -1.2253, 19.1329, 9.8187, -1.6455,
    0.6493, 1.3160, 2.2526, -1.8189, 43.5400, 3.1279, 10.2816, 0.6233,
    27.7232, -30.7255, 0.3565, 29.0884, 9.1771, 7.3503, -8.0420, 14.5371,
    0.5533, 13.9716, -39.9177, 14.8484, -10.0154, 16.1588, 8.4652, 13.1098,
    37.2542, -25.0003, -0.8557, 19.3870, 2.3483, 21.8231, -0.8941, 20.3213,
    -4.6253, 5.5894, 1.3152, -10.8232, 1.3174, -11.6099, 2.4895, 17.3031,
    -12.5247, 0.5246, 3.0003, 0.6972, -26.9009, -6.7453, -1.9697, 0.9405,
    6.3567, -3.9841, 18.5297, -1.1870, 0.5482, -1.0974, -4.0320, -0.9539,
    -2.6283, 0.5457, 1.1218, 3.2808, -11.1916, -40.1699, -0.4817, -3.7826,
    -10.1268, -2.5749, 2.1129, -12.8163, -6.9338, -0.4149, -0.7530, -2.7736,
    -1.3887, 15.1618, -4.2699, -0.4381, 10.7685, -0.4184, 0.4132, -0.5683,
    -3.0891, -37.0586, 3.7443, -3.4262, 0.7982, 11.2543, 0.3516, 1.7945,
    15.1541, 5.3727, 10.9893, -12.1097, 1.3240, -10.8557, 0.4501, 28.5685,
    -0.5436, 1.0688, -4.0096, 0.5131, 47.4227, -0.5644, 0.6424, -2.0830,
    6.1150, 0.4571, -1.0143, -0.4657, 2.4589, 10.2804, 2.7196, -0.3674,
    32.0037, -2.0465, -0.4143, 2.1000, 0.3581, -0.7460, 15.4532, 6.0395,
    -14.9839, -0.7694, -38.9688, 3.9508, 41.0130, -23.4663, 1.1519, -0.9907,
    1.4221, 11.8652, 5.2609, 18.9480, -21.9895, 0.7228, 25.6898, -1.3619,
    -11.5029, 1.0447, 1.0083, 3.8167, 0.4319, -1.3041, -1.0777, -8.8163,
    6.7994, -14.5714, -2.4248, -39.1383, -0.6763, 13.2674, -0.4058, -0.5019,
    -27.6223, 1.0942, 6.0113, -5.9743, -6.8262, -30.6558, -0.5526, 4.4949,
    -0.4222, -0.8084, -3.3795, 21.6369, 25.7783, -1.5932, -5.6110, 3.9644,
    -2.0846, -32.6294, -9.1964, 48.0458, -0.4493, -19.3177, -1.3670, -10.4115,
    2.4264, 12.7935, 0.4950, 2.1552, -11.7497, -10.0093, 1.9362, 3.2647,
    6.4454, -1.4147, -24.2864, 1.8497, -11.2951, 12.5549, 18.4699, 48.4130,
    45.9686, -3.0786, -0.5485, -22.0178, 0.5945, 8.4500, 3.2526, 0.6205,
    -4.4486, 3.1308, 2.8764, -2.3412, -16.2496, -3.6923, -10.1643, 3.6543,
    15.1992, -0.5404, -6.8895, 6.4220, -7.4233, 6.4353, -1.5126, 2.4574,
    -1.1254, -1.0306, -3.0045, -8.1719, 15.4491, -9.1433, 1.6172, 21.2405,
    9.8597, -0.5402, -0.7356, -7.5720, 0.4255, -33.2450, -15.3795, -9.4754,
    -0.7625, -5.6333, -12.3756, 18.5254, 3.0695, -0.7300, -1.0985, 25.2456,
    3.1974, 0.5766, 0.3675, 47.6171, 7.8671, -1.0306, -4.4177, 17.3085,
    -5.0114, -0.9255, -0.8242, -9.6292, 13.8880, -8.5472, -2.1244, 7.8471,
    -0.6515, -0.8841, -6.6881, -8.4762, -16.9598, -0.6447, -9.7682, 0.3639,
    0.6685, 2.7951, 11.4252, -38.6160, 5.7872, -36.2558, -6.2120, -7.9746,
    31.1643, 24.5567, -1.8160, -1.0535, -1.2125, -0.4479, 2.7929, -18.4731,
    -9.2126, 24.1589, 14.6135, -2.1823, 31.0223, -28.6788, 0.9336, 1.2434,
    8.6658, -35.9428, 4.7575, -2.7070, 3.6441, -2.7915, 37.7129, 2.0484,
    5.4158, 0.4434, 2.8649, -6.8693, -1.0390, 9.5758, 0.4235, 1.0041,
    0.7551, 18.8455, -0.6412, -10.1775, -34.6449, 23.7052, -2.7889, -2.8932,
    2.7767, 3.7088, 0.3595, 0.4042, -30.9624, -13.0989, 42.9484, 15.5501,
    -2.3576, -0.6735, 0.4120, -11.5776, -11.7645, -3.6045, 1.2354, 8.4603,
    -11.3359, 3.2239, -6.4409, 0.6003, 4.0165, -0.9219, -0.3540, -0.8851,
    1.8654, 2.3238, 12.1453, -10.0763, -11.9290, 0.6325, 7.2727, 0.5727,
    16.9017, 4.7496, 5.0233, 17.1861, 0.3640, 0.6764, -0.4307, 8.2740,
    1.7659, 0.3425, -4.7958, -3.1272, 1.7024, 0.6822, -20.8880, 2.5135,
    -4.1258, 2.0493, -6.4976, -18.9107, 0.7284, 27.4679, 2.4935, -19.4052,
    -1.1343, -0.4918, 0.4222, -8.8152, 1.7567, 43.6363, -0.3384, 30.2895,
    26.8572, 28.5060, -0.5187, -17.8179, 45.9934, 27.7956, -10.3631, 1.6792,
    8.4143, -0.3948, -1.1912, 1.4940, -0.7861, 42.6061, -0.6429, -1.7058,
    0.3813, 4.7923, -33.7688, -7.7746, 9.6927, -3.7713, 0.4760, 1.4976,
    0.5487, 13.9478, 0.5144, 0.9734, -17.5199, -1.0433, 12.7862, 0.7691,
    -0.8413, -36.6983, 3.2887, -8.8633, 18.0153, -47.7063, 5.9342, 0.5125,
    -2.1882, -17.7256, -1.0746, 13.4022, -14.8502, -0.4170, -16.9628, 15.9685,
    -35.6380, 16.5698, -30.9581, -1.7332, 0.4912, -5.7109, 1.3899, 10.5945,
    27.3063, -45.2730, -35.1708, -0.3417, 0.8852, -36.0659, 4.4451, -0.6657,
    -0.7022, -0.6415, 0.4092, 0.5662, 48.2658, 7.9954, -20.1914, 0.4071,
    -4.3605, -1.2701, -31.2389, 7.1306, 0.5503, 25.1649, -1.9230, -20.1893,
    -43.4277, 17.4664, 0.6507, -25.5450, -10.3408, -0.4619, 3.4695, -27.2522,
    -0.4025, 1.1121, -0.6221, -17.7571, 1.2923, 1.4041, -1.9759, -3.4307,
    0.6484, 4.1147, -1.0433, -0.5225, -1.4547, -1.2854, 2.6603, -41.0906,
    -32.7794, -2.0575, 49.9940, -1.5579, 26.4737, 2.7917, 1.0970, -14.1645,
    4.9067, -0.6416, 1.4820, -6.4672, 28.0232, 26.2066, 3.5259, -8.2326,
    -3.4773, -1.1508, 0.7191, -2.4653, -2.3877, 3.3875, -22.4413, 3.0630,
    1.3014, 16.9551, -6.1947, -4.6190, -0.4048, -2.1227, -0.6093, -22.8726,
    -2.8268, -15.2523, 3.8177, -1.5192, 10.9055, 39.2468, -10.2063, 3.6665,
    -12.9808, 4.3271, 12.1420, -9.6750, -0.5904, -0.5230, -21.7613, 0.5665,
    0.9689, 0.7669, 27.4244, 1.8962, 22.6919, 0.3405, -2.2881, 0.4990,
    -23.7101, 35.8071, -11.6297, 0.5719, 0.3646, -0.4035, 4.9627, 1.7163,
    -6.8874, -0.8694, 3.2363, 1.1880, -0.4463, 2.8113, -0.9066, -4.4813,
    -2.4751, -2.6682, -9.8693, -0.9686, 1.6443, -40.7320, 1.0650, 8.5446,
    2.6970, 2.6485, 0.5519, -1.4303, 5.6647, 1.8865, 12.3145, 23.9689,
    0.4236, 0.9143, -2.9596, -15.6080, 8.6023, 8.4536, 6.2455, 6.8653,
    -0.4536, 0.3838, -42.6926, -7.5378, 2.0184, 1.5414, -14.7434, -1.5239,
    43.8171, 18.2668, -0.5221, 4.3259, -13.5232, 0.7213, -1.4764, -0.4524,
    -0.8027, -0.3579, 3.4593, -9.3217, -20.6289, 2.4918, -29.5549, -4.9173,
    2.1889, 36.4780, -25.4149, -9.5289, 9.9369, -12.0742, 0.3926, 0.3504,
    -1.7638, -25.0928, 0.5049, 1.1683, 12.7511, -1.0157, 12.9651, -28.4795,
    -0.6231, 9.5865, -0.9358, -24.5529, -0.8555, 2.8489, 0.9438, -17.1671,
    9.3419, -2.7534, -0.9385, 4.8776, 32.9251, 1.5152, -1.1017, 0.4546,
    0.6393, 5.4734, -24.6250, 3.6206, 1.1917, 8.5122, 1.0940, -8.2935,
    -19.3265, -2.3836, -6.6860, -0.9684, -0.7502, -1.4332, -1.1576, -0.5613,
    -18.4173, -0.4108, 1.5016, -49.6227, -41.3693, -3.2672, 5.7548, -9.4532,
    46.4924, 0.7477, -5.2059, 8.4991, 18.4795, -0.7393, 34.4152, -3.1669,
    1.5889, 0.3425, -0.5262, -32.9662, 7.0713, 4.0759, 38.7602, -0.5239,
    13.2327, -2.0529, -11.3059, 0.4529, -5.0832, 3.8488, -2.5309, -7.5687,
    1.0545, 15.6816, 0.3603, -33.9734, -2.1180, -3.6557, 6.9180, -1.3370,
    15.6405, 9.2982, 2.4203, -2.5993, -1.8862, 0.3960, -5.8581, -21.8254,
    0.6525, 33.4507, -1.1730, 2.4934, 47.0259, -9.0120, -14.3977, -48.9796,
    -35.4531, -0.8041, 23.7195, 0.4640, -10.1728, 0.4015, 11.5091, 12.6167,
    -14.6642, 0.9585, -17.7863, 0.8542, 8.9192, 6.1672, -14.7457, -14.1100,
    -0.5526, 16.6482, 2.2632, -3.5841, -3.2274, -3.5019, 5.7719, -28.2049
};

/**
 * Function that computes the Harris responses in a
 * 2*r x 2*r patch at given points in the image
 */
static void
HarrisResponses(InputArray _img, InputArray _diff_x, InputArray _diff_y,
                std::vector<KeyPoint>& pts,
                OutputArray _response, int r, float harris_k)
{
    size_t ptidx, ptsize = pts.size();

    // Get mats
    Mat img = _img.getMat(), diff_x = _diff_x.getMat(),
        diff_y = _diff_y.getMat(), response;
    CV_Assert( img.type() == CV_8UC1 );


    bool compute_response = _response.needed();
    if (compute_response) response = _response.getMat();

    const int* dx00 = diff_x.ptr<int>();
    const int* dy00 = diff_y.ptr<int>();
    float* r00 = response.ptr<float>();
    int step = diff_x.step1();
    int r_step = response.step1();

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        float kp_x = pts[ptidx].pt.x;
        float kp_y = pts[ptidx].pt.y;
        int x0 = (int)kp_x;
        int y0 = (int)kp_y;

        float xd = 2;
        float yd = 2;
        //float xd = 0.5;
        //float yd = 0.5;

        const int* dx0 = dx00 + (y0)*step + x0;
        const int* dy0 = dy00 + (y0)*step + x0;
        int a = 0, b = 0, c = 0, d = 0;
        float* r0 = r00 + ptidx*r_step;

        for( int i = -r; i < r; i++ )
        {
            for( int j = -r; j < r; j++ )
            {
                const int ofs = i*step + j;
                const int* dx = dx0 + ofs;
                const int* dy = dy0 + ofs;
                const int Ix = (float)dx[-1]*(xd) + (float)dx[0] + (float)dx[1]*xd + (float)dx[-step]*(yd) + (float)dx[step]*yd;
                const int Iy = (float)dy[-1]*(xd) + (float)dy[0] + (float)dy[1]*xd + (float)dy[-step]*(yd) + (float)dy[step]*yd;
                a += (Ix*Ix);
                b += (Iy*Iy);
                c += (Ix*Iy);
                d += Ix;
            }
        }
        if (compute_response) {
            r0[0] = (float)a;
            r0[1] = (float)b;
            r0[2] = (float)c;
            r0[3] = (float)d;
        }
        else
            pts[ptidx].response = ((float)a * b - (float)c * c -
                                   harris_k * ((float)a + b) * ((float)a + b));
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline float pick( const Mat& img, float x, float y)
{
    int step = img.step; // The same as step1() for single channel image
    int x0 = (int)x;
    int y0 = (int)y;
    float xd = (x - (float)x0);
    float yd = (y - (float)y0);
    const uchar* ptr = img.ptr<uchar>() + y0*step + x0;
    return ptr[-1]*(1 - xd) + ptr[0] + ptr[1]*xd + ptr[-step]*(1 - yd) + ptr[step]*yd;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

static void
computeSkew( const Mat& responses, Mat& skew, int nkeypoints, bool fullRotation)
{
    for ( int i = 0; i < nkeypoints; i++ )
    {
        // Declare pointers to the rows corresponding to keypoint `i`
        const float* resp_row = responses.ptr<float>(i);
        float* skew_row = skew.ptr<float>(i);
        int skew_sgn = sgn(resp_row[3]);

        // Calculate eigenvectors and values
        float a = resp_row[0], b = resp_row[2], c = resp_row[2], d = resp_row[1];
        float T = a + d;
        float T2 = T*T;
        float D = a*d - b*c;
        float L1 = T/2.f + std::sqrt(T2/4.f-D);
        float L2 = T/2 - std::sqrt(T2/4-D);
        float V1 = -1*b;
        float V2 = a - L1;

        // Normalize eigenvectors and values
        float V_norm = std::sqrt(V1*V1 + V2*V2);
        V1 = V1 / V_norm;
        V2 = V2 / V_norm;
        const float val_sq = std::sqrt(L1*L2);
        if (fullRotation) {
            L1 = L1 / val_sq * skew_sgn;
            L2 = L2 / val_sq * skew_sgn;
        } else {
            L1 = L1 / val_sq;
            L2 = L2 / val_sq;
        }

        // Calculate transformation matrix based on the matrix multiplication
        // of skew `diag(eig_val)` and rotate [-1*vec[1] vec[0]; vec[0] vec[1]]
        skew_row[0] = -1*V2*L1;
        skew_row[1] = V1*L2;
        skew_row[2] = V1*L1;
        skew_row[3] = V2*L2;
    }
}

static void
computeBAFTDescriptors( const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
                       const std::vector<float>& layerScale, const Mat& harrisResponse, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, int dsize, int patchSize, bool fullRotation)
{
    // Compute skew matrix for each keypoint
    int nkeypoints = (int)keypoints.size();
    float scale_modifier = (float)patchSize / 50.f;
    Mat skew(nkeypoints, 4, CV_32F), points_kp, s, img_roi;
    computeSkew(harrisResponse, skew, nkeypoints, fullRotation);

    // Now for each keypoint, collect data for each point and construct the descriptor
    KeyPoint kp;
    for (int i = 0; i < nkeypoints; i++)
    {
        // Find image region of interest in image pyramid
        kp = keypoints[i];
        float scale = scale_modifier;
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
            float x0 = (p[2*j]*s[0] + p[2*j+1]*s[1])*scale + x;
            float y0 = (p[2*j]*s[2] + p[2*j+1]*s[3])*scale + y;

            picked = pick(img_roi, x0, y0);
            if (picked < min_val)
            {
                min_idx = j % 4;
                min_val = picked;
            }
            if (picked > max_val)
            {
                max_idx = j % 4;
                max_val = picked;
            }
            if ((j+1) % 4 == 0) {
                if ((j+1) % 8 == 0) {
                    desc[(j >> 3)] = (uchar)((byte_val << 4) + ((max_idx << 2) + (min_idx)));
                }
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


class BAFT_Impl : public BAFT
{
public:
    explicit BAFT_Impl(int _nfeatures, int _size, int _patchSize,
                       int _gaussianBlurSize, bool _fullRotation,
                       float _scaleFactor, int _nlevels,
                       int _edgeThreshold, int _fastThreshold) :
        nfeatures(_nfeatures), size(_size), patchSize(_patchSize),
        gaussianBlurSize(_gaussianBlurSize), fullRotation(_fullRotation),
        scaleFactor(_scaleFactor), nlevels(_nlevels), edgeThreshold(_edgeThreshold),
        fastThreshold(_fastThreshold)
    {}

    void setMaxFeatures(int maxFeatures) { nfeatures = maxFeatures; }
    int getMaxFeatures() const { return nfeatures; }

    void setSize(int size_) { size = size_; }
    int getSize() const { return size; }

    void setPatchSize(int patchSize_) { patchSize = patchSize_; }
    int getPatchSize() const { return patchSize; }

    void setGaussianBlurSize(int gaussianBlurSize_) { gaussianBlurSize = gaussianBlurSize_; }
    int getGaussianBlurSize() const { return gaussianBlurSize; }

    void setFullRotation(bool fullRotation_) { fullRotation = fullRotation_; }
    bool getFullRotation() const { return fullRotation; }

    void setScaleFactor(double scaleFactor_) { scaleFactor = scaleFactor_; }
    double getScaleFactor() const { return scaleFactor; }

    void setNLevels(int nlevels_) { nlevels = nlevels_; }
    int getNLevels() const { return nlevels; }

    void setEdgeThreshold(int edgeThreshold_) { edgeThreshold = edgeThreshold_; }
    int getEdgeThreshold() const { return edgeThreshold; }

    void setFastThreshold(int fastThreshold_) { fastThreshold = fastThreshold_; }
    int getFastThreshold() const { return fastThreshold; }

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;
    // returns the default norm type
    int defaultNorm() const;

    // Compute the BAFT_Impl features and descriptors on an image
    void detectAndCompute( InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false );

protected:

    int nfeatures;
    int size;
    int patchSize;
    int gaussianBlurSize;
    bool fullRotation;
    double scaleFactor;
    int nlevels;
    int edgeThreshold;
    int fastThreshold;
};

int BAFT_Impl::descriptorSize() const
{
    return size;
}

int BAFT_Impl::descriptorType() const
{
    return CV_8U;
}

int BAFT_Impl::defaultNorm() const
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


/** Compute the BAFT_Impl keypoints and their Harris Response on an image
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
static void computeKeyPoints(InputArray _imagePyramid,
                             InputArray _maskPyramid,
                             InputArray _diff_x, InputArray _diff_y,
                             const std::vector<Rect>& layerInfo,
                             const std::vector<float>& layerScale,
                             std::vector<KeyPoint>& allKeypoints,
                             OutputArray _response,
                             int nfeatures, double scaleFactor,
                             int edgeThreshold, int patchSize,
                             int fastThreshold  )
{
    int i, nkeypoints, level, nlevels = (int)layerInfo.size();
    std::vector<int> nfeaturesPerLevel = featuresPerLevel(nlevels, nfeatures, scaleFactor);

    // Get mats
    Mat imagePyramid = _imagePyramid.getMat(), maskPyramid = _maskPyramid.getMat(),
        diff_x = _diff_x.getMat(), diff_y = _diff_y.getMat(), response = _response.getMat();

    allKeypoints.clear();
    std::vector<KeyPoint> keypoints;
    keypoints.reserve(nfeaturesPerLevel[0]*2);

    Mat cur_response(nfeatures, 4, CV_32F);
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
        HarrisResponses(img, dx, dy, keypoints, noArray(), 1, HARRIS_K);
        KeyPointsFilter::retainBest(keypoints, featuresNum);

        nkeypoints = (int)keypoints.size();
        HarrisResponses(img, dx, dy, keypoints, cur_response, 6, HARRIS_K);
        for( i = 0; i < nkeypoints; i++ )
        {
            keypoints[i].octave = level;
            keypoints[i].size = patchSize*layerScale[level];
            keypoints[i].pt *= layerScale[level];
            int index = i;
            float* response_row = response.ptr<float>(i + responseOffset);
            float* cur_row = cur_response.ptr<float>(index);
            for ( int j = 0; j < 4; j++ )
                response_row[j] = cur_row[j];
        }

        responseOffset += nkeypoints;
        std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(allKeypoints));
    }
}


/** Compute the BAFT_Impl features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 */
void BAFT_Impl::detectAndCompute( InputArray _image, InputArray _mask,
                                 std::vector<KeyPoint>& keypoints,
                                 OutputArray _descriptors, bool useProvidedKeypoints )
{
    CV_Assert(patchSize >= 2);

    bool do_keypoints = !useProvidedKeypoints;
    bool do_descriptors = _descriptors.needed();

    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;

    //int border = std::max(edgeThreshold, patchSize);
    int border = 30;//std::max(edgeThreshold, patchSize);

    Mat orig_image = _image.getMat();
    Mat image, mask = _mask.getMat();
    if( orig_image.type() != CV_8UC1) {
        cvtColor(orig_image, image, COLOR_BGR2GRAY);
    }
    else
        image = orig_image;
    //int kp_x = 670, kp_y = 440;

    int i, level, nLevels = this->nlevels, nkeypoints = (int)keypoints.size();

    // TODO: The way this should work: Find min size. Find corresponding size to each
    // level, assuming scaleFactor that we have currently and assign correct levels.
    // Note the highest octave/level assigned and use that as nLevels
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
    Mat harrisResponse(nfeatures, 4, CV_32F);
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
    else // supplied keypoints TODO: support could be much much better
    {
        KeyPointsFilter::runByImageBorder(keypoints, image.size(), edgeThreshold);
        HarrisResponses(image, diff_x(layerInfo[0]), diff_y(layerInfo[0]), keypoints, noArray(), 1, HARRIS_K);
        HarrisResponses(image, diff_x(layerInfo[0]), diff_y(layerInfo[0]), keypoints, harrisResponse, 6, HARRIS_K);
    }

    if( do_descriptors )
    {
        int dsize = descriptorSize();
        nkeypoints = (int)keypoints.size();
        _descriptors.create(nkeypoints, dsize, CV_8U);
        Mat descriptors = _descriptors.getMat();
        if (gaussianBlurSize > 0) {
            int s = gaussianBlurSize;
            for( level = 0; level < nLevels; level++ )
            {
                Mat workingMat = imagePyramid(layerInfo[level]);
                GaussianBlur(workingMat, workingMat, Size(s, s), 2, 2, BORDER_REFLECT_101);
            }
        }
        computeBAFTDescriptors(imagePyramid, layerInfo, layerScale, harrisResponse,
                               keypoints, descriptors, dsize, patchSize, fullRotation);
    }
}

Ptr<BAFT> BAFT::create(int nfeatures, int size, int patchSize,
                       int gaussianBlurSize, bool fullRotation,
                       float scaleFactor, int nlevels,
                       int edgeThreshold, int fastThreshold)
{
    return makePtr<BAFT_Impl>(nfeatures, size, patchSize, gaussianBlurSize,
                              fullRotation, scaleFactor, nlevels,
                              edgeThreshold, fastThreshold);
}

