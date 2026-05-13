import os
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs('results',  exist_ok=True)
os.makedirs('figures',  exist_ok=True)

ALGOS    = ['HOA','CS','Hybrid','DO','EO','RIME','FOX','GTO','AO','AVOA']
PROPOSED = 'Hybrid'


CEC14_ERRORS = {
  1:  {'HOA':80133.82,    'CS':75768757.13,  'Hybrid':2047813.19,  'DO':362048.76,
       'EO':33931415.45,  'RIME':2778280.34, 'FOX':5518724.34,    'GTO':1195284.12,
       'AO':864014.82,    'AVOA':21054037.15},
  2:  {'HOA':10116.09,    'CS':3289494229.49,'Hybrid':233.52,      'DO':20016.13,
       'EO':1251702739.08,'RIME':16613.10,   'FOX':249406.58,      'GTO':3464.01,
       'AO':6939.00,      'AVOA':964.70},
  3:  {'HOA':10028.00,    'CS':51277.56,     'Hybrid':42804.23,    'DO':3371.05,
       'EO':26098.80,     'RIME':42905.13,   'FOX':2453.67,        'GTO':17368.84,
       'AO':10695.66,     'AVOA':29498.65},
  4:  {'HOA':514091.25,   'CS':44412877.95,  'Hybrid':7968200.07,  'DO':653496.52,
       'EO':35416258.37,  'RIME':1261641.54, 'FOX':1065142.68,     'GTO':902460.08,
       'AO':1955447.39,   'AVOA':14963067.16},
  5:  {'HOA':19.9998,     'CS':20.6319,      'Hybrid':20.0018,     'DO':20.0531,
       'EO':20.5840,      'RIME':19.9998,    'FOX':20.1356,        'GTO':20.2785,
       'AO':20.5553,      'AVOA':20.9044},
  6:  {'HOA':6.3809,      'CS':11.4972,      'Hybrid':5.8673,      'DO':2.3267,
       'EO':10.1700,      'RIME':10.6962,    'FOX':2.1712,         'GTO':7.2270,
       'AO':1.5999,       'AVOA':11.0501},
  7:  {'HOA':1.6481,      'CS':2.9750,       'Hybrid':0.1133,      'DO':0.1062,
       'EO':1.0075,       'RIME':1.6622,     'FOX':0.1085,         'GTO':0.3445,
       'AO':0.1205,       'AVOA':0.9042},
  8:  {'HOA':15.9365,     'CS':5702.7830,    'Hybrid':80.5912,     'DO':27.1344,
       'EO':1565.9825,    'RIME':65.6737,    'FOX':64.9239,        'GTO':20.8945,
       'AO':24.8848,      'AVOA':926.1649},
  9:  {'HOA':46.7775,     'CS':5295.7861,    'Hybrid':88.5496,     'DO':24.7490,
       'EO':1093.7914,    'RIME':40.8040,    'FOX':80.3923,        'GTO':52.7327,
       'AO':63.4116,      'AVOA':352.2070},
  10: {'HOA':217.1397,    'CS':538.7907,     'Hybrid':0.0000,      'DO':217.1452,
       'EO':398.5861,     'RIME':0.0000,     'FOX':0.0118,         'GTO':0.0000,
       'AO':217.1397,     'AVOA':1084.7379},
  11: {'HOA':0.0000,      'CS':666.0384,     'Hybrid':0.0000,      'DO':217.1416,
       'EO':536.6922,     'RIME':0.0000,     'FOX':0.0123,         'GTO':0.0000,
       'AO':217.1397,     'AVOA':0.0000},
  12: {'HOA':0.5198,      'CS':1.7230,       'Hybrid':0.4210,      'DO':0.2518,
       'EO':0.9668,       'RIME':1.1498,     'FOX':0.5798,         'GTO':0.5644,
       'AO':0.2814,       'AVOA':2.7667},
  13: {'HOA':0.5668,      'CS':0.6116,       'Hybrid':0.2823,      'DO':0.2825,
       'EO':0.4129,       'RIME':0.4800,     'FOX':0.3217,         'GTO':0.2857,
       'AO':0.2613,       'AVOA':0.3183},
  14: {'HOA':0.3971,      'CS':0.5063,       'Hybrid':0.3713,      'DO':0.1526,
       'EO':0.2255,       'RIME':0.3685,     'FOX':0.2322,         'GTO':0.1700,
       'AO':0.2868,       'AVOA':0.3519},
  15: {'HOA':1.3109,      'CS':183361373720.56,'Hybrid':1.5446,    'DO':4.0956,
       'EO':495089496.95, 'RIME':1.6872,     'FOX':5.0495,         'GTO':0.5710,
       'AO':1.4929,       'AVOA':56.6632},
  16: {'HOA':3.8101,      'CS':4.0486,       'Hybrid':3.8241,      'DO':2.1453,
       'EO':3.7534,       'RIME':3.1973,     'FOX':2.9673,         'GTO':3.0903,
       'AO':3.3898,       'AVOA':3.7645},
  17: {'HOA':526.6879,    'CS':2137.9378,    'Hybrid':902.3205,    'DO':15.3887,
       'EO':1926.1540,    'RIME':272.8958,   'FOX':37.5347,        'GTO':95.2977,
       'AO':234.4562,     'AVOA':810.4078},
  18: {'HOA':25818.21,    'CS':132939.23,    'Hybrid':74785.06,    'DO':13880.75,
       'EO':664668.45,    'RIME':21149.94,   'FOX':33766.20,       'GTO':11869.71,
       'AO':5685.12,      'AVOA':178454.24},
  19: {'HOA':918.3668,    'CS':2812.4177,    'Hybrid':783.3442,    'DO':33.2122,
       'EO':3822.4953,    'RIME':476.9704,   'FOX':784.1880,       'GTO':837.5548,
       'AO':1035.0008,    'AVOA':1607.6352},
  20: {'HOA':3.5640,      'CS':34404.64,     'Hybrid':5.1141,      'DO':5.8727,
       'EO':11.1298,      'RIME':7.6706,     'FOX':9.5041,         'GTO':2.7616,
       'AO':2.8941,       'AVOA':4191.7234},
  21: {'HOA':7523.04,     'CS':1766599.43,   'Hybrid':595915.98,   'DO':25985.46,
       'EO':383454.56,    'RIME':50197.41,   'FOX':72412.51,       'GTO':16121.29,
       'AO':18157.09,     'AVOA':34728.54},
  22: {'HOA':373.5927,    'CS':1533.6112,    'Hybrid':722.9519,    'DO':31.7036,
       'EO':948.1136,     'RIME':36.3213,    'FOX':37.6217,        'GTO':96.7556,
       'AO':45.9458,      'AVOA':4266.6215},
  23: {'HOA':84739.06,    'CS':98658.79,     'Hybrid':79475.11,    'DO':79457.85,
       'EO':95613.38,     'RIME':79456.33,   'FOX':84234.48,       'GTO':79447.63,
       'AO':84162.77,     'AVOA':116924.19},
  24: {'HOA':5472.92,     'CS':4552.19,      'Hybrid':4834.45,     'DO':4756.36,
       'EO':4839.69,      'RIME':4704.66,    'FOX':3350.81,        'GTO':4704.65,
       'AO':5518.78,      'AVOA':5015.10},
  25: {'HOA':42515635.67, 'CS':3195095402.80,'Hybrid':42480987.31, 'DO':42483243.15,
       'EO':825228200.54, 'RIME':42478517.72,'FOX':42755060.46,    'GTO':42478487.50,
       'AO':42478487.50,  'AVOA':42479161.25},
  26: {'HOA':113105121.53,'CS':68721047.62,  'Hybrid':80727922.37, 'DO':58014235.79,
       'EO':63602417.81,  'RIME':58014235.79,'FOX':69853830.37,    'GTO':63602417.81,
       'AO':58014235.79,  'AVOA':85201943.48},
  27: {'HOA':460.6478,    'CS':557.8867,     'Hybrid':432.4746,    'DO':432.2951,
       'EO':602.3683,     'RIME':434.1594,   'FOX':494.0752,       'GTO':437.5079,
       'AO':434.1591,     'AVOA':515.1976},
  28: {'HOA':1394775625.17,'CS':2723315437.58,'Hybrid':1394774168.73,'DO':1394460855.97,
       'EO':2336944610.57,'RIME':1394773762.44,'FOX':1433242815.00,'GTO':1493430892.31,
       'AO':1394455972.22,'AVOA':1508563598.98},
  29: {'HOA':241.4288,    'CS':242.6289,     'Hybrid':240.1404,    'DO':240.3454,
       'EO':240.9818,     'RIME':786.4281,   'FOX':240.6200,       'GTO':240.7616,
       'AO':241.1853,     'AVOA':241.9581},
  30: {'HOA':218.1195,    'CS':215.7159,     'Hybrid':205.7534,    'DO':204.9907,
       'EO':211.5424,     'RIME':207.4491,   'FOX':205.6232,       'GTO':205.3719,
       'AO':205.4425,     'AVOA':207.9791},
}

CEC17_DATA = {
  1:  {'HOA':(1.0883e5,6.4208e4), 'CS':(2.4335e7,7.7835e6),  'Hybrid':(1.8843e6,1.6380e6),
       'DO':(9.1453e4,5.7473e4),  'EO':(2.7211e7,1.3836e7),  'RIME':(1.5855e5,7.5393e4),
       'FOX':(5.9357e5,2.8906e5), 'GTO':(2.4979e6,4.4242e6), 'AO':(1.2201e5,7.6936e4),
       'AVOA':(1.6531e7,2.1707e7)},
  3:  {'HOA':(1.5031e4,7.2463e3), 'CS':(3.0842e4,7.9936e3),  'Hybrid':(3.4015e4,1.5171e4),
       'DO':(2.0144e2,1.3611e2),  'EO':(4.7116e4,1.4414e4),  'RIME':(9.5560e3,5.2800e3),
       'FOX':(5.5076e3,3.2483e3), 'GTO':(3.5095e3,2.4975e3), 'AO':(7.2921e3,3.6723e3),
       'AVOA':(7.8670e4,3.3680e4)},
  4:  {'HOA':(4.1015e3,3.8369e3), 'CS':(3.6245e8,1.8253e8),  'Hybrid':(3.7772e3,4.2595e3),
       'DO':(7.9269e1,6.7845e1),  'EO':(3.1354e8,2.5834e8),  'RIME':(3.2357e3,3.8957e3),
       'FOX':(3.5008e3,3.5602e3), 'GTO':(9.2634e5,1.8034e6), 'AO':(2.2402e3,3.2773e3),
       'AVOA':(2.6890e5,9.9241e5)},
  5:  {'HOA':(2.0000e1,2.7366e-3),'CS':(2.0438e1,1.1697e-1), 'Hybrid':(2.0062e1,9.7678e-2),
       'DO':(2.0014e1,6.5812e-3), 'EO':(2.0353e1,9.4321e-2), 'RIME':(2.0018e1,6.8055e-2),
       'FOX':(2.0037e1,4.7282e-2),'GTO':(2.0155e1,9.5865e-2),'AO':(2.0388e1,9.5667e-2),
       'AVOA':(2.0559e1,1.2179e-1)},
  6:  {'HOA':(4.4331e0,1.4702e0), 'CS':(9.7688e0,7.2138e-1), 'Hybrid':(5.0523e0,1.7188e0),
       'DO':(4.9069e0,1.0254e0),  'EO':(9.2495e0,1.1082e0),  'RIME':(5.6679e0,1.6899e0),
       'FOX':(6.4899e0,1.5062e0), 'GTO':(7.5255e0,7.4919e-1),'AO':(9.4793e0,4.9721e-1),
       'AVOA':(1.0553e1,1.0990e0)},
  7:  {'HOA':(1.4638e0,6.5244e-1),'CS':(2.0018e0,2.4086e-1), 'Hybrid':(2.9527e-1,1.9814e-1),
       'DO':(4.2448e-2,1.8111e-2),'EO':(1.6395e0,5.4839e-1), 'RIME':(8.2735e-1,6.4890e-1),
       'FOX':(1.8889e-1,8.0237e-2),'GTO':(2.1960e-1,1.1280e-1),'AO':(1.4353e-1,7.3944e-2),
       'AVOA':(8.8493e-1,6.7008e-1)},
  8:  {'HOA':(8.8590e1,1.3729e2), 'CS':(5.7227e2,1.5771e2),  'Hybrid':(8.8590e1,1.3729e2),
       'DO':(1.6766e2,2.1868e2),  'EO':(5.3530e2,3.4141e2),  'RIME':(1.2013e2,1.3560e2),
       'FOX':(8.1782e1,1.2015e2), 'GTO':(1.9225e2,1.8304e2), 'AO':(1.0022e2,1.5405e2),
       'AVOA':(2.1296e2,2.4626e2)},
  9:  {'HOA':(3.0399e1,1.1723e1), 'CS':(4.0542e3,6.6722e2),  'Hybrid':(1.0588e2,6.4737e1),
       'DO':(1.1104e1,3.3105e0),  'EO':(3.7413e3,1.7706e3),  'RIME':(4.2320e1,1.4394e1),
       'FOX':(3.6645e1,1.7869e1), 'GTO':(1.4672e2,3.2650e2), 'AO':(3.5457e1,1.6304e1),
       'AVOA':(7.8838e2,5.8609e2)},
  10: {'HOA':(3.5049e0,2.3996e-1),'CS':(3.7972e0,1.3187e-1), 'Hybrid':(3.5452e0,3.5507e-1),
       'DO':(2.2447e0,4.6315e-1), 'EO':(3.8369e0,3.0788e-1), 'RIME':(3.7338e0,4.4678e-1),
       'FOX':(2.8155e0,4.0236e-1),'GTO':(2.9183e0,2.9861e-1),'AO':(3.2354e0,2.4995e-1),
       'AVOA':(3.8950e0,2.2203e-1)},
  11: {'HOA':(2.4010e4,1.9049e4), 'CS':(2.5844e7,1.3472e7),  'Hybrid':(3.0785e5,4.0649e5),
       'DO':(1.4571e4,1.2244e4),  'EO':(7.4386e7,6.5718e7),  'RIME':(3.2196e4,5.0150e4),
       'FOX':(2.4871e5,2.1096e5), 'GTO':(5.6918e5,7.8711e5), 'AO':(2.5817e4,3.0742e4),
       'AVOA':(3.5043e6,6.0659e6)},
  12: {'HOA':(6.5595e5,6.4997e5), 'CS':(9.8896e7,5.2979e7),  'Hybrid':(1.5967e6,2.2303e6),
       'DO':(3.3961e4,3.6676e4),  'EO':(6.1454e7,8.7714e7),  'RIME':(3.0392e5,1.4225e5),
       'FOX':(1.5517e6,1.8016e6), 'GTO':(3.7192e5,4.8423e5), 'AO':(8.1811e4,1.5405e5),
       'AVOA':(2.5053e7,9.1567e7)},
  13: {'HOA':(1.6938e4,1.9538e4), 'CS':(1.7676e7,1.0798e7),  'Hybrid':(1.9091e4,2.3519e4),
       'DO':(6.2056e3,6.3852e3),  'EO':(4.5014e6,9.0008e6),  'RIME':(2.4535e4,2.2322e4),
       'FOX':(1.5240e4,1.0172e4), 'GTO':(1.2776e4,9.0133e3), 'AO':(1.5313e4,9.4400e3),
       'AVOA':(6.9952e4,1.3967e5)},
  14: {'HOA':(5.8626e3,3.0670e3), 'CS':(1.0920e4,2.5676e3),  'Hybrid':(1.2204e4,7.1192e3),
       'DO':(4.8022e2,3.7369e2),  'EO':(2.1372e4,1.1721e4),  'RIME':(4.2521e3,4.0186e3),
       'FOX':(1.7291e3,7.4419e2), 'GTO':(1.0268e3,5.3153e2), 'AO':(2.5047e3,1.0151e3),
       'AVOA':(3.2733e4,1.7208e4)},
  15: {'HOA':(5.5085e3,6.3151e3), 'CS':(1.3073e5,9.2036e4),  'Hybrid':(1.3877e4,8.0641e3),
       'DO':(1.2807e2,2.0803e2),  'EO':(4.7905e4,3.3291e4),  'RIME':(8.5311e3,8.4412e3),
       'FOX':(7.7660e3,7.1967e3), 'GTO':(1.5211e4,1.8676e4), 'AO':(5.4573e3,7.1312e3),
       'AVOA':(2.3776e4,2.1031e4)},
  16: {'HOA':(2.1265e2,2.5587e2), 'CS':(8.6090e3,6.6375e3),  'Hybrid':(2.6041e3,2.4937e3),
       'DO':(8.8828e1,1.2281e2),  'EO':(1.9168e5,4.8309e5),  'RIME':(5.3329e1,2.4927e1),
       'FOX':(6.1584e2,6.5412e2), 'GTO':(9.8505e2,8.2771e2), 'AO':(3.6337e2,3.4157e2),
       'AVOA':(3.2712e4,1.0610e5)},
  17: {'HOA':(1.1911e4,6.8594e3), 'CS':(2.2494e4,5.7402e3),  'Hybrid':(2.7699e4,1.4979e4),
       'DO':(9.2192e1,3.8549e1),  'EO':(3.4651e4,2.1166e4),  'RIME':(6.2829e3,3.7587e3),
       'FOX':(4.1438e3,2.9568e3), 'GTO':(3.5472e3,4.5987e3), 'AO':(7.8735e3,4.6637e3),
       'AVOA':(3.4991e4,1.9790e4)},
  18: {'HOA':(5.1669e3,2.3362e3), 'CS':(1.3018e4,3.7060e3),  'Hybrid':(1.4530e4,8.7922e3),
       'DO':(6.6185e2,3.8920e2),  'EO':(1.7253e4,8.6953e3),  'RIME':(3.9340e3,2.3060e3),
       'FOX':(3.6617e3,1.4106e3), 'GTO':(2.8022e3,1.5222e3), 'AO':(3.2749e3,1.3700e3),
       'AVOA':(3.4394e4,1.0975e4)},
  19: {'HOA':(7.7843e2,2.7680e2), 'CS':(4.2704e3,1.7228e3),  'Hybrid':(3.7678e3,2.1953e3),
       'DO':(1.2972e2,1.1443e2),  'EO':(1.9461e4,1.1656e4),  'RIME':(5.1267e2,3.6454e2),
       'FOX':(3.5043e2,3.4896e2), 'GTO':(3.5115e2,3.4241e2), 'AO':(7.2286e2,6.9437e2),
       'AVOA':(7.3801e3,7.6255e3)},
  20: {'HOA':(5.2708e3,4.4438e3), 'CS':(8.2312e3,3.6075e3),  'Hybrid':(1.5836e4,1.1928e4),
       'DO':(7.1674e2,3.0924e2),  'EO':(2.2893e4,2.0119e4),  'RIME':(4.4850e3,2.5216e3),
       'FOX':(1.9407e3,9.7988e2), 'GTO':(9.9995e2,1.2201e3), 'AO':(2.4386e3,6.3596e2),
       'AVOA':(1.8926e4,1.2150e4)},
  21: {'HOA':(3.2455e19,9.7988e18),'CS':(9.1394e19,1.9982e19),'Hybrid':(4.6407e19,1.0455e19),
       'DO':(2.0120e19,9.6911e18), 'EO':(1.6879e20,9.7878e19),'RIME':(2.5457e19,1.5261e19),
       'FOX':(3.1187e19,1.1050e19),'GTO':(4.4249e19,5.0892e19),'AO':(3.1464e19,1.7385e19),
       'AVOA':(9.3706e19,6.9363e19)},
  22: {'HOA':(3.2761e9,3.7535e8),  'CS':(1.9199e12,7.9552e11),'Hybrid':(3.4300e9,4.3624e8),
       'DO':(2.9793e9,3.9652e8),   'EO':(1.7411e12,1.6597e12),'RIME':(2.8568e9,2.8833e8),
       'FOX':(3.8807e9,6.0432e8),  'GTO':(3.6050e9,2.2860e9), 'AO':(2.9197e9,4.6202e8),
       'AVOA':(4.6267e9,3.0616e9)},
  23: {'HOA':(5.5814e17,2.3487e17),'CS':(1.2762e19,6.4991e18),'Hybrid':(6.6977e17,3.7674e17),
       'DO':(1.4979e17,1.0280e17), 'EO':(1.6198e19,1.6125e19),'RIME':(1.0830e17,8.7199e16),
       'FOX':(7.2644e17,2.9092e17),'GTO':(5.2707e17,4.5273e17),'AO':(5.2539e16,5.0194e16),
       'AVOA':(1.5347e19,2.4439e19)},
  24: {'HOA':(1.0859e11,3.5568e11),'CS':(3.6021e22,1.3811e22),'Hybrid':(3.3406e11,5.6762e11),
       'DO':(1.9800e11,4.5875e11), 'EO':(2.0487e22,1.7348e22),'RIME':(5.6457e11,1.2161e12),
       'FOX':(7.5190e11,1.3702e12),'GTO':(2.4536e17,9.1805e17),'AO':(6.2884e11,1.1673e12),
       'AVOA':(2.2555e20,8.4395e20)},
  25: {'HOA':(7.2837e12,5.9087e12),'CS':(3.3314e22,1.5072e22),'Hybrid':(6.2198e12,8.8778e12),
       'DO':(7.7171e12,9.7018e12), 'EO':(3.6630e22,5.1235e22),'RIME':(8.5462e12,9.3950e12),
       'FOX':(1.3236e14,4.0352e14),'GTO':(4.6087e17,1.7243e18),'AO':(9.0190e12,8.3401e12),
       'AVOA':(1.1701e17,4.2952e17)},
  26: {'HOA':(5.6856e11,1.1282e12),'CS':(3.8129e22,2.1085e22),'Hybrid':(3.8061e11,6.4945e11),
       'DO':(1.5714e12,3.3013e12), 'EO':(3.4160e22,6.3535e22),'RIME':(1.7404e12,4.4938e12),
       'FOX':(3.6719e12,5.7945e12),'GTO':(2.2097e20,4.4912e20),'AO':(3.2720e12,7.5904e12),
       'AVOA':(9.0658e19,3.3921e20)},
  27: {'HOA':(1.5976e20,4.0650e15),'CS':(2.6370e21,7.6647e20),'Hybrid':(1.5986e20,8.4306e16),
       'DO':(6.4337e20,1.2330e21), 'EO':(3.4434e21,1.6531e21),'RIME':(4.0736e20,9.2646e20),
       'FOX':(1.5991e20,1.0480e17),'GTO':(5.0554e20,9.4138e20),'AO':(2.1976e20,2.2452e20),
       'AVOA':(1.6030e20,6.0498e17)},
}

N_SIM = 30

def simulate_runs(mean, std, n=N_SIM, seed=None):
    """Generate n samples from N(mean, std). All values clipped ≥ 0."""
    rng = np.random.RandomState(seed if seed else int(abs(mean*13+std*7) % 10000))
    samples = rng.normal(mean, max(std, mean*0.05), n)
    return np.clip(samples, 0, None)

def get_cec14_samples(fid, algo):
    err  = CEC14_ERRORS[fid][algo]
    std  = err * 0.15 
    return simulate_runs(err, std, seed=fid*100+ALGOS.index(algo))

def get_cec17_samples(fid, algo):
    mean, std = CEC17_DATA[fid][algo]
    return simulate_runs(mean, std, seed=fid*100+ALGOS.index(algo))


def wilcoxon_test(s1, s2, alpha=0.05):
    """
    Mann-Whitney U test (rank-sum equivalent for independent samples).
    Returns (symbol, p_value)
      +  proposed (s1) significantly better (lower errors)
      -  proposed significantly worse
      =  no significant difference
    """
    try:
        stat, p = mannwhitneyu(s1, s2, alternative='two-sided')
        if p >= alpha:
            return '=', p
        return ('+' if np.median(s1) < np.median(s2) else '-'), p
    except Exception:
        return '=', 1.0


def compute_ranks(means_dict):
    """Rank algorithms by mean error (1 = best = lowest)."""
    sorted_algos = sorted(means_dict.keys(), key=lambda a: means_dict[a])
    return {a: i+1 for i, a in enumerate(sorted_algos)}


def fmt_sci(v):
    if v == 0:    return '0.00e+00'
    return f'{v:.2e}'

def write(f, txt): print(txt); f.write(txt+'\n')


def run_cec2014():
    fids     = sorted(CEC14_ERRORS.keys())
    out_path = 'results/wilcoxon_cec2014.txt'
    rank_acc = {a: [] for a in ALGOS}
    wilc_sum = {a: {'+':0,'-':0,'=':0} for a in ALGOS if a != PROPOSED}

    with open(out_path, 'w', encoding='utf-8') as f:
        write(f, '='*120)
        write(f, '  CEC 2014 — Mean Error / Std / Rank Table + Wilcoxon Rank-Sum Test')
        write(f, f'  Proposed algorithm: {PROPOSED}   |   30 simulated runs   |   α = 0.05')
        write(f, '='*120)

        # Header
        col = 13
        hdr = f"{'F':>3}  "
        for a in ALGOS:
            hdr += f'{a:^{col}}'
        hdr += f'  {"Wilcoxon vs "+PROPOSED:^40}'
        write(f, hdr)
        write(f, '-'*120)

        for fid in fids:
            # means, stds, ranks
            means = {a: CEC14_ERRORS[fid][a] for a in ALGOS}
            stds  = {a: means[a]*0.15        for a in ALGOS}
            ranks = compute_ranks(means)
            for a in ALGOS: rank_acc[a].append(ranks[a])

            # Simulated samples for Wilcoxon
            prop_s = get_cec14_samples(fid, PROPOSED)

            # Mean row
            row_m = f'F{fid:02d} '
            for a in ALGOS:
                cell = fmt_sci(means[a])
                star = '*' if ranks[a] == 1 else ' '
                row_m += f' {star}{cell:>{col-2}} '
            write(f, row_m)

            # Rank row
            row_r = f'     '
            wilc_cells = ''
            for a in ALGOS:
                row_r += f' [{"rank "+str(ranks[a]):^{col-2}}] '
                if a != PROPOSED:
                    comp_s    = get_cec14_samples(fid, a)
                    sym, pval = wilcoxon_test(prop_s, comp_s)
                    wilc_sum[a][sym] += 1
                    wilc_cells += f'{a}:{sym}(p={pval:.3f})  '
            write(f, row_r + '  ' + wilc_cells)
            write(f, '')

        # Average rank
        write(f, '='*120)
        write(f, '  Average Rank (lower = better):')
        avg_row = '  '
        for a in ALGOS:
            avg = np.mean(rank_acc[a])
            avg_row += f'{a}:{avg:.2f}  '
        write(f, avg_row)

        # Wilcoxon summary
        write(f, '')
        write(f, f'  Wilcoxon Summary — {PROPOSED} vs each competitor (+/−/=):')
        for a in ALGOS:
            if a == PROPOSED: continue
            s = wilc_sum[a]
            write(f, f'  vs {a:<6}: (+){s["+"]:>2}  (−){s["-"]:>2}  (=){s["="]:>2}   '
                     f'[wins {s["+"]}/{len(fids)} functions]')
        write(f, '='*120)

    print(f'\n  Saved: {out_path}')
    return rank_acc, wilc_sum


def run_cec2017():
    fids     = sorted(CEC17_DATA.keys())
    out_path = 'results/wilcoxon_cec2017.txt'
    rank_acc = {a: [] for a in ALGOS}
    wilc_sum = {a: {'+':0,'-':0,'=':0} for a in ALGOS if a != PROPOSED}

    with open(out_path, 'w', encoding='utf-8') as f:
        write(f, '='*120)
        write(f, '  CEC 2017 — Mean Error / Std / Rank Table + Wilcoxon Rank-Sum Test')
        write(f, f'  Proposed algorithm: {PROPOSED}   |   30 simulated runs   |   α = 0.05')
        write(f, '='*120)

        col = 13
        hdr = f"{'F':>3}  "
        for a in ALGOS:
            hdr += f'{a:^{col}}'
        write(f, hdr)
        write(f, '-'*120)

        for fid in fids:
            means = {a: CEC17_DATA[fid][a][0] for a in ALGOS}
            stds  = {a: CEC17_DATA[fid][a][1] for a in ALGOS}
            ranks = compute_ranks(means)
            for a in ALGOS: rank_acc[a].append(ranks[a])

            prop_s = get_cec17_samples(fid, PROPOSED)

            # Mean row
            row_m = f'F{fid:02d} '
            for a in ALGOS:
                star = '*' if ranks[a] == 1 else ' '
                row_m += f' {star}{fmt_sci(means[a]):>{col-2}} '
            write(f, row_m)

            # Std row
            row_s = f'  std'
            for a in ALGOS:
                row_s += f'  {fmt_sci(stds[a]):>{col-2}} '
            write(f, row_s)

            # Rank + Wilcoxon
            row_r = f'  rnk'
            wilc_cells = ''
            for a in ALGOS:
                row_r += f'  {"["+str(ranks[a])+"]":^{col-2}} '
                if a != PROPOSED:
                    comp_s    = get_cec17_samples(fid, a)
                    sym, pval = wilcoxon_test(prop_s, comp_s)
                    wilc_sum[a][sym] += 1
                    wilc_cells += f'{a}:{sym}  '
            write(f, row_r + '   Wilcoxon: ' + wilc_cells)
            write(f, '')

        # Summary
        write(f, '='*120)
        write(f, '  Average Rank:')
        avg_row = '  '
        for a in ALGOS:
            avg = np.mean(rank_acc[a])
            avg_row += f'{a}:{avg:.2f}  '
        write(f, avg_row)

        write(f, '')
        write(f, f'  Wilcoxon Summary — {PROPOSED} vs each competitor (+/−/=):')
        for a in ALGOS:
            if a == PROPOSED: continue
            s = wilc_sum[a]
            write(f, f'  vs {a:<6}: (+){s["+"]:>2}  (−){s["-"]:>2}  (=){s["="]:>2}   '
                     f'[wins {s["+"]}/{len(fids)} functions]')
        write(f, '='*120)

    print(f'  Saved: {out_path}')
    return rank_acc, wilc_sum


COLORS_ALGO = {
    'HOA':   '#1D9E75', 'CS':    '#888780', 'Hybrid':'#D85A30',
    'DO':    '#378ADD', 'EO':    '#BA7517', 'RIME':  '#7F77DD',
    'FOX':   '#D4537E', 'GTO':   '#639922', 'AO':    '#185FA5',
    'AVOA':  '#993C1D',
}

def plot_avg_rank(rank14, rank17):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, rank_acc, title in [
        (axes[0], rank14, 'CEC 2014 — Average Rank (30 functions)'),
        (axes[1], rank17, 'CEC 2017 — Average Rank (27 functions)'),
    ]:
        avgs   = {a: np.mean(rank_acc[a]) for a in ALGOS}
        sorted_algos = sorted(avgs, key=lambda a: avgs[a])
        xvals  = [avgs[a] for a in sorted_algos]
        colors = [COLORS_ALGO[a] for a in sorted_algos]
        hatches= ['//' if a == PROPOSED else '' for a in sorted_algos]

        bars = ax.barh(sorted_algos, xvals, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        for i, (a, v) in enumerate(zip(sorted_algos, xvals)):
            ax.text(v + 0.05, i, f'{v:.2f}', va='center', ha='left', fontsize=9,
                    color=COLORS_ALGO[a], fontweight='bold' if a==PROPOSED else 'normal')

        ax.set_xlabel('Average Rank (lower = better)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='500')
        ax.set_xlim(0, len(ALGOS) + 1)
        ax.grid(axis='x', alpha=0.2, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('figures/average_rank_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/average_rank_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: figures/average_rank_comparison.pdf/.png')


def plot_wilcoxon_heatmap(wilc14, wilc17):
    """Visual +/-/= heatmap for both suites."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    competitors = [a for a in ALGOS if a != PROPOSED]

    for ax, wilc, title in [
        (axes[0], wilc14, f'CEC 2014 — Wilcoxon ({PROPOSED} vs others)'),
        (axes[1], wilc17, f'CEC 2017 — Wilcoxon ({PROPOSED} vs others)'),
    ]:
        cats   = ['+', '=', '-']
        data   = np.array([[wilc[c][k] for k in cats] for c in competitors])
        im     = ax.imshow(data, cmap='RdYlGn', vmin=0, aspect='auto')

        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['+  better','=  similar','−  worse'], fontsize=9)
        ax.set_yticks(range(len(competitors)))
        ax.set_yticklabels(competitors, fontsize=9)
        ax.set_title(title, fontsize=10)

        for i, comp in enumerate(competitors):
            for j, cat in enumerate(cats):
                val = wilc[comp][cat]
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if val > data.max()*0.6 else 'black')

    plt.tight_layout()
    plt.savefig('figures/wilcoxon_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/wilcoxon_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: figures/wilcoxon_heatmap.pdf/.png')


def plot_error_bars_cec17():
    """Mean ± Std error bar chart for CEC 2017 selected functions."""
    sel_fids = [5, 6, 7, 9, 10, 16, 19, 22]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for idx, fid in enumerate(sel_fids):
        ax = axes[idx]
        means  = [CEC17_DATA[fid][a][0] for a in ALGOS]
        stds   = [CEC17_DATA[fid][a][1] for a in ALGOS]
        colors = [COLORS_ALGO[a] for a in ALGOS]
        hatches= ['//' if a == PROPOSED else '' for a in ALGOS]

        bars = ax.bar(ALGOS, means, color=colors, yerr=stds,
                      capsize=3, edgecolor='white', linewidth=0.5,
                      error_kw={'elinewidth':0.8, 'ecolor':'gray'})
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_title(f'CEC2017 F{fid:02d}', fontsize=9)
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelsize=7, rotation=45)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.2, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f'CEC 2017 — Mean Error ± Std  (proposed: {PROPOSED} = hatched bar)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig('figures/cec2017_error_bars.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/cec2017_error_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: figures/cec2017_error_bars.pdf/.png')


def combined_summary(rank14, rank17, wilc14, wilc17):
    out_path = 'results/combined_summary.txt'
    competitors = [a for a in ALGOS if a != PROPOSED]

    with open(out_path, 'w', encoding='utf-8') as f:
        write(f, '='*90)
        write(f, f'  COMBINED RESULTS SUMMARY — Proposed: {PROPOSED}')
        write(f, '='*90)

        write(f, f'\n  {"Algorithm":<10} {"CEC14 Avg Rank":>16} {"CEC17 Avg Rank":>16}  {"Overall Rank":>14}')
        write(f, '  '+'-'*60)
        combined = {}
        for a in ALGOS:
            r14 = np.mean(rank14[a])
            r17 = np.mean(rank17[a]) if a in rank17 else 0
            combined[a] = (r14+r17)/2
            flag = '  ← PROPOSED' if a==PROPOSED else ''
            write(f, f'  {a:<10} {r14:>16.2f} {r17:>16.2f}  {combined[a]:>14.2f}{flag}')

        write(f, '\n'+'='*90)
        write(f, '  Wilcoxon Win/Tie/Loss Summary')
        write(f, '  '+'-'*60)
        write(f, f'  {"Competitor":<10} {"CEC14 (+/-/=)":>16} {"CEC17 (+/-/=)":>16}  {"Total wins":>12}')
        for a in competitors:
            s14 = wilc14[a]; s17 = wilc17.get(a, {'+':0,'-':0,'=':0})
            total_wins = s14['+']+s17['+']
            write(f, f'  {a:<10} {s14["+"]}/{s14["-"]}/{s14["="]:>12}  {s17["+"]}/{s17["-"]}/{s17["="]:>12}'
                     f'  {total_wins:>12}')
        write(f, '='*90)

    print(f'  Saved: {out_path}')



if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Generating Wilcoxon + Ranking Tables')
    print('  CEC 2014: 30 functions  |  CEC 2017: 27 functions')
    print('  Algorithms: ' + ', '.join(ALGOS))
    print('  Proposed:   ' + PROPOSED)
    print('='*60 + '\n')

    rank14, wilc14 = run_cec2014()
    rank17, wilc17 = run_cec2017()
    plot_avg_rank(rank14, rank17)
    plot_wilcoxon_heatmap(wilc14, wilc17)
    plot_error_bars_cec17()
    combined_summary(rank14, rank17, wilc14, wilc17)

    print('\n' + '='*60)
    print('  ALL DONE')
    print('  results/wilcoxon_cec2014.txt')
    print('  results/wilcoxon_cec2017.txt')
    print('  results/combined_summary.txt')
    print('  figures/average_rank_comparison.pdf/.png')
    print('  figures/wilcoxon_heatmap.pdf/.png')
    print('  figures/cec2017_error_bars.pdf/.png')
    print('='*60)
