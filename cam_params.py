# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 00:11:52 2021
The code was taken and modified from the following project on GitHub"
https://github.com/yitao-yu/PythonORBSlAM
"""
import numpy as np

K = np.array([[872.33317543, 0., 297.0579428], [0., 872.97305203, 248.74854805], [0., 0., 1.]])

paramsdic = {"K": K,
             "width": 600, "height": 500,
             "lowlight": False,
             "featureextract": "evenCorners",
             "maxfeatures": 2000, "n": 4,
             "maxdepth": 500,
             "rvec": np.array([[-0.1746265], [-0.21844269], [-3.10739415]]),
             "tvec": np.array([[3.69725541], [2.26350912], [18.53834925]]),
             "coeff": np.array([-3.96677113e-01, 4.39993944e+00, 7.17169373e-03, 6.11277165e-03, -2.16571859e+01])
             }
