#!/usr/bin/env python
# encoding: utf-8
"""
@author: wayne
@file: 123.py
@time: 2018/11/9 11:17
"""
"""
图像语义分割标签图annotations，针对每个像素点标记了类别，下面的代码可以看到具体情况。
"""
import numpy as np
import scipy.misc as misc

#filename = r"D:\FCN\tensorflow-FCNS\FCN.tensorflow\Data_zoo\MIT_SceneParsing\ADEChallengeData2016\annotations\training\ADE_train_00000008.png"
filename = r"D:\FCN\caffe-fcn\label\a\000001_json\label.png"
image = misc.imread(filename)
resize_image = misc.imresize(image,[224, 224], interp='nearest')
new_image = np.expand_dims(resize_image, axis=2)
print(image.shape)
print(image)

