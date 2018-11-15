#!/usr/bin/env python
# encoding: utf-8
"""
@author: wayne
@file: conv_transpose_valid.py
@time: 2018/11/12 10:30
"""

import tensorflow as tf
import numpy as np


"""
conv2d_transpose(
    x,
    filter,
    output_shape,
    strides,
    padding='SAME',
    data_format='NHWC',
    name=None
)
#参数解释
filter: [kernel_size,kernel_size, output_channels, input_channels]  这里的转置卷积核和正向卷积核有区别，在于通道参数的放置位置

#实现过程
#Step 1 扩充: 将 inputs 进行填充扩大。扩大的倍数与strides有关。扩大的方式是在元素之间插strides - 1 个 0。padding = "VALID"时，在插完值后继续在周围填充[2*（kenel_size-1）],填充值为0
             padding = "SAME"时，在插完值后根据output尺寸进行填充,填充值为0
#Step 2 卷积: 对扩充变大的矩阵，用大小为kernel_size卷积核做卷积操作，这样的卷积核有filters个，并且这里的步长为1(与参数strides无关，一定是1)
#注意：conv2d_transpose会计算output_shape能否通过给定的filter,strides,padding计算出inputs的维度，如果不能，则报错。
        也就是说，conv2d_transpose中的filter,strides,padding参数，与反过程中的conv2d的参数相同。
"""
# 输入：1张图片，尺寸64 64 高宽，通道数3
x = np.ones((1, 28, 28, 3), dtype=np.float32)
# 卷积核尺寸4x4 ，5表输出通道数，3代表输入通道数
w = np.ones((4, 4, 5, 3), dtype=np.float32)
output = tf.nn.conv2d_transpose(x, w, (1, 56, 56, 5), [1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    m = sess.run(output)
    print(m.shape)
