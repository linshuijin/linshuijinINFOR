#!/usr/bin/env python
#
# Copyright 2018-2022 Enflame. All Rights Reserved.
#
# -*- coding: utf-8 -*-
import os
os.environ['ENABLE_NEW_EXECUTABLE'] = "true"
os.environ['ENABLE_SDK_MEMPOOL'] = "false"
os.environ['ENFLAME_ENABLE_EF32'] = 'true'
# os.environ['ENABLE_PAVO_NON4C'] = 'true'
# os.environ['ENFLAME_DEVICE_MODE'] = "ONEDEVICE_EX"

import test_op as tt
import unittest
import random
import numpy as np
import tensorflow as tf
import time
import struct
from numpy import *

FP32_MAX_DIFF_GAP = 5e-2
FP16_MAX_DIFF_GAP = 5e-2
BF16_MAX_DIFF_GAP = 5e-2

NP_DTYPE = {"float16":"float16",
            "bfloat16":tf.bfloat16.as_numpy_dtype,
            "float32":"float32"}

## Fixed test set
class TestDtuOp(tf.test.TestCase):
    def setUp(self):
        print("\n==================Conv3d BPK Test Begin===================")

    def tearDown(self):
        print("\n==================Conv3d BPK Test End=====================")
        pass

    def f32Cpu_trans_ef32Dtu(self,num,num_shape):
        loop=num_shape[0]*num_shape[1]*num_shape[2]*num_shape[3]*num_shape[4]
        new_num=num.flatten()
        for i in range(0,loop):
            a_bin_str=format(struct.unpack('!I', struct.pack('!f', new_num[i]))[0], '032b')
            a_bin_str = a_bin_str[:-12] + '0'*12
            new_num[i]=struct.unpack('!f',struct.pack('!I', int(a_bin_str, 2)))[0]
        
        return new_num.reshape(num_shape)

    def conv3d_bpk_test_case(self,
                        input_shape,
                        kernel_shape,
                        out_backprop_shape,
                        dilations=[1, 1, 1, 1, 1],
                        strides=[1, 1, 1, 1, 1],
                        padding="VALID",
                        dtype = "float32"):
        op_name = "conv3d_backprop_filter"
        case_name = "conv3d_bpk"
        MAX_DIFF_GAP = FP16_MAX_DIFF_GAP if dtype == "float16" else BF16_MAX_DIFF_GAP if dtype == "bfloat16" else FP32_MAX_DIFF_GAP
        # a = np.random.uniform(size=input_shape, low=-1.0, high=1.0).astype(NP_DTYPE[dtype])
        # o = np.random.uniform(size=out_backprop_shape, low=-1.0, high=1.0).astype(NP_DTYPE[dtype])
        a = np.random.randint(low = 1,high = 2, size = input_shape).astype(NP_DTYPE[dtype]) + 0.125
        o = np.random.randint(low = 1,high = 2, size = out_backprop_shape).astype(NP_DTYPE[dtype]) + 0.125
        # a = self.f32Cpu_trans_ef32Dtu(a,input_shape)
        # o = self.f32Cpu_trans_ef32Dtu(o,out_backprop_shape)

        # debug
        # np.set_printoptions(precision=6,threshold=np.inf)

        with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
            input = tf.placeholder(dtype, shape=input_shape, name='input')
            out_backprop = tf.placeholder(dtype, shape=out_backprop_shape, name='out_backprop')
            conv = tf.nn.conv3d_backprop_filter(input=tf.cast(input, "float32"), filter_sizes=kernel_shape, out_backprop=tf.cast(out_backprop, "float32"), strides=strides, dilations=dilations, padding=padding)
            conv = tf.cast(conv, dtype)
            feed_dict = {input:a, out_backprop:o}
            init = tf.initialize_all_variables()
            print("################################## cpu")
            with tf.Session() as sess:
                sess.run(init)
                r_in_tf = sess.run(conv, feed_dict)
                r_in_man = r_in_tf
                print("conv3d_shape:input",a.shape)
                print("conv3d_shape:out_backprop",o.shape)
                print("conv3d_shape:cpu_output", r_in_man.shape)
                # print("cpu_in_man", r_in_man)
            print("################################## cpu end")
        with tf.device("/job:localhost/replica:0/task:0/device:XLA_DTU:0"):
            input = tf.placeholder(dtype, shape=input_shape, name='input')
            out_backprop = tf.placeholder(dtype, shape=out_backprop_shape, name='out_backprop')
            conv = tf.nn.conv3d_backprop_filter(input=input, filter_sizes=kernel_shape, out_backprop=out_backprop, strides=strides, dilations=dilations, padding=padding)
            feed_dict = {input:a, out_backprop:o}
            init = tf.initialize_all_variables()
            print("################################## dtu")
            with tf.Session() as sess:
                sess.run(init)
                dtu_res = sess.run(conv, feed_dict)
                dtu_in_man = dtu_res
                print("conv3d_shape:dtu_output", dtu_in_man.shape)
                # print("dtu_in_man", dtu_in_man)
        self.assertAllCloseAccordingToType(dtu_in_man, r_in_man, rtol=MAX_DIFF_GAP, atol=MAX_DIFF_GAP)
    def conv3d_bpk_random(self):
        op_name = "conv3d_backprop_filter"
        case_name = "conv3d_bpk"
        dtype = np.random.choice(["float32"]) # float32
        MAX_DIFF_GAP = FP16_MAX_DIFF_GAP if dtype == "float16" else BF16_MAX_DIFF_GAP if dtype == "bfloat16" else FP32_MAX_DIFF_GAP
        bpe = 2 if dtype == "float16" else 2 if dtype == "bfloat16" else 4
        MAX_ELEMENT = 1024 * 1024 * 30
        element = MAX_ELEMENT
        count = 1
        while element >= MAX_ELEMENT:
            Ci = random.randint(1, 256)
            T = random.randint(1, 7)
            R = random.randint(1, 7)
            S = random.randint(1, 7)
            DT = random.randint(1, 7)
            DR = random.randint(1, 7)
            DS = random.randint(1, 7)
            SD = random.randint(1, 4)
            SH = random.randint(1, 4)
            SW = random.randint(1, 4)
            # DT = 1
            # DR = 1
            # DS = 1
            # SD = 1
            # SH = 1
            # SW = 1

            AT = (T - 1) * DT + 1
            AR = (R - 1) * DR + 1
            AS = (S - 1) * DS + 1
            MD = max(DT, DR, DS)
            MT = (T - 1) * MD + 1
            MR = (R - 1) * MD + 1
            MS = (S - 1) * MD + 1
            Di = min(MT + random.randint(0, 249), 256)
            Hi = min(MR + random.randint(0, 249), 256)
            Wi = min(MS + random.randint(0, 249), 256)
            N = random.randint(2, 256)
            # N = 24

            element = N * Di * Hi * Wi * Ci
            PD = np.random.choice([1,2],1,p=[0.5,0.5])[0]
            # PD = 2
            if PD == 1:
                paddings = "VALID"
                Do = (Di + SD - AT) // SD
                Ho = (Hi + SH - AR) // SH
                Wo = (Wi + SW - AS) // SW
            else:
                paddings = "SAME"
                # Do = (Di - AT  + Pad_head + Pad_tail)/ SD + 1,
                # Pad_head + Pad_tail = AT -1,
                # so, Do = (Di + SD - 1) / SD .
                Do = (Di + SD - 1) // SD
                Ho = (Hi + SH - 1) // SH
                Wo = (Wi + SW - 1) // SW
            if (N * Do * Ho * Wo) > 0 :
                CM = MAX_ELEMENT // (N * Do * Ho * Wo)
            else:
                CM = 0
            Co = random.randint(1, 256)
            output_ele = N * Do * Ho * Wo * Co
            if CM < 1 :
                element = MAX_ELEMENT
            if Co > CM :
                element = MAX_ELEMENT
            if AT > 7 :
                 element = MAX_ELEMENT
            if AR > 7 :
                 element = MAX_ELEMENT
            if AS > 7 :
                 element = MAX_ELEMENT

            count = count + 1
            continue

        # debug
        # N = 196
        # Di = 2
        # Hi = 6
        # Wi = 228
        # Ci = 56
        # Do = 1
        # Ho = 1
        # Wo = 57
        # Co = 198
        # T = 2
        # R = 6
        # S = 2
        # DT = 1
        # DR = 1
        # DS = 1
        # SD = 3
        # SH = 4
        # SW = 4
        # paddings = "VALID"


        # print("random_test:count  = ", count)
        print("random_test:paddings  = ", paddings)
        print("random_test:max_ele     = ", MAX_ELEMENT)
        print("random_test:input_ele   = ", element)
        # print("random_test:output_ele  = ", output_ele)
        # print("random_test:CM  = ", CM)
        # print("random_test:N  = ", N)
        # print("random_test:Di = ", Di)
        # print("random_test:Hi = ", Hi)
        # print("random_test:Wi = ", Wi)
        # print("random_test:Ci = ", Ci)
        # print("random_test:Do = ", Do)
        # print("random_test:Ho = ", Ho)
        # print("random_test:Wo = ", Wo)
        # print("random_test:Co = ", Co)
        # print("random_test:T = ", T)
        # print("random_test:R = ", R)
        # print("random_test:S = ", S)
        # print("random_test:DT = ", DT)
        # print("random_test:DR = ", DR)
        # print("random_test:DS = ", DS)
        # print("random_test:SD = ", SD)
        # print("random_test:SH = ", SH)
        # print("random_test:SW = ", SW)
        dilation = [1, DT, DR, DS, 1]
        stride = [1, SD, SH, SW, 1]
        input_shape = [N, Di, Hi, Wi, Ci]
        kernel_shape = [T, R, S, Ci, Co]
        out_shape = [N, Do, Ho, Wo, Co]
        cpu_res = []
        dtu_res = []

        # for float32
        a = np.random.uniform(size=input_shape, low=-1.0, high=1.0).astype(NP_DTYPE[dtype]) + 0.125
        o = np.random.uniform(size=out_shape, low=-1.0, high=1.0).astype(NP_DTYPE[dtype]) + 0.125
        # a = self.f32Cpu_trans_ef32Dtu(a,input_shape)
        # o = self.f32Cpu_trans_ef32Dtu(o,out_shape)
        # a = np.random.randint(low = 1,high = 2, size = input_shape).astype(NP_DTYPE[dtype])
        # o = np.random.randint(low = 1,high = 2, size = out_shape).astype(NP_DTYPE[dtype])
        # print("input",a)
        # print("out_backprop",o)

        with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
            input = tf.placeholder(dtype, shape=[N, Di, Hi, Wi, Ci], name='input')
            out_backprop = tf.placeholder(dtype, shape=[N, Do, Ho, Wo, Co], name='out_backprop')
            conv = tf.nn.conv3d_backprop_filter(input=input, filter_sizes=kernel_shape, out_backprop=out_backprop, strides=stride, dilations=dilation, padding=paddings)
            feed_dict = {input:a, out_backprop:o}
            init = tf.initialize_all_variables()
            print("################################## cpu")
            with tf.Session() as sess:
                sess.run(init)
                r_in_tf = sess.run(conv, feed_dict)
                r_in_man = r_in_tf
                # print("conv3d_shape:input",a.shape)
                # print("conv3d_shape:out_backprop",o.shape)
                # print("conv3d_shape:cpu_output", r_in_man.shape)
                # print("r_in_man",r_in_man)
            print("################################## cpu end")
        with tf.device("/job:localhost/replica:0/task:0/device:XLA_DTU:0"):
            input = tf.placeholder(dtype, shape=[N, Di, Hi, Wi, Ci], name='input')
            out_backprop = tf.placeholder(dtype, shape=[N, Do, Ho, Wo, Co], name='out_backprop')
            conv = tf.nn.conv3d_backprop_filter(input=input, filter_sizes=kernel_shape, out_backprop=out_backprop, strides=stride, dilations=dilation, padding=paddings)
            feed_dict = {input:a, out_backprop:o}
            init = tf.initialize_all_variables()
            print("################################## dtu")
            with tf.Session() as sess:
                sess.run(init)
                dtu_res = sess.run(conv, feed_dict)
                dtu_in_man = dtu_res
                # print("conv3d_shape:dtu_output", dtu_in_man.shape)
                # print("dtu_in_man",dtu_in_man)
        self.assertAllCloseAccordingToType(dtu_in_man, r_in_man, rtol=MAX_DIFF_GAP, atol=MAX_DIFF_GAP)

    # ############################# random test ############################
    # @tt.show_test_case_name
    def test_conv3d_ndhwc_random(self):
        self.conv3d_bpk_random()

    # def test_conv3d_bpk_valid_case0(self):
    #     self.conv3d_bpk_test_case(input_shape = [3, 3, 106, 12, 134], kernel_shape = [2, 1, 3, 134, 173], out_backprop_shape = (3, 1, 53, 5, 173),
    #                               dilations=[1, 1, 1, 1, 1], strides = [1, 2, 2, 2, 1], padding = "VALID", dtype = "float32")

    # def test_conv3d_bpk_valid_case1(self):
    #     self.conv3d_bpk_test_case(input_shape = [11, 32, 19, 27, 102], kernel_shape = [7, 3, 1, 102, 158], out_backprop_shape = (11, 26, 15, 27, 158),
    #                               dilations=[1, 1, 2, 5, 1], strides = [1, 1, 1, 1, 1], padding = "VALID", dtype = "float32")

    # def test_conv3d_bpk_same_case0(self):
    #     self.conv3d_bpk_test_case(input_shape = [32, 14, 14, 14, 192], kernel_shape = [3, 3, 3, 192, 32], out_backprop_shape = (32, 14, 14, 14, 32),
    #                               dilations=[1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1], padding = "SAME", dtype = "float32")

    # def test_conv3d_bpk_same_case1(self):
    #     self.conv3d_bpk_test_case(input_shape = [32, 14, 14, 14, 96], kernel_shape = [3, 3, 3, 96, 32], out_backprop_shape = (32, 7, 7, 7, 32),
    #                               dilations=[1, 1, 1, 1, 1], strides = [1, 2, 2, 2, 1], padding = "SAME", dtype = "float32")

if __name__ == "__main__":

  # run the test
  unittest.main()
