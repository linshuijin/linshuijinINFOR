#!/usr/bin/env python
#
# Copyright 2020-2021 Enflame. All Rights Reserved.
#
# -*- coding: utf-8 -*-
import os
os.environ['ENABLE_NEW_EXECUTABLE'] = "true"
os.environ['ENABLE_SDK_MEMPOOL'] = "false"
os.environ['ENFLAME_ENABLE_EF32'] = 'true'

import test_op as tt
import unittest
import random
import numpy as np
import tensorflow as tf
import time
import struct
from numpy import *

FP32_MAX_DIFF_GAP = 1e-3
FP16_MAX_DIFF_GAP = 1e-2
BF16_MAX_DIFF_GAP = 1e-2

NP_DTYPE = {"float16":"float16",
            "bfloat16":tf.bfloat16.as_numpy_dtype,
            "float32":"float32"}

## Fixed test set
class TestDtuOp(tf.test.TestCase):
    def setUp(self):
        print("\n==================Conv3d FF Test Begin===================")

    def tearDown(self):
        print("\n==================Conv3d FF Test End=====================")
        pass

    def f32Cpu_trans_ef32Dtu(self,num,num_shape):
        loop=num_shape[0]*num_shape[1]*num_shape[2]*num_shape[3]*num_shape[4]
        new_num=num.flatten()
        for i in range(0,loop):
            a_bin_str=format(struct.unpack('!I', struct.pack('!f', new_num[i]))[0], '032b')
            a_bin_str = a_bin_str[:-12] + '0'*12
            new_num[i]=struct.unpack('!f',struct.pack('!I', int(a_bin_str, 2)))[0]
        
        return new_num.reshape(num_shape)
    
    def conv3d_test_case(self,
                     input_shape,
                     kernel_shape,
                     dilations=[1, 1, 1, 1, 1],
                     strides=[1, 1, 1, 1, 1],
                     padding="VALID",
                     dtype = "float32"):
        op_name = "conv3d"
        case_name = "conv3d_ff_general_test"
        # a = np.random.uniform(size=input_shape, low=-1.0, high=1.0).astype(dtype)
        # a=self.f32Cpu_trans_ef32Dtu(a,input_shape)
        # k = np.random.uniform(size=kernel_shape, low=-1.0, high=1.0).astype(dtype)
        # k=self.f32Cpu_trans_ef32Dtu(k,kernel_shape)
        # a = np.random.random_integers(
        #     low=0, high=9, size=input_shape).astype("float32") + 0.125
        # k = np.random.random_integers(
        #     low=0, high=9, size=kernel_shape).astype("float32") + 0.125

        a = np.random.random_integers(
            low=1, high=2, size=input_shape).astype("float32") 
        k = np.random.random_integers(
            low=1, high=2, size=kernel_shape).astype("float32") 
        MAX_DIFF_GAP = FP16_MAX_DIFF_GAP if dtype == "float16" else BF16_MAX_DIFF_GAP if dtype == "bfloat16" else FP32_MAX_DIFF_GAP

        cpu_res = []
        dtu_res = []

        # for conv3d 
        with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
            input = tf.placeholder(dtype, shape=input_shape, name='input')
            kernel = tf.placeholder(dtype, shape=kernel_shape, name='kernel')
            if dtype in ["float16", "bfloat16"]:
                conv = tf.nn.conv3d(tf.cast(input, "float32"), tf.cast(kernel, "float32"), strides=strides, dilations = dilations, padding=padding)
                conv = tf.cast(conv, dtype)
                feed_dict = {input:a, kernel:k}
                init = tf.initialize_all_variables()
                print("################################## cpu")
                with tf.Session() as sess:
                    sess.run(init)
                    r_in_tf = sess.run(conv, feed_dict={input: a, kernel: k})
                    r_in_man = r_in_tf.transpose(0, 4, 1, 2, 3)
                    print("conv3d_shape:cpu_output", r_in_man.shape)
            else:
                conv = tf.nn.conv3d(input, kernel, strides=strides, dilations = dilations, padding=padding)
                feed_dict = {input:a, kernel:k}
                init = tf.initialize_all_variables()
                print("################################## cpu")
                with tf.Session() as sess:
                    sess.run(init)
                    r_in_tf = sess.run(conv, feed_dict={input: a, kernel: k})
                    r_in_man = r_in_tf.transpose(0, 4, 1, 2, 3)
                    print("conv3d_shape:cpu_output", r_in_man.shape)

            print("################################## cpu end")
        with tf.device("/job:localhost/replica:0/task:0/device:XLA_DTU:0"):
            input = tf.placeholder(dtype, shape=input_shape, name='input')
            kernel = tf.placeholder(dtype, shape=kernel_shape, name='kernel')
            conv = tf.nn.conv3d(input, kernel, strides=strides, dilations = dilations, padding=padding)
            feed_dict = {input:a, kernel:k}
            init = tf.initialize_all_variables()
            print("################################## dtu")
            with tf.Session() as sess:
                sess.run(init)
                dtu_res = sess.run(conv, feed_dict)
                # print(dtu_res.shape)
                dtu_in_man = dtu_res.transpose(0, 4, 1, 2, 3)
                # [batch, in_depth,in_channels,in_height, in_width].
                print("output_shape", dtu_in_man.shape)
                # print(dtu_in_man)
                # compare = r_in_man==dtu_in_man
                # print(compare.all())
                # print("compare_res")
        self.assertAllCloseAccordingToType(dtu_in_man, r_in_man, rtol=MAX_DIFF_GAP, atol=MAX_DIFF_GAP)

    def conv3d_ff_random(self):
        op_name = "conv3d"
        case_name = "conv3d_ff_same"
        dtype = np.random.choice(["float32"])
        MAX_DIFF_GAP = FP16_MAX_DIFF_GAP if dtype == "float16" else BF16_MAX_DIFF_GAP if dtype == "bfloat16" else FP32_MAX_DIFF_GAP
        bpe = 4
        MAX_ELEMENT = 1024 * 1024 * 1024
        element = MAX_ELEMENT
        count = 1
        while element >= MAX_ELEMENT:
            T = random.randint(1, 5)
            R = random.randint(1, 5)
            S = random.randint(1, 5)
            Ci = random.randint(1, 256)
            DT = random.randint(1, 4)
            DR = random.randint(1, 4)
            DS = random.randint(1, 4)
            SD = random.randint(1, 7)
            SH = random.randint(1, 7)
            SW = random.randint(1, 7)
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
            N = random.randint(2, 36)
            # N = 24


            element = N * Di * Hi * Wi * Ci
            PD = random.randint(1, 2)
            PD = 1
            if PD == 1:
                paddings = "VALID"
                Do = (Di + SD - AT) / SD
                Ho = (Hi + SH - AR) / SH
                Wo = (Wi + SW - AS) / SW
            else:
                paddings = "SAME"
                Do = (Di + SD - 1) / SD
                Ho = (Hi + SH - 1) / SH
                Wo = (Wi + SW - 1) / SW
            if (N * Do * Ho * Wo) > 0 :
                CM = MAX_ELEMENT / (N * Do * Ho * Wo)
            else:
                CM = 0
            Co = random.randint(1, 256)
            output_ele = N * Do * Ho * Wo * Co
            if CM < 1 :
                element = MAX_ELEMENT
            if Co > CM :
                element = MAX_ELEMENT
            # if AT > 7 :
            #      element = MAX_ELEMENT
            # if AR > 7 :
            #      element = MAX_ELEMENT
            # if AS > 7 :
            #      element = MAX_ELEMENT

            if AT > 5 :
                 element = MAX_ELEMENT
            if AR > 5 :
                 element = MAX_ELEMENT
            if AS > 5 :
                 element = MAX_ELEMENT
               
            count = count + 1
            continue

        # debug
        # N = 27
        # Di = 124
        # Hi = 45
        # Wi = 45
        # Ci = 148
        # Co = 111
        # T = 4
        # R = 3
        # S = 5
        # DT = 1
        # DR = 1
        # DS = 1
        # SD = 7
        # SH = 4
        # SW = 2
        # paddings = "VALID"

        print("random_test:count  = ", count)
        print("random_test:paddings  = ", paddings)
        print("random_test:max_ele     = ", MAX_ELEMENT)
        print("random_test:input_ele   = ", element)
        print("random_test:output_ele  = ", output_ele)
        print("random_test:CM  = ", CM)
        print("random_test:N  = ", N)
        print("random_test:Di = ", Di)
        print("random_test:Hi = ", Hi)
        print("random_test:Wi = ", Wi)
        print("random_test:Ci = ", Ci)
        print("random_test:Co = ", Co)
        print("random_test:T = ", T)
        print("random_test:R = ", R)
        print("random_test:S = ", S)
        print("random_test:DT = ", DT)
        print("random_test:DR = ", DR)
        print("random_test:DS = ", DS)
        print("random_test:SD = ", SD)
        print("random_test:SH = ", SH)
        print("random_test:SW = ", SW)
        dilation = [1, DT, DR, DS, 1]
        stride = [1, SD, SH, SW, 1]
        input_shape = [N, Di, Hi, Wi, Ci]
        kernel_shape = [T, R, S, Ci, Co]
        cpu_res = []
        dtu_res = []
        # a = np.random.uniform(size=input_shape, low=-1.0, high=1.0).astype(dtype)
        # a=self.f32Cpu_trans_ef32Dtu(a,input_shape)
        # k = np.random.uniform(size=kernel_shape, low=-1.0, high=1.0).astype(dtype)
        # k=self.f32Cpu_trans_ef32Dtu(k,kernel_shape)
        a = np.random.random_integers(
            low=0, high=9, size=input_shape).astype("float32") + 0.125
        k = np.random.random_integers(
            low=0, high=9, size=kernel_shape).astype("float32") + 0.125

        with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
        # with tf.device("cpu:0"):
            input = tf.placeholder(dtype=tf.float32, shape=[N, Di, Hi, Wi, Ci], name='input')
            kernel = tf.placeholder(dtype=tf.float32, shape=[T, R, S, Ci, Co], name='kernel')
            conv = tf.nn.conv3d(input, kernel, strides=stride, dilations=dilation, padding=paddings)
            feed_dict = {input:a, kernel:k}
            init = tf.initialize_all_variables()
            print("################################## cpu")
            with tf.Session() as sess:
                sess.run(init)
                r_in_tf = sess.run(conv, feed_dict={input: a, kernel: k})
                r_in_man = r_in_tf.transpose(0, 4, 1, 2, 3)
                print("conv3d_shape:input",a.shape)
                print("conv3d_shape:kernel",k.shape)
                print("conv3d_shape:cpu_output", r_in_man.shape)
                # print(r_in_man)
            print("################################## cpu end")
        with tf.device("/job:localhost/replica:0/task:0/device:XLA_DTU:0"):
            input = tf.placeholder(dtype=tf.float32, shape=[N, Di, Hi, Wi, Ci], name='input')
            kernel = tf.placeholder(dtype=tf.float32, shape=[T, R, S, Ci, Co], name='kernel')
            conv = tf.nn.conv3d(input, kernel, strides=stride, dilations=dilation, padding=paddings)
            feed_dict = {input:a, kernel:k}
            init = tf.initialize_all_variables()
            print("################################## dtu")
            with tf.Session() as sess:
                sess.run(init)
                dtu_res = sess.run(conv, feed_dict)
                dtu_in_man = dtu_res.transpose(0, 4, 1, 2, 3)
                print("conv3d_shape:dtu_output", dtu_in_man.shape)
                # print(dtu_in_man)
                # print("a:")
                # print(a)
                # print("k:")
                # print(k)
        self.assertAllCloseAccordingToType(dtu_in_man, r_in_man, rtol=MAX_DIFF_GAP, atol=MAX_DIFF_GAP)

    def conv3d_ff_random_16(self):
        op_name = "conv3d"
        case_name = "conv3d_ff_same"
        dtype = np.random.choice(["float16", "bfloat16"]) # float32 float16 bfloat16
        MAX_DIFF_GAP = FP16_MAX_DIFF_GAP if dtype == "float16" else BF16_MAX_DIFF_GAP if dtype == "bfloat16" else FP32_MAX_DIFF_GAP
        bpe = 2 if dtype == "float16" else 2 if dtype == "bfloat16" else 4
        MAX_ELEMENT = 1024 * 1024 * 1024
        element = MAX_ELEMENT
        count = 1
        while element >= MAX_ELEMENT:
            Ci = random.randint(1, 256)
            T = random.randint(1, 7)
            R = random.randint(1, 7)
            S = random.randint(1, 7)
            SD = random.randint(1, 7)
            SH = random.randint(1, 7)
            SW = random.randint(1, 7)
            Di  = T + random.randint(0, 249)
            Hi = R + random.randint(0, 249)
            Wi = S + random.randint(0, 249)
            N  = random.randint(2, 36)
            element = N * Di * Hi * Wi * Ci
            Do = 1 + (Di - T) / SD
            Ho = 1 + (Hi - R) / SH
            Wo = 1 + (Wi - S) / SW
            CB = 1073741824 / (N * Do * Ho * Wo)
            if CB < 1 :
                element = MAX_ELEMENT
            count = count + 1
            continue
        CB = min(256,CB)
        Co = random.randint(1, CB)    
        NB = random.randint(1, 2)
        N = N / NB
        PD = random.randint(1, 2)
        if PD == 1:
            paddings = "VALID"
        else:
            paddings = "SAME"
        print("random_test:count  = ", count)
        print("random_test:paddings  = ", paddings)
        print("random_test:max_ele     = ", MAX_ELEMENT)
        print("random_test:element  = ", element)
        print("random_test:CB  = ", CB)
        print("random_test:NB  = ", NB)
        print("random_test:N  = ", N)
        print("random_test:Di = ", Di)
        print("random_test:Hi = ", Hi)
        print("random_test:Wi = ", Wi)
        print("random_test:Ci = ", Ci)
        print("random_test:Co = ", Co)
        print("random_test:T = ", T)
        print("random_test:R = ", R)
        print("random_test:S = ", S)
        print("random_test:SD = ", SD)
        print("random_test:SH = ", SH)
        print("random_test:SW = ", SW)
        dilations = [1, 1, 1, 1, 1]
        stride = [1, SD, SH, SW, 1]
        input_shape = [N, Di, Hi, Wi, Ci]
        kernel_shape = [T, R, S, Ci, Co]
        cpu_res = []
        dtu_res = []

        # for bf16 and fp16
        a = np.random.uniform(size=input_shape, low=-1.0, high=1.0).astype(NP_DTYPE[dtype])
        k = np.random.uniform(size=kernel_shape, low=-1.0, high=1.0).astype(NP_DTYPE[dtype])

        with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
            input = tf.placeholder(dtype, shape=[N, Di, Hi, Wi, Ci], name='input')
            kernel = tf.placeholder(dtype, shape=[T, R, S, Ci, Co], name='kernel')
            conv = tf.nn.conv3d(tf.cast(input, "float32"), tf.cast(kernel, "float32"), strides=stride, padding=paddings)
            conv = tf.cast(conv, dtype)
            feed_dict = {input:a, kernel:k}
            init = tf.initialize_all_variables()
            print("################################## cpu")
            with tf.Session() as sess:
                sess.run(init)
                r_in_tf = sess.run(conv, feed_dict)
                r_in_man = r_in_tf.transpose(0, 4, 1, 2, 3)
                print("conv3d_shape:input",a.shape)
                print("conv3d_shape:kernel",k.shape)
                print("conv3d_shape:cpu_output", r_in_man.shape)
            print("################################## cpu end")
        with tf.device("/job:localhost/replica:0/task:0/device:XLA_DTU:0"):
            input = tf.placeholder(dtype, shape=[N, Di, Hi, Wi, Ci], name='input')
            kernel = tf.placeholder(dtype, shape=[T, R, S, Ci, Co], name='kernel')
            conv = tf.nn.conv3d(input, kernel, strides=stride, padding=paddings)
            feed_dict = {input:a, kernel:k}
            init = tf.initialize_all_variables()
            print("################################## dtu")
            with tf.Session() as sess:
                sess.run(init)
                dtu_res = sess.run(conv, feed_dict)
                dtu_in_man = dtu_res.transpose(0, 4, 1, 2, 3)
                print("conv3d_shape:dtu_output", dtu_in_man.shape)
        self.assertAllCloseAccordingToType(dtu_in_man, r_in_man, rtol=MAX_DIFF_GAP, atol=MAX_DIFF_GAP)
    # ############################# random test ############################
    # @tt.show_test_case_name
    def test_conv3d_ndhwc_random(self):
        self.conv3d_ff_random()

    # def test_conv3d_ndhwc_random(self):
    #     # DT = random.randint(1, 2)
    #     DT = np.random.choice([1,2],1,p=[0.2,0.8])[0]
    #     # DT = 1
    #     if DT == 1:
    #         self.conv3d_ff_random_16()
    #     else:
    #         self.conv3d_ff_random()

    # def test_conv3d_ndhwc_valid_case1(self):
    #     self.conv3d_test_case([12, 64, 128, 128, 16], [2, 3, 4, 16, 32], strides = [1, 7, 7, 7, 1], padding = "VALID")

    # def test_conv3d_ndhwc_valid_case2(self):
    #     self.conv3d_test_case([6, 16, 256, 256, 32], [5, 5, 5, 32, 32], strides = [1, 3, 5, 7, 1], padding = "VALID")

    # def test_conv3d_ndhwc_valid_case3(self):
    #     self.conv3d_test_case([6, 32, 64, 128, 64], [3, 3, 3, 64, 32], strides = [1, 7, 5, 3, 1], padding = "VALID")

    # def test_conv3d_ndhwc_same_case1(self):
    #     self.conv3d_test_case([6, 32, 128, 128, 13], [1, 3, 5, 13, 32], strides = [1, 4, 6, 2, 1], padding = "SAME")

    # def test_conv3d_ndhwc_same_case2(self):
    #     self.conv3d_test_case([13, 16, 128, 128, 37], [5, 4, 3, 37, 32], strides = [1, 1, 2, 3, 1], padding = "SAME")

    # def test_conv3d_ndhwc_same_case3(self):
    #     self.conv3d_test_case([9, 32, 128, 128, 55], [2, 5, 3, 55, 32], strides = [1, 4, 5, 6, 1], padding = "SAME")

    # def test_conv3d_ndhwc_same_case4(self):
    #     self.conv3d_test_case([32, 4, 16, 16, 240], [3, 3, 3, 240, 240], strides = [1, 1, 1, 1, 1], padding = "SAME")

    # def test_conv3d_ndhwc_same_case4(self):
    #     self.conv3d_test_case([1, 1, 16, 16, 16], [1, 1, 1, 16, 128], strides = [1, 1, 1, 1, 1],dilations = [1, 2, 3, 4, 1], padding = "VALID")

    # def test_conv3d_ndhwc_valid_case3(self):
    #     self.conv3d_test_case([6, 32, 64, 128, 64], [3, 3, 3, 64, 32], strides = [1, 7, 5, 3, 1],dilations = [1, 2, 3, 4, 1], padding = "VALID")

    # def test_conv3d_ndhwc_valid_case3(self):
    #     self.conv3d_test_case([9, 170, 85, 42, 132], [2, 1, 1, 132, 252], strides = [1, 6, 5, 4, 1],dilations = [1, 3, 1, 1, 1], padding = "VALID")

    # def test_conv3d_ndhwc_valid_case3(self):
    #     self.conv3d_test_case([1, 1, 1, 1, 16], [1, 3, 3, 16, 128], strides = [1, 1, 1, 1, 1],dilations = [1, 1, 1, 1, 1], padding = "SAME")

    

if __name__ == "__main__":

  # run the test
  unittest.main()
