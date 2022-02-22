# Conv3D 算子设计文档

修订记录

| 版本 | 描述 | 时间       | 作者       |
| ---- | ---- | ---------- | ---------- |
|      | 初版 | 2021.07.02 | Yinjun.Pan |
|      |      |            |            |





## 1. Conv3D 简介

**XLA ConvND Semantic **

| **Arguments**         | **Type**                         | **Semantics**                    |
| --------------------- | -------------------------------- | -------------------------------- |
| `lhs`                 | `XlaOp`                          | rank n+2 array of inputs         |
| `rhs`                 | `XlaOp`                          | rank n+2 array of kernel weights |
| `window_strides`      | `ArraySlice<int64>`              | n-d array of kernel strides      |
| `padding`             | `ArraySlice< pair<int64,int64>>` | n-d array of (low, high) padding |
| `lhs_dilation`        | `ArraySlice<int64>`              | n-d lhs dilation factor array    |
| `rhs_dilation`        | `ArraySlice<int64>`              | n-d rhs dilation factor array    |
| `feature_group_count` | int64                            | the number of feature groups     |
| `batch_group_count`   | int64                            | the number of batch groups       |

**XLA全语义的3D卷积**
$$
[N,ID,IH,IW,IC]\ *\ [T,R,S,KCi,KCo]\ =\ [N,OD,OH,OW,KCo]\\

OD=\frac{(ID+Pad_{Top}+Pad_{Bottom}-T-(T-1)*(D_{D}-1))}{S_{D}}+1\\
OH=\frac{(IH+Pad_{Top}+Pad_{Bottom}-R-(R-1)*(D_{H}-1))}{S_{H}}+1\\
OW=\frac{(IW+Pad_{Left}+Pad_{Right}-S-(S-1)*(D_{W}-1))}{S_{W}}+1
$$

**Conv2D vs Conv3D**

<img src="_static/conv2dvs3d.png" alt="semantic" style="zoom: 50%;" />





**Conv3D 实例**

<img src="_static\conv3dcase1.png" alt="Depthwise Convolution2D CM" style="zoom: 50%;"/>

```
    image_in_man = np.linspace(1, 96, 96).reshape(1, 3, 2, 4, 4)
    # [batch, in_depth, in_channels, in_height, in_width]
    image_in_tf = image_in_man.transpose(0, 1, 3, 4, 2)
    # [batch, in_depth, in_height, in_width, in_channels].
    # shape:[1,3,4,4,2]
    weight_in_man = np.array(
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]).reshape(1, 2, 2, 3, 3)  
    weight_in_tf = weight_in_man.transpose(1, 3, 4, 2, 0)
    # [filter_depth, filter_height, filter_width, in_channels,out_channels]
    # shape: [2,3,3,2,1]    
    with tf.device("cpu:0"):
        x = tf.placeholder(dtype=tf.float32, shape=[1, 3, 4, 4, 2], name='x')
        w = tf.placeholder(dtype=tf.float32, shape=[2, 3, 3, 2, 1], name='w')
        conv = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='VALID')
        with tf.Session() as sess:
            r_in_tf = sess.run(conv, feed_dict={x: image_in_tf, w: weight_in_tf})
            # [batch, in_depth, in_height, in_width, in_channels].
            r_in_man = r_in_tf.transpose(0, 1, 4, 2, 3)
            # [batch, in_depth,in_channels,in_height, in_width].
     # shape: [1,2,2,2,1]

```

## 2.   Feature specification

根据前面XLA的描述和分析，以及sip 2.0能支持的操作，Conv3D的实现特性描述如下：

- 支持数据类型为f32, f16, bf16(暂不支持s32, s16, s8, u32, u16, u8)

- 支持Input/Output数据格式：NDHWC

- size of each dimension(DHW): 1 <= {D, H, W} <= 1024

- 支持Kenrel格式：TRS(KCi)(KCo)

- Kernel : 1 <= {T, R, S} <= 13

- Stride : 1 <= {Sd, Sh, Sw} <= 13

- Dilation : 1 <= {Dd, Dh, Dw} <= 13

- Pad :  slide times of window >= 1

- PadValue : 0

- 支持operands维度：5(3+2)

  

- 支持扩展精度实现（使用f32实现f16,bf16中间操作）

- 支持1c1s --> 1cns --> 4cns --> ncns

- 支持pull mode下任意维度处理

- 精度影响

  o  fp32 log (RxSxKCi) x 1e-3

  o  fp16 log (RxSxKCi) x 1e-2

  o  bf16 log (RxSxKCi) x 5e-2

- 算子实现采用factor集成到lib/ops中，上接hlir，下接factor
- 支持sip2.0多个target（Pavo and Dorado）
- 提供python/hlir/tops api端测试接口
- 预留fusion接口input x output




## 3. 概要设计

目前设计思路是将3D卷积分解成2D卷积去实现，所以先要考虑清楚通用Conv2的实现。

### 3.1 通用Conv2d实现

当前实现仅支持NHWC Format，为了泛化通用性，SIP kernel部分采用Intrinsic去写但会参考convgen的思路，核心部分采用VMM指令，VMM指令就是向量和矩阵相乘，[1,m] x [m,n] = [1,n]  ，需要VR的行和SMR的列之间存在对应元素乘加求和的关系，dataflow将在HiWiCiCo四个维度上切分 ，优先保证R,S维度全切，尽量多切Ci, Hi/Wi(overlap in H/W)维度数据,如下图实例所示：Ci和Co切16的倍数，RSCiCo 被slice出CixCo的矩阵装载在SMR里，VR装载input feature向量。

<img src="./_static\vmm1.png" alt="cross_group_bigKCi" style="zoom: 50%;" />

dataflow factor 伪代码

    func_(func_name, func_types, {out_hbm_type}, [&](auto args, auto results) {
    	...
    	for_(0, batches_per_sip, dim_(csb_out0, 0), [&](auto n) {
     	 for_(0, dim_(hbm_out, 1), dim_(csb_out0, 1), [&](auto h) {
      	  for_(0, dim_(hbm_out, 2), dim_(csb_out0, 2), [&](auto w) {
       	   for_(0, op_param.co, dim_(csb_out0, 3), [&](auto co) {
            for_(0, op_param.ci, dim_(csb_in0, 3), [&](auto ci) {
    			// update parameters and resources
    			...
    			// async_load_ input feature and weight
    			...
         		call_(c_func_name, ...);
        	}); // ci
        	//async_store_ output
       	   }); // co
          });  // w
         });   // h
        });    // n
    	...   
    }

### 3.2 设计方案1

将[N,D,H,W,Ci]切分成N个[D,H,W,Ci]作为Conv2D的input，将[T,R,S,Ci,Co]分成T个[R,S,Ci,Co]作为Conv2D的weight,再分别用各个卷积核对input做Conv2D,再对output做reduce.

<center class="half">
    <img src="./_static/conv3dx3.png" alt="kci_1_keq3x3_seq1x1_deq1x1_peq0" style="zoom: 100%;" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
    </div>
 </center>


```
PreProcess

Loop T：

    prepare sub_conv2d_info

    call common conv2d

PostProcess
```

### 3.3 设计方案2

直接从Conv3D语义出发，对输入[N,D,H,W,Ci]，在维度D将T个二维卷积核分别做Conv2d卷积，再将输出做reduce,如下图所示.

<center class="half">
    <img src="./_static/conv3dc2.png" alt="kci_1_keq3x3_seq1x1_deq1x1_peq0" style="zoom: 100%;" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
    </div>
 </center>

dataflow factor 伪代码需做改动

    func_(func_name, func_types, {out_hbm_type}, [&](auto args, auto results) {
    	...
    	for_(0, dim_(hbm_out, 1), Stride_d, [&](auto d) {
     	 for_(0, dim_(hbm_out, 1), dim_(csb_out0, 1), [&](auto h) {
      	  for_(0, dim_(hbm_out, 2), dim_(csb_out0, 2), [&](auto w) {
           for_(0, op_param.co, dim_(csb_out0, 3), [&](auto co) {
            for_(0, op_param.T, 1, [&](auto t) {  
             for_(0, op_param.ci, dim_(csb_in0, 3), [&](auto ci) {
    		    // update parameters and resources
    			...
    		    // async_load_ input feature and weight
    			...
         	    call_(c_func_name, ...);
             }); // ci
            }); // T
            //async_store_ output
           }); // co
          });  // w
         });   // h
        });    // n
        ...
    }

相对方案1总计算量不变，优势有下面两点。

- 减少输出数据在HBM上的缓存空间
- 减少输出buffer的DMA搬移次数

### 3. 4 预留fusion接口

做完Conv3D卷积后可能有fusion需求提高性能，如bias或激活函数，下面以bias为例参考convgen的方法如下图所示，Conv3D卷积SIP Kernel实现会被多次调用，它会通过传入的参数判断在需要做reduce时只加一次bias.

<center class="half">
    <img src="./_static/conv3d_bias.png" alt="kci_1_keq3x3_seq1x1_deq1x1_peq0" style="zoom: 100%;" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
    </div>
 </center>




### 3. 5 通用Conv2d性能优化

通用版kernerl实现主要考虑泛化，覆盖所有参数范围，适合Ci较大的场景，Ci较小需要做pad,性能会有损失，后续可以做一些分类优化

- 针对较常用的RxS==1x1,3x3,5x5,7x7在已知RxS的前提下作dataflow切分写对应的SIP kernel实现。

- 针对Ci==1作dataflow切分写对应的SIP kernel，可以通过VMAC实现。

- 针对较小的Ci作dataflow切分写对应的SIP kernel实现。

## 4. Factor 接口

Dataflow采用factor开发，使用pull mode，dma的各项操作在sip内部完成。

## 5. Non4C support

|               | N    |  Di  |  Hi  |  Wi  | ICi  |  T   |  R   |  S   | KCi  | KCo  |  N   |  Do  | Ho   | Wo   | OCo  |
| :-----------: | ---- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | ---- | ---- | ---- |
| split_support | √    |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  √   |  -   | -    | -    | -    |
|   split_num   | 4    |      |      |      |      |      |      |      |      |      |  4   |      |      |      |      |



## 6. 测试用例设计

### 6.1 场景测试

Verify patterns of conv3d arising in the typical network and other odd patterns：

- all dims small
- all dims large
- Random stress test
- asymmetrical window/stride/dialation/input
- etc. 

### 6.2 性能测试

Test patterns of conv3d arising in the typical network on zebu.

## 7. Roadmap

2021.07.05-2021.08.30  采用f32数据类型，完成Conv3D feature specific中功能支持.

2021.09.01-  支持不同数据类型，进一步调试优化.

......

Conv3D BPI and BPK: TBD

