/* Copyright 2018-2022 The Enflame Tech Company. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See t-he License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <string>
#include <vector>

#include "dtu/factor/func.h"
#include "ops/common/dtu_conv3d_op.h"
#include "ops/common/dtu_conv3d_op_utils.h"
#include "ops/common/dtu_op_non4c_common.h"
#include "ops/common/dtu_op_utils.h"

namespace factor {
using namespace mlir::dtu_hlir;
using namespace dtu::op;
using namespace hlir;
using namespace conv3d_utils;

void Common20KernelFuncConv3dBPK(
    const dtu::op::DtuOpContext& context, std::string func_name,
    std::vector<Type> func_types, std::vector<Type> results_types,
    Conv3dPadStrideDilationParam& pad_stride_dilation_param,
    const dtu::op::DataType element_type, bool ef32_mode,
    std::vector<int64_t>& bitcast_out_sip_shape_vec,
    std::vector<int64_t>& bitcast_lhs_csb_shape_vec,
    std::vector<int64_t>& bitcast_rhs_csb_shape_vec) {
  EFLOG(DBG) << "Enter Common20KernelFuncConv3dBPK !!!" << std::endl;
  auto pad_head = pad_stride_dilation_param.pad_head;
  auto pad_tail = pad_stride_dilation_param.pad_tail;
  auto pad_top = pad_stride_dilation_param.pad_top;
  auto pad_bot = pad_stride_dilation_param.pad_bot;
  auto pad_left = pad_stride_dilation_param.pad_left;
  auto pad_right = pad_stride_dilation_param.pad_right;
  auto stride_d = pad_stride_dilation_param.stride_d;
  auto stride_h = pad_stride_dilation_param.stride_h;
  auto stride_w = pad_stride_dilation_param.stride_w;
  auto base_dilation_d = pad_stride_dilation_param.base_dilation_d;
  auto base_dilation_h = pad_stride_dilation_param.base_dilation_h;
  auto base_dilation_w = pad_stride_dilation_param.base_dilation_w;
  auto window_dilation_d = pad_stride_dilation_param.window_dilation_d;
  auto window_dilation_h = pad_stride_dilation_param.window_dilation_h;
  auto window_dilation_w = pad_stride_dilation_param.window_dilation_w;
  auto sip_lhs_type = func_types[10];
  auto sip_rhs_type = func_types[12];
  auto sip_out_type = func_types[14];

  // bitcast
  auto bitcast_out_sip_type = L1Type(FloatType(32), bitcast_out_sip_shape_vec);
  auto bitcast_lhs_csb_type =
      SRAMType(FloatType(32), bitcast_lhs_csb_shape_vec);
  auto bitcast_rhs_csb_type =
      SRAMType(FloatType(32), bitcast_rhs_csb_shape_vec);

  // sip kernel
  std::string c_func_name = "conv3d_bpk_fp32_kernel";
  if (element_type == dtu::op::DataType::BF16) {
    EFLOG(DBG) << "conv3d_shape: BF16:";
    c_func_name = "conv3d_bpk_bf16_kernel";
    bitcast_out_sip_type = L1Type(BFloatType(), bitcast_out_sip_shape_vec);
    bitcast_lhs_csb_type = SRAMType(BFloatType(), bitcast_lhs_csb_shape_vec);
    bitcast_rhs_csb_type = SRAMType(BFloatType(), bitcast_rhs_csb_shape_vec);
  } else if (element_type == dtu::op::DataType::F16) {
    EFLOG(DBG) << "conv3d_shape: F16:";
    c_func_name = "conv3d_bpk_fp16_kernel";
    bitcast_out_sip_type = L1Type(FloatType(16), bitcast_out_sip_shape_vec);
    bitcast_lhs_csb_type = SRAMType(FloatType(16), bitcast_lhs_csb_shape_vec);
    bitcast_rhs_csb_type = SRAMType(FloatType(16), bitcast_rhs_csb_shape_vec);
  }
  if (element_type == dtu::op::DataType::F32 && ef32_mode == true) {
    EFLOG(DBG) << "conv3d_shape: EF32 element_type:" << element_type;
    c_func_name = "conv3d_bpk_ef32_kernel";
  }

  std::vector<Type> c_func_types{sip_lhs_type,
                                 sip_rhs_type,
                                 sip_out_type,
                                 IntType(32) /*n*/,
                                 IntType(32) /*hi*/,
                                 IntType(32) /*wi*/,
                                 IntType(32) /*ci*/,
                                 IntType(32) /*r*/,
                                 IntType(32) /*s*/,
                                 IntType(32) /*co*/,
                                 IntType(32) /*ho*/,
                                 IntType(32) /*wo*/,
                                 IntType(32) /*stride_h*/,
                                 IntType(32) /*stride_w*/,
                                 IntType(32) /*window_dilation_h*/,
                                 IntType(32) /*window_dilation_w*/,
                                 IntType(32) /*ld_flag*/,
                                 IntType(32) /*st_flag*/};

//   c_func_(c_func_name, c_func_types, {},
// #include "lib/ops/common/c_sources/conv3d_bpk_general.inc"
//   );  // NOLINT
if (element_type == dtu::op::DataType::F32 && ef32_mode == false) {
    c_func_(c_func_name, c_func_types, {},
#include "lib/ops/common/c_sources/conv3d_bpk_general.inc"
            );  // NOLINT
  } else if (element_type == dtu::op::DataType::F32 && ef32_mode == true) {
    c_func_(c_func_name, c_func_types, {},
#include "lib/ops/common/c_sources/conv3d_bpk_ef32.inc"
            );  // NOLINT
  }

  func_(func_name, func_types, results_types, [&](auto args, auto results) {
    // this implemetation is for split in nhw, not c
    int args_idx = 0;
    auto hbm_lhs = args[args_idx++];
    auto hbm_rhs = args[args_idx++];

    auto csb_lhs0 = args[args_idx++];
    auto csb_lhs1 = args[args_idx++];
    auto csb_lhs2 = args[args_idx++];
    auto csb_lhs3 = args[args_idx++];
    auto csb_rhs0 = args[args_idx++];
    auto csb_rhs1 = args[args_idx++];
    auto csb_out0 = args[args_idx++];
    auto csb_out1 = args[args_idx++];

    auto sip_lhs0 = args[args_idx++];
    auto sip_lhs1 = args[args_idx++];
    auto sip_rhs0 = args[args_idx++];
    auto sip_rhs1 = args[args_idx++];
    auto sip_out0 = args[args_idx++];
    auto sip_out1 = args[args_idx++];

    auto lhs_cdma0 = args[args_idx++];
    auto lhs_cdma1 = args[args_idx++];
    auto rhs_cdma0 = args[args_idx++];
    auto rhs_cdma1 = args[args_idx++];
    auto out_cdma = args[args_idx++];

    auto lhs_sdma0 = args[args_idx++];
    auto lhs_sdma1 = args[args_idx++];
    auto lhs_sdma2 = args[args_idx++];
    auto lhs_sdma3 = args[args_idx++];
    auto rhs_sdma0 = args[args_idx++];
    auto rhs_sdma1 = args[args_idx++];
    auto out_sdma = args[args_idx++];
    auto n_loop_all = args[args_idx++];
    auto do_loop_num = args[args_idx++];
    auto ho_loop_num = args[args_idx++];
    auto wo_loop_num = args[args_idx++];
    auto ci_loop_num = args[args_idx++];
    auto t_loop_num = args[args_idx++];
    auto r_loop_num = args[args_idx++];
    auto s_loop_num = args[args_idx++];
    auto co_loop_num = args[args_idx++];

    auto n_loop_offset_this_sip = args[args_idx++];
    auto n_loop_num = args[args_idx++];

    // auto total_loop_num = n_loop_num * d_loop_num * ho_loop_num * wo_loop_num
    // *
    //                       ci_loop_num * t_loop_num * r_loop_num * s_loop_num
    //                       * co_loop_num;

    // auto actual_loop_num =
    //     total_loop_num - d_loop_num * ho_loop_num * wo_loop_num * r_loop_num
    //     * (pad_head + pad_tail); ?
    auto sip_idx = args[args_idx++];

    auto hbm_out = results[0];

    auto t = dim_(sip_rhs0, 0);
    auto r = dim_(sip_rhs0, 1);
    auto s = dim_(sip_rhs0, 2);

    EFLOG(INFO) << "args_idx=" << args_idx;

    // [1,d,ho,wo,co]
    auto cur_n_offset = n_loop_offset_this_sip;
    auto cur_do_offset = 0;
    auto cur_ho_offset = 0;
    auto cur_wo_offset = 0;
    auto cur_ci_offset = 0;
    auto cur_t_offset = 0;
    auto cur_r_offset = 0;
    auto cur_s_offset = 0;
    auto cur_co_offset = 0;

    auto actual_t = (t - 1) * window_dilation_d + 1;
    auto actual_r = (r - 1) * window_dilation_h + 1;
    auto actual_s = (s - 1) * window_dilation_w + 1;

    // how many valid ele in the front
    auto cur_d_offset = cur_do_offset * stride_d - pad_head + cur_t_offset;
    auto cur_h_offset = cur_ho_offset * stride_h - pad_top + cur_r_offset;
    auto cur_w_offset = cur_wo_offset * stride_w - pad_left + cur_s_offset;
    auto cur_d_offset_end =
        ((dim_(csb_out0, 0) + cur_do_offset - 1) * stride_d + cur_t_offset +
         actual_t - pad_head) -
        1;
    auto cur_h_offset_end =
        ((dim_(csb_out0, 1) + cur_ho_offset - 1) * stride_h + cur_r_offset +
         actual_r - pad_top) -
        1;
    auto cur_w_offset_end =
        ((dim_(csb_out0, 2) + cur_wo_offset - 1) * stride_w + cur_s_offset +
         actual_s - pad_left) -
        1;
    auto cur_ci_offset_end = dim_(csb_rhs0, 0) + cur_ci_offset - 1;
    auto cur_n_offset_end = dim_(csb_lhs0, 4) + cur_n_offset - 1;

    auto cur_d_len0 = select_(cur_d_offset_end > dim_(hbm_lhs, 1) - 1,
                              dim_(hbm_lhs, 1) - cur_d_offset,
                              cur_d_offset_end - cur_d_offset + 1);
    auto cur_d_len = select_(cur_d_offset_end < 0, 0, cur_d_len0);
    auto cur_h_len0 = select_(cur_h_offset_end > dim_(hbm_lhs, 2) - 1,
                              dim_(hbm_lhs, 2) - cur_h_offset,
                              cur_h_offset_end - cur_h_offset + 1);
    auto cur_h_len = select_(cur_h_offset_end < 0, 0, cur_h_len0);
    auto cur_w_len0 = select_(cur_w_offset_end > dim_(hbm_lhs, 3) - 1,
                              dim_(hbm_lhs, 3) - cur_w_offset,
                              cur_w_offset_end - cur_w_offset + 1);
    auto cur_w_len = select_(cur_w_offset_end < 0, 0, cur_w_len0);
    auto cur_ci_len = select_(cur_ci_offset_end > dim_(hbm_lhs, 0) - 1,
                              dim_(hbm_lhs, 0) - cur_ci_offset,
                              cur_ci_offset_end - cur_ci_offset + 1);
    auto cur_n_len = select_(cur_n_offset_end > dim_(hbm_lhs, 4) - 1,
                             dim_(hbm_lhs, 4) - cur_n_offset,
                             cur_n_offset_end - cur_n_offset + 1);

    auto cur_block_pad_top = cur_h_offset + pad_top - cur_ho_offset * stride_h;
    auto cur_block_pad_left =
        cur_w_offset + pad_left - cur_wo_offset * stride_w;

    auto cur_block_pad_bot = (dim_(csb_out0, 1) - 1) * stride_h + actual_r -
                             (cur_h_len - 1) - 1 - cur_block_pad_top;
    auto cur_block_pad_right = (dim_(csb_out0, 2) - 1) * stride_w + actual_s -
                               (cur_w_len - 1) - 1 - cur_block_pad_left;

    auto cur_lhs_cdma = lhs_cdma0;
    auto cur_lhs_sdma = lhs_sdma0;
    auto cur_lhs_sdma_pad = lhs_sdma2;
    auto cur_rhs_cdma = rhs_cdma0;
    auto cur_rhs_sdma = rhs_sdma0;

    auto cur_csb_lhs = csb_lhs0;
    auto cur_csb_lhs_pad = csb_lhs2;
    auto cur_csb_rhs = csb_rhs0;
    auto cur_csb_out = csb_out0;
    auto cur_sip_lhs = sip_lhs0;
    auto cur_sip_rhs = sip_rhs0;
    auto cur_sip_out = sip_out0;
    // print_("cur_n_offset:", {cur_n_offset});
    // print_("stride_w:", {stride_w});
    // print_("actual_s:", {actual_s});
    // print_("cur_ci_len:", {cur_ci_len});
    // print_("cur_d_len:", {cur_d_len});
    // print_("cur_h_len:", {cur_h_len});
    // print_("cur_w_len:", {cur_w_len});
    // print_("cur_n_len:", {cur_n_len});
    // print_("dim_(csb_out0, 2):", {dim_(csb_out0, 2)});
    // print_("cur_block_pad_top:", {cur_block_pad_top});
    // print_("cur_block_pad_left:", {cur_block_pad_left});
    // print_("cur_block_pad_bot:", {cur_block_pad_bot});
    // print_("cur_block_pad_right:", {cur_block_pad_right});
    // print_("cur_d_offset:", {cur_d_offset});
    // print_("cur_h_offset:", {cur_h_offset});
    // print_("cur_w_offset:", {cur_w_offset});
    // print_("cur_ho_offset:", {cur_ho_offset});
    // print_("cur_wo_offset:", {cur_wo_offset});
    // print_("cur_d_offset_end:", {cur_d_offset_end});
    // print_("cur_h_offset_end:", {cur_h_offset_end});
    // print_("cur_w_offset_end:", {cur_w_offset_end});
    // print_("cur_s_offset:", {cur_s_offset});
    // print_("n_loop_offset_this_sip:", {n_loop_offset_this_sip});

    std::vector<Value> cur_offset = {0, 0, 0, 0, 0};
    std::vector<Value> cur_padding_low = {0, 0, cur_block_pad_top,
                                          cur_block_pad_left, 0};
    std::vector<Value> cur_padding_high = {
        0, dim_(csb_lhs0, 0) - cur_ci_len, cur_block_pad_bot,
        cur_block_pad_right, dim_(csb_lhs0, 4) - cur_n_len};
    std::vector<Value> cur_padding_mid = {0, 0, 0, 0, 0};

    if_(
        logical_and_(cur_ci_len > 0, cur_d_len > 0, cur_h_len > 0,
                     cur_w_len > 0),
        [&]() {
          // d2c slice
          auto hbm_lhs_0 =
              memview_(hbm_lhs, {cur_ci_offset, 0, 0, 0, 0},
                       {cur_ci_len, dim_(hbm_lhs, 1), dim_(hbm_lhs, 2),
                        dim_(hbm_lhs, 3), dim_(hbm_lhs, 4)});
          auto cur_lhs_temp = memview_(
              cur_csb_lhs, {0, 0, 0, 0, 0},
              {cur_ci_len, dim_(csb_lhs0, 1), cur_h_len, cur_w_len, cur_n_len});
          async_load_(cur_lhs_cdma, hbm_lhs_0, cur_lhs_temp,
                      {0, cur_d_offset, cur_h_offset, cur_w_offset,
                       n_loop_offset_this_sip})
              .notify_(cur_lhs_sdma_pad);
          // c2c padding
          // ci_1_hi_wi_n->1_ci_hi_wi_n
          auto bitcast_csb_lhs = bitcast_(bitcast_lhs_csb_type, cur_csb_lhs);
          auto bitcast_csb_lhs_pad =
              bitcast_(bitcast_lhs_csb_type, cur_csb_lhs_pad);
          auto cur_lhs_temp1 = memview_(bitcast_csb_lhs, {0, 0, 0, 0, 0},
                                        {dim_(bitcast_csb_lhs, 0), cur_ci_len,
                                         cur_h_len, cur_w_len, cur_n_len});
          async_load_(cur_lhs_sdma_pad, cur_lhs_temp1, bitcast_csb_lhs_pad,
                      cur_offset, cur_padding_low, cur_padding_high,
                      cur_padding_mid, 0)
              .wait_on_(cur_lhs_cdma)
              .notify_(cur_lhs_sdma);
          // c2s transpose
          //   wait_dma_(cur_lhs_sdma_pad);
          // 1_ci_hi_wi_n->1_n_hi_wi_ci
          async_load_(cur_lhs_sdma, bitcast_csb_lhs_pad, cur_sip_lhs,
                      {0, 0, 0, 0, 0}, {0, 4, 2, 3, 1})
              .wait_on_(cur_lhs_sdma_pad);
        },
        [&]() {
          // memset zeros
          async_memset_(cur_lhs_sdma, cur_sip_lhs, 0);
        });

    auto hbm_rhs_0 = memview_(hbm_rhs, {cur_ci_offset, 0, 0, 0, 0},
                              {cur_ci_len, dim_(hbm_rhs, 1), dim_(hbm_rhs, 2),
                               dim_(hbm_rhs, 3), dim_(hbm_rhs, 4)});
    async_load_(cur_rhs_cdma, hbm_rhs_0, cur_csb_rhs,
                {0, cur_t_offset, cur_r_offset, cur_s_offset, cur_co_offset})
        .notify_(cur_rhs_sdma);
    // ci_1_r_s_co->1_ci_r_s_co
    auto bitcast_csb_rhs = bitcast_(bitcast_rhs_csb_type, cur_csb_rhs);
    // 1_ci_r_s_co->1_r_s_ci_co
    async_load_(cur_rhs_sdma, bitcast_csb_rhs, cur_sip_rhs, {0, 0, 0, 0, 0},
                {0, 2, 3, 1, 4})
        .wait_on_(cur_rhs_cdma);

    var_ current_store_idx(IntType(32));
    current_store_idx = 0;
    var_ current_loop_idx(IntType(32));
    current_loop_idx = 0;

    // print_("s_loop_num:", {s_loop_num});
    for_(0, n_loop_num, 1, [&](auto n) {
      for_(0, do_loop_num, 1, [&](auto d) {
        for_(0, ho_loop_num, 1, [&](auto h) {
          for_(0, wo_loop_num, 1, [&](auto w) {
            // print_("xxxw:", {w});
            for_(0, co_loop_num, 1, [&](auto co) {
              auto sip_out =
                  select_(current_store_idx % 2 == 0, sip_out0, sip_out1);
              for_(0, t_loop_num, 1, [&](auto t) {
                for_(0, ci_loop_num, 1, [&](auto ci) {
                  for_(0, r_loop_num, 1, [&](auto r) {
                    for_(0, s_loop_num, 1, [&](auto s) {
                      auto next_s_idx = (s + 1) % s_loop_num;
                      auto next_r_idx =
                          select_(next_s_idx == 0, ((r + 1) % r_loop_num), r);
                      auto next_r_edge = logical_and_(next_r_idx == 0, next_s_idx == 0);
                      auto next_ci_idx =
                          select_(next_r_edge, ((ci + 1) % ci_loop_num), ci);
                      auto next_ci_edge =
                          logical_and_(next_r_edge, next_ci_idx == 0);

                      auto next_t_idx =
                          select_(next_ci_edge, ((t + 1) % t_loop_num), t);
                      auto next_t_edge =
                          logical_and_(next_ci_edge, next_t_idx == 0);

                      auto next_co_idx =
                          select_(next_t_edge, ((co + 1) % co_loop_num), co);
                      auto next_co_edge =
                          logical_and_(next_t_edge, next_co_idx == 0);

                      auto next_wo_idx =
                          select_(next_co_edge, ((w + 1) % wo_loop_num), w);
                      auto next_wo_edge =
                          logical_and_(next_co_edge, next_wo_idx == 0);

                      auto next_ho_idx =
                          select_(next_wo_edge, ((h + 1) % ho_loop_num), h);
                      auto next_ho_edge =
                          logical_and_(next_wo_edge, next_ho_idx == 0);

                      auto next_do_idx =
                          select_(next_ho_edge, ((d + 1) % do_loop_num), d);
                      auto next_d_edge =
                          logical_and_(next_ho_edge, next_do_idx == 0);

                      auto next_n_idx = select_(next_d_edge, (n + 1), n);

                      auto ld_flag = ci + t + r + s;

                      auto next_n_offset = next_n_idx * dim_(csb_lhs0, 4) +
                                           n_loop_offset_this_sip;
                      auto next_do_offset = next_do_idx * dim_(csb_out0, 0);
                      auto next_ho_offset = next_ho_idx * dim_(csb_out0, 1);
                      auto next_wo_offset = next_wo_idx * dim_(csb_out0, 2);
                      auto next_ci_offset = next_ci_idx * dim_(csb_lhs0, 0);
                      auto next_t_offset = next_t_idx * dim_(csb_rhs0, 1);
                      auto next_r_offset = next_r_idx * dim_(csb_rhs0, 2);
                      auto next_s_offset = next_s_idx * dim_(csb_rhs0, 3);
                      auto next_co_offset = next_co_idx * dim_(csb_out0, 4);

                      auto next_d_offset =
                          next_do_offset * stride_d - pad_head + next_t_offset * window_dilation_d;
                      auto next_h_offset =
                          next_ho_offset * stride_h - pad_top + next_r_offset * window_dilation_h;
                      auto next_w_offset =
                          next_wo_offset * stride_w - pad_left + next_s_offset * window_dilation_w;

                      auto next_d_offset_end =
                          (next_do_offset + dim_(csb_out0, 0) - 1) * stride_d +
                          next_t_offset * window_dilation_d + actual_t - pad_head - 1;
                      auto next_h_offset_end =
                          (next_ho_offset + dim_(csb_out0, 1) - 1) * stride_h +
                          next_r_offset * window_dilation_h + actual_r - pad_top - 1;
                      auto next_w_offset_end =
                          (next_wo_offset + dim_(csb_out0, 2) - 1) * stride_w +
                          next_s_offset * window_dilation_w + actual_s - pad_left - 1;
                      auto next_ci_offset_end =
                          dim_(csb_lhs0, 0) + next_ci_offset - 1;
                      auto next_n_offset_end =
                          dim_(csb_lhs0, 4) + next_n_offset - 1;

                      auto next_block_pad_top = next_h_offset + pad_top -
                                                next_ho_offset * stride_h -
                                                next_r_offset * window_dilation_h;
                      auto next_block_pad_left = next_w_offset + pad_left -
                                                 next_wo_offset * stride_w -
                                                 next_s_offset * window_dilation_w;
                      auto next_d_len0 =
                          select_(next_d_offset_end > dim_(hbm_lhs, 1) - 1,
                                  dim_(hbm_lhs, 1) - next_d_offset,
                                  next_d_offset_end - next_d_offset + 1);
                      auto next_d_len =
                          select_(next_d_offset_end < 0, 0, next_d_len0);

                      auto next_h_len0 =
                          select_(next_h_offset_end > dim_(hbm_lhs, 2) - 1,
                                  dim_(hbm_lhs, 2) - next_h_offset,
                                  next_h_offset_end - next_h_offset + 1);
                      auto next_h_len =
                          select_(next_h_offset_end < 0, 0, next_h_len0);
                      auto next_w_len0 =
                          select_(next_w_offset_end > dim_(hbm_lhs, 3) - 1,
                                  dim_(hbm_lhs, 3) - next_w_offset,
                                  next_w_offset_end - next_w_offset + 1);
                      auto next_w_len =
                          select_(next_w_offset_end < 0, 0, next_w_len0);

                      auto next_ci_len =
                          select_(next_ci_offset_end > dim_(hbm_lhs, 0) - 1,
                                  dim_(hbm_lhs, 0) - next_ci_offset,
                                  next_ci_offset_end - next_ci_offset + 1);
                      auto next_n_len =
                          select_(next_n_offset_end > dim_(hbm_lhs, 4) - 1,
                                  dim_(hbm_lhs, 4) - next_n_offset,
                                  next_n_offset_end - next_n_offset + 1);

                      auto next_block_pad_bot =
                          (dim_(csb_out0, 1) - 1) * stride_h + actual_r -
                          (next_h_len - 1) - 1 - next_block_pad_top;
                      auto next_block_pad_right =
                          (dim_(csb_out0, 2) - 1) * stride_w + actual_s -
                          (next_w_len - 1) - 1 - next_block_pad_left;

                      auto next_csb_lhs = select_(current_loop_idx % 2 == 0,
                                                  csb_lhs1, csb_lhs0);
                      auto next_csb_lhs_pad = select_(current_loop_idx % 2 == 0,
                                                      csb_lhs3, csb_lhs2);
                      auto next_sip_lhs = select_(current_loop_idx % 2 == 0,
                                                  sip_lhs1, sip_lhs0);
                      auto next_lhs_cdma = select_(current_loop_idx % 2 == 0,
                                                   lhs_cdma1, lhs_cdma0);
                      auto next_lhs_sdma = select_(current_loop_idx % 2 == 0,
                                                   lhs_sdma1, lhs_sdma0);
                      auto cur_sip_lhs = select_(current_loop_idx % 2 == 0,
                                                 sip_lhs0, sip_lhs1);
                      auto cur_lhs_sdma = select_(current_loop_idx % 2 == 0,
                                                  lhs_sdma0, lhs_sdma1);

                      auto next_lhs_sdma_pad = select_(
                          current_loop_idx % 2 == 0, lhs_sdma3, lhs_sdma2);

                      auto next_rhs_cdma = select_(current_loop_idx % 2 == 0,
                                                   rhs_cdma1, rhs_cdma0);
                      auto next_csb_rhs = select_(current_loop_idx % 2 == 0,
                                                  csb_rhs1, csb_rhs0);
                      auto next_sip_rhs = select_(current_loop_idx % 2 == 0,
                                                  sip_rhs1, sip_rhs0);
                      auto next_rhs_sdma = select_(current_loop_idx % 2 == 0,
                                                   rhs_sdma1, rhs_sdma0);
                      auto cur_rhs_sdma = select_(current_loop_idx % 2 == 0,
                                                  rhs_sdma0, rhs_sdma1);
                      auto cur_sip_rhs = select_(current_loop_idx % 2 == 0,
                                                 sip_rhs0, sip_rhs1);

                      // 1.slice dma  2.pad c to 32x, dilationHW & pad
                      std::vector<Value> next_offset = {0, 0, 0, 0, 0};
                      std::vector<Value> next_padding_low = {
                          0, 0, next_block_pad_top, next_block_pad_left, 0};
                      std::vector<Value> next_padding_high = {
                          0, dim_(csb_lhs0, 0) - next_ci_len,
                          next_block_pad_bot, next_block_pad_right,
                          dim_(csb_lhs0, 4) - next_n_len};
                      std::vector<Value> next_padding_mid = {0, 0, 0, 0, 0};
                      //   print_("next_n_idx:", {next_n_idx});
                      //   print_("next_ci_len:", {next_ci_len});
                      //   print_("next_d_len:", {next_d_len});
                      //   print_("next_h_len:", {next_h_len});
                      //   print_("next_w_len:", {next_w_len});
                      //   print_("next_block_pad_top:", {next_block_pad_top});
                      //   print_("next_block_pad_left:",
                      //   {next_block_pad_left}); print_("next_block_pad_bot:",
                      //   {next_block_pad_bot});
                      //   print_("next_block_pad_right:",
                      //   {next_block_pad_right}); print_("next_n_offset:",
                      //   {next_n_offset}); print_("next_do_offset:",
                      //   {next_do_offset}); print_("next_ho_offset:",
                      //   {next_ho_offset}); print_("next_wo_offset:",
                      //   {next_wo_offset}); print_("next_co_offset:",
                      //   {next_co_offset}); print_("next_t_offset:",
                      //   {next_t_offset}); print_("next_r_offset:",
                      //   {next_r_offset}); print_("next_s_offset:",
                      //   {next_s_offset}); print_("next_n_len:",
                      //   {next_n_len}); print_("next_ci_offset:",
                      //   {next_ci_offset}); print_("next_d_offset:",
                      //   {next_d_offset}); print_("next_h_offset:",
                      //   {next_h_offset}); print_("next_w_offset:",
                      //   {next_w_offset}); print_("next_d_offset:",
                      //   {next_d_offset}); print_("next_d_offset_end:",
                      //   {next_d_offset_end}); print_("next_h_offset_end:",
                      //   {next_h_offset_end}); print_("next_w_offset_end:",
                      //   {next_w_offset_end}); print_("next_n_offset_end:",
                      //   {next_n_offset_end});

                      if_(next_n_idx < n_loop_num, [&]() {
                        if_(
                            logical_and_(next_ci_len > 0, next_d_len > 0,
                                         next_h_len > 0, next_w_len > 0),
                            [&]() {
                              // d2c slice
                              auto hbm_lhs_ci = memview_(
                                  hbm_lhs, {next_ci_offset, 0, 0, 0, 0},
                                  {next_ci_len, dim_(hbm_lhs, 1),
                                   dim_(hbm_lhs, 2), dim_(hbm_lhs, 3),
                                   dim_(hbm_lhs, 4)});
                              auto next_lhs_temp = memview_(
                                  next_csb_lhs, {0, 0, 0, 0, 0},
                                  {next_ci_len, dim_(csb_lhs0, 1), next_h_len,
                                   next_w_len, next_n_len});
                              async_load_(next_lhs_cdma, hbm_lhs_ci,
                                          next_lhs_temp,
                                          {0, next_d_offset, next_h_offset,
                                           next_w_offset, next_n_offset})
                                  .notify_(next_lhs_sdma_pad);
                              // c2c padding
                              // ci_1_hi_wi_n->1_ci_hi_wi_n
                              auto bitcast_csb_lhs =
                                  bitcast_(bitcast_lhs_csb_type, next_csb_lhs);
                              auto bitcast_csb_lhs_pad = bitcast_(
                                  bitcast_lhs_csb_type, next_csb_lhs_pad);
                              auto next_lhs_temp1 = memview_(
                                  bitcast_csb_lhs, {0, 0, 0, 0, 0},
                                  {dim_(bitcast_csb_lhs, 0), next_ci_len,
                                   next_h_len, next_w_len, next_n_len});
                              async_load_(next_lhs_sdma_pad, next_lhs_temp1,
                                          bitcast_csb_lhs_pad, next_offset,
                                          next_padding_low, next_padding_high,
                                          next_padding_mid, 0)
                                  .wait_on_(next_lhs_cdma)
                                  .notify_(next_lhs_sdma);
                              // c2s transpose
                              //   wait_dma_(next_lhs_sdma_pad);
                              // 1_ci_hi_wi_n->1_n_hi_wi_ci
                              async_load_(next_lhs_sdma, bitcast_csb_lhs_pad,
                                          next_sip_lhs, {0, 0, 0, 0, 0},
                                          {0, 4, 2, 3, 1})
                                  .wait_on_(next_lhs_sdma_pad);
                            },
                            [&]() {
                              // memset zeros
                              async_memset_(next_lhs_sdma, next_sip_lhs, 0);
                            });

                        auto hbm_rhs_ci = memview_(
                            hbm_rhs, {next_ci_offset, 0, 0, 0, 0},
                            {next_ci_len, dim_(hbm_rhs, 1), dim_(hbm_rhs, 2),
                             dim_(hbm_rhs, 3), dim_(hbm_rhs, 4)});

                        async_load_(next_rhs_cdma, hbm_rhs_ci, next_csb_rhs,
                                    {0, next_t_offset, next_r_offset,
                                     next_s_offset, next_co_offset})
                            .notify_(next_rhs_sdma);
                        // ci_1_r_s_co->1_ci_r_s_co
                        auto bitcast_csb_rhs =
                            bitcast_(bitcast_rhs_csb_type, next_csb_rhs);
                        // 1_ci_r_s_co->1_r_s_ci_co });
                        async_load_(next_rhs_sdma, bitcast_csb_rhs,
                                    next_sip_rhs, {0, 0, 0, 0, 0},
                                    {0, 2, 3, 1, 4})
                            .wait_on_(next_rhs_cdma);
                      });
                      wait_dma_(cur_lhs_sdma);
                      wait_dma_(cur_rhs_sdma);
                      //   print_("a:wait_dma_");

                      auto st_flag = select_(next_ci_edge, 1, 0);

                      call_(c_func_name, cur_sip_lhs, cur_sip_rhs, sip_out,
                            dim_(cur_sip_lhs, 1), dim_(cur_sip_lhs, 2),
                            dim_(cur_sip_lhs, 3), dim_(cur_sip_lhs, 4),
                            dim_(cur_sip_rhs, 1), dim_(cur_sip_rhs, 2),
                            dim_(cur_sip_rhs, 4), dim_(cur_sip_out, 2),
                            dim_(cur_sip_out, 3), stride_h, stride_w,
                            window_dilation_h, window_dilation_w, ld_flag,
                            st_flag);

                      //   print_("current_loop_idx:", {current_loop_idx});
                      current_loop_idx += 1;
                    });  // s
                  });    // r
                });      // t
              });        // ci
              // For storing, the synchronization has been done at "call", no
              // need to wait sdma/cdma
              // partition_params.csb_out_shape = {1, ho, wo, n, co}; //
              // {dout, ho, wo, n, co} partition_params.sip_out_shape = {n, 1,
              // ho, wo, co}; // {n, dout, ho, wo, co}
              if_(current_store_idx != 0, [&]() { wait_dma_(out_cdma); });
              auto csb_out =
                  select_(current_store_idx % 2 == 0, csb_out0, csb_out1);
              // auto bitcast_sip_out = bitcast_(
              //     bitcast_out_sip_type,
              //     memview_(sip_out, {0, 0, 0, 0, 0},
              //              {dim_(sip_out, 0), dim_(sip_out, 1),
              //               dim_(sip_out, 2), dim_(sip_out, 3),
              //               dim_(sip_out, 4)}));  // citrs(co)->trsci(co)
              //{1, n, ho, wo, co}->{1, ho, wo, n, co}
              async_store_(out_sdma, sip_out, csb_out, {0, 0, 0, 0, 0},
                           {0, 2, 3, 1, 4})
                  .notify_(out_cdma);
              async_store_(out_cdma, csb_out, hbm_out,
                           {d * dim_(csb_out0, 0), h * dim_(csb_out0, 1),
                            w * dim_(csb_out0, 2),
                            (n_loop_offset_this_sip + n * dim_(csb_out0, 3)),
                            co * dim_(csb_out0, 4)})
                  .wait_on_(out_sdma);
              current_store_idx += 1;
            });  // co
          });    // wo
        });      // ho
      });        // do
    });          // n
    wait_dma_(out_cdma);
  });
  EFLOG(DBG) << "Exit Common20KernelFuncConv3dBPK !!!" << std::endl;
}

void Common20Conv3dBPKImpl(const DtuOpContext& context) {
  EFLOG(DBG) << "Enter Common20Conv3dBPKImpl impl !!!" << std::endl;
  auto op = context.op_desc.op;
  non4c::SplitParams split_params;
  if (::dtu::FLAGS_ENFLAME_CLUSTER_PARALLEL) {
    split_params = non4c::CreateSplitParam(op, {0, non4c::kBCAST}, {0});
  } else {
    split_params = non4c::CreateSplitParam(op, {non4c::kNONE, non4c::kNONE},
                                           {non4c::kNONE}, 1);
  }
  non4c::PreRun(context, split_params);

  auto tr = TargetResource(context.target);
  int32_t sip_cnt = Cast64To32Bit(tr.get_sip_number());
  int32_t cluster_num = Cast64To32Bit(tr.get_cluster_number());
  int32_t clusters_used = Cast64To32Bit(split_params.clusters_used);
  EFLOG(DBG) << "sip_cnt: " << sip_cnt;
  EFLOG(DBG) << "cluster_num: " << cluster_num;
  EFLOG(DBG) << "clusters_used: " << clusters_used;

  auto cluster_hbm_lhs = split_params.inputs[0];
  auto cluster_hbm_rhs = split_params.inputs[1];
  auto cluster_hbm_out = split_params.results[0];

  auto hbm_lhs_shape = split_params.input_shapes[0];
  auto hbm_rhs_shape = split_params.input_shapes[1];
  auto hbm_out_shape = split_params.result_shapes[0];

  bool ef32_mode = false;
  if (::dtu::FLAGS_ENFLAME_ENABLE_EF32) {
    ef32_mode = true;
  } else {
    ef32_mode = false;
  }

  for (int32_t cluster_id = 0; cluster_id < clusters_used; cluster_id++) {
    EFLOG(DBG) << "============cluster_id============== " << cluster_id;
    auto lhs_hbm_tile = cluster_hbm_lhs[cluster_id];
    auto rhs_hbm_tile = cluster_hbm_rhs[cluster_id];
    auto out_hbm_tile = cluster_hbm_out[cluster_id];
    auto lhs_hbm_shape = hbm_lhs_shape[cluster_id];
    auto rhs_hbm_shape = hbm_rhs_shape[cluster_id];
    auto out_hbm_shape = hbm_out_shape[cluster_id];

    EFLOG(DBG) << "xxxxx hbm_lhs_shape: "
               << getShapeStr(lhs_hbm_shape.Dimensions().begin(),
                              lhs_hbm_shape.Dimensions().end());
    EFLOG(DBG) << "xxxxx hbm_rhs_shape: "
               << getShapeStr(rhs_hbm_shape.Dimensions().begin(),
                              rhs_hbm_shape.Dimensions().end());
    EFLOG(DBG) << "xxxxx hbm_out_shape: "
               << getShapeStr(out_hbm_shape.Dimensions().begin(),
                              out_hbm_shape.Dimensions().end());

    std::vector<int64_t> lhs_hbm_shape_vec = std::vector<int64_t>(
        lhs_hbm_shape.Dimensions().begin(), lhs_hbm_shape.Dimensions().end());
    std::vector<int64_t> rhs_hbm_shape_vec = std::vector<int64_t>(
        rhs_hbm_shape.Dimensions().begin(), rhs_hbm_shape.Dimensions().end());
    std::vector<int64_t> out_hbm_shape_vec = std::vector<int64_t>(
        out_hbm_shape.Dimensions().begin(), out_hbm_shape.Dimensions().end());

    std::vector<int32_t> hbm_lhs_shape32 =
        CastVec64ToVec32Bit(lhs_hbm_shape_vec);
    std::vector<int32_t> hbm_rhs_shape32 =
        CastVec64ToVec32Bit(rhs_hbm_shape_vec);
    std::vector<int32_t> hbm_out_shape32 =
        CastVec64ToVec32Bit(out_hbm_shape_vec);

    // split data flow
    int32_t bpe =
        Cast64To32Bit(lhs_hbm_tile.type_().element_type_().size_in_bits_() / 8);
    Conv3dPartitionParam partition_params =
        InitConv3dPartitionParamBPK(context, bpe);
    std::vector<int32_t> csb_lhs_shape =
        CastVec64ToVec32Bit(partition_params.csb_lhs_shape);
    std::vector<int32_t> csb_rhs_shape =
        CastVec64ToVec32Bit(partition_params.csb_rhs_shape);
    std::vector<int32_t> csb_out_shape =
        CastVec64ToVec32Bit(partition_params.csb_out_shape);
    std::vector<int32_t> sip_lhs_shape =
        CastVec64ToVec32Bit(partition_params.sip_lhs_shape);
    std::vector<int32_t> sip_rhs_shape =
        CastVec64ToVec32Bit(partition_params.sip_rhs_shape);
    std::vector<int32_t> sip_out_shape =
        CastVec64ToVec32Bit(partition_params.sip_out_shape);

    EFLOG(DBG) << "xxxxx csb_lhs_shape: "
               << getShapeStr(csb_lhs_shape.begin(), csb_lhs_shape.end());
    EFLOG(DBG) << "xxxxx csb_rhs_shape: "
               << getShapeStr(csb_rhs_shape.begin(), csb_rhs_shape.end());
    EFLOG(DBG) << "xxxxx csb_out_shape: "
               << getShapeStr(csb_out_shape.begin(), csb_out_shape.end());

    EFLOG(DBG) << "xxxxx sip_lhs_shape: "
               << getShapeStr(sip_lhs_shape.begin(), sip_lhs_shape.end());
    EFLOG(DBG) << "xxxxx sip_rhs_shape: "
               << getShapeStr(sip_rhs_shape.begin(), sip_rhs_shape.end());
    EFLOG(DBG) << "xxxxx sip_out_shape: "
               << getShapeStr(sip_out_shape.begin(), sip_out_shape.end());

    auto lhs_hbm_type =
        DRAMType(lhs_hbm_tile.type_().element_type_(), lhs_hbm_shape_vec);
    auto rhs_hbm_type =
        DRAMType(rhs_hbm_tile.type_().element_type_(), rhs_hbm_shape_vec);
    auto out_hbm_type =
        DRAMType(out_hbm_tile.type_().element_type_(), out_hbm_shape_vec);
    auto csb_lhs_type = SRAMType(lhs_hbm_tile.type_().element_type_(),
                                 CastVec32ToVec64Bit(csb_lhs_shape));
    auto csb_rhs_type = SRAMType(rhs_hbm_tile.type_().element_type_(),
                                 CastVec32ToVec64Bit(csb_rhs_shape));
    auto csb_out_type = SRAMType(out_hbm_tile.type_().element_type_(),
                                 CastVec32ToVec64Bit(csb_out_shape));
    auto sip_lhs_type = L1Type(lhs_hbm_tile.type_().element_type_(),
                               CastVec32ToVec64Bit(sip_lhs_shape));
    auto sip_rhs_type = L1Type(rhs_hbm_tile.type_().element_type_(),
                               CastVec32ToVec64Bit(sip_rhs_shape));
    auto sip_out_type = L1Type(out_hbm_tile.type_().element_type_(),
                               CastVec32ToVec64Bit(sip_out_shape));

    // bitcast
    std::vector<int64_t> bitcast_out_sip_shape_vec = {
        csb_out_shape[0], csb_out_shape[1], csb_out_shape[2], csb_out_shape[3],
        csb_out_shape[4]};
    std::vector<int64_t> bitcast_lhs_csb_shape_vec = {
        csb_lhs_shape[1], csb_lhs_shape[0], csb_lhs_shape[2], csb_lhs_shape[3],
        csb_lhs_shape[4]};
    std::vector<int64_t> bitcast_rhs_csb_shape_vec = {
        csb_rhs_shape[1], csb_rhs_shape[0], csb_rhs_shape[2], csb_rhs_shape[3],
        csb_rhs_shape[4]};

    std::string func_name = context.identify + "_cluster" +
                            std::to_string(cluster_id) + "_" +
                            "conv3dbpk_dataflow";
    EFLOG(DBG) << "func_name: " << func_name;

    std::vector<Type> func_types;
    func_types.push_back(lhs_hbm_type);
    func_types.push_back(rhs_hbm_type);

    func_types.push_back(csb_lhs_type);
    func_types.push_back(csb_lhs_type);
    func_types.push_back(csb_lhs_type);
    func_types.push_back(csb_lhs_type);
    func_types.push_back(csb_rhs_type);
    func_types.push_back(csb_rhs_type);
    func_types.push_back(csb_out_type);
    func_types.push_back(csb_out_type);

    func_types.push_back(sip_lhs_type);
    func_types.push_back(sip_lhs_type);
    func_types.push_back(sip_rhs_type);
    func_types.push_back(sip_rhs_type);
    func_types.push_back(sip_out_type);
    func_types.push_back(sip_out_type);

    func_types.push_back(CDMAType());
    func_types.push_back(CDMAType());
    func_types.push_back(CDMAType());
    func_types.push_back(CDMAType());
    func_types.push_back(CDMAType());

    func_types.push_back(SDMAType());
    func_types.push_back(SDMAType());
    func_types.push_back(SDMAType());
    func_types.push_back(SDMAType());
    func_types.push_back(SDMAType());
    func_types.push_back(SDMAType());
    func_types.push_back(SDMAType());

    func_types.push_back(IntType(32));
    func_types.push_back(IntType(32));
    func_types.push_back(IntType(32));
    func_types.push_back(IntType(32));
    func_types.push_back(IntType(32));
    func_types.push_back(IntType(32));
    func_types.push_back(IntType(32));
    func_types.push_back(IntType(32));
    func_types.push_back(IntType(32));

    func_types.push_back(IntType(32));
    func_types.push_back(IntType(32));

    func_types.push_back(IntType(32));

    std::vector<Type> results_types;
    results_types.push_back(out_hbm_type);
    Conv3dPadStrideDilationParam pad_stride_dilation_param =
        GetConv3dPadStrideDilationParam(context);
    auto& op_desc = context.op_desc;
    auto op = op_desc.op;
    auto conv_op = llvm::cast<DialectNS::ConvOp>(op);
    auto element_type =
        hlir::ShapeUtil::GetElementTypeFromMlir(conv_op.lhs().getType());
    Common20KernelFuncConv3dBPK(context, func_name, func_types, results_types,
                                pad_stride_dilation_param, element_type,
                                ef32_mode, bitcast_out_sip_shape_vec,
                                bitcast_lhs_csb_shape_vec,
                                bitcast_rhs_csb_shape_vec);

    int32_t n_loop_times =
        (hbm_out_shape32[3] + csb_out_shape[3] - 1) / csb_out_shape[3];
    int32_t d_loop_times =
        (hbm_out_shape32[0] + csb_out_shape[0] - 1) / csb_out_shape[0];
    int32_t h_loop_times =
        (hbm_out_shape32[1] + csb_out_shape[1] - 1) / csb_out_shape[1];
    int32_t w_loop_times =
        (hbm_out_shape32[2] + csb_out_shape[2] - 1) / csb_out_shape[2];
    int32_t ci_loop_times =
        (hbm_lhs_shape32[0] + csb_lhs_shape[0] - 1) / csb_lhs_shape[0];
    int32_t t_loop_times =
        (hbm_rhs_shape32[1] + csb_rhs_shape[1] - 1) / csb_rhs_shape[1];
    int32_t r_loop_times =
        (hbm_rhs_shape32[2] + csb_rhs_shape[2] - 1) / csb_rhs_shape[2];
    int32_t s_loop_times =
        (hbm_rhs_shape32[3] + csb_rhs_shape[3] - 1) / csb_rhs_shape[3];
    int32_t co_loop_times =
        (hbm_out_shape32[4] + csb_out_shape[4] - 1) / csb_out_shape[4];

    int32_t sip_num_used = n_loop_times > sip_cnt ? sip_cnt : n_loop_times;
    int32_t n_loop_times_per_sip =
        (n_loop_times + sip_num_used - 1) / sip_num_used;
    EFLOG(DBG) << "xxxx n_loop_times_per_sip : " << n_loop_times_per_sip;
    EFLOG(DBG) << "xxxxx sip_num_used : " << sip_num_used;

    std::vector<int32_t> n_loop_offset_per_sip;
    std::vector<int32_t> n_loop_len_per_sip;
    for (int32_t s_id = 0; s_id < sip_num_used; s_id++) {
      int32_t n_off = s_id * n_loop_times_per_sip;
      n_loop_offset_per_sip.push_back(n_off * csb_out_shape[3]);
      n_loop_len_per_sip.push_back(n_off + n_loop_times_per_sip < n_loop_times
                                       ? n_loop_times_per_sip
                                       : n_loop_times - n_off);
    }

    std::vector<Value> csb_input0_vec;
    std::vector<Value> csb_input1_vec;
    std::vector<Value> csb_input2_vec;
    std::vector<Value> csb_input3_vec;
    std::vector<Value> csb_weight0_vec;
    std::vector<Value> csb_weight1_vec;
    std::vector<Value> csb_output0_vec;
    std::vector<Value> csb_output1_vec;

    std::vector<Value> sip_input0_vec;
    std::vector<Value> sip_input1_vec;
    std::vector<Value> sip_weight0_vec;
    std::vector<Value> sip_weight1_vec;
    std::vector<Value> sip_output0_vec;
    std::vector<Value> sip_output1_vec;

    std::vector<Value> in_cdma0_vec;
    std::vector<Value> in_cdma1_vec;
    std::vector<Value> wt_cdma0_vec;
    std::vector<Value> wt_cdma1_vec;
    std::vector<Value> out_cdma_vec;

    std::vector<Value> in_sdma0_vec;
    std::vector<Value> in_sdma1_vec;
    std::vector<Value> in_sdma2_vec;
    std::vector<Value> in_sdma3_vec;
    std::vector<Value> wt_sdma0_vec;
    std::vector<Value> wt_sdma1_vec;
    std::vector<Value> out_sdma_vec;

    for (int32_t s_id = 0; s_id < sip_num_used; s_id++) {
      // alloc csb buffer
      auto csb_input0 = alloc_(csb_lhs_type, cluster_id, s_id);
      auto csb_input1 = alloc_(csb_lhs_type, cluster_id, s_id);
      auto csb_input2 = alloc_(csb_lhs_type, cluster_id, s_id);
      auto csb_input3 = alloc_(csb_lhs_type, cluster_id, s_id);
      auto csb_weight0 = alloc_(csb_rhs_type, cluster_id, s_id);
      auto csb_weight1 = alloc_(csb_rhs_type, cluster_id, s_id);
      auto csb_out0 = alloc_(csb_out_type, cluster_id, s_id);
      auto csb_out1 = alloc_(csb_out_type, cluster_id, s_id);
      csb_input0_vec.push_back(csb_input0);
      csb_input1_vec.push_back(csb_input1);
      csb_input2_vec.push_back(csb_input2);
      csb_input3_vec.push_back(csb_input3);
      csb_weight0_vec.push_back(csb_weight0);
      csb_weight1_vec.push_back(csb_weight1);
      csb_output0_vec.push_back(csb_out0);
      csb_output1_vec.push_back(csb_out1);

      // alloc sip buffer
      auto sip_input0 = alloc_(sip_lhs_type, cluster_id, s_id);
      auto sip_input1 = alloc_(sip_lhs_type, cluster_id, s_id);
      auto sip_weight0 = alloc_(sip_rhs_type, cluster_id, s_id);
      auto sip_weight1 = alloc_(sip_rhs_type, cluster_id, s_id);
      auto sip_out0 = alloc_(sip_out_type, cluster_id, s_id);
      auto sip_out1 = alloc_(sip_out_type, cluster_id, s_id);
      sip_input0_vec.push_back(sip_input0);
      sip_input1_vec.push_back(sip_input1);
      sip_weight0_vec.push_back(sip_weight0);
      sip_weight1_vec.push_back(sip_weight1);
      sip_output0_vec.push_back(sip_out0);
      sip_output1_vec.push_back(sip_out1);

      // alloc cdma
      auto in_cdma0 = alloc_dma_(CDMAType(), cluster_id, s_id);
      auto in_cdma1 = alloc_dma_(CDMAType(), cluster_id, s_id);
      auto wt_cdma0 = alloc_dma_(CDMAType(), cluster_id, s_id);
      auto wt_cdma1 = alloc_dma_(CDMAType(), cluster_id, s_id);
      auto out_cdma = alloc_dma_(CDMAType(), cluster_id, s_id);
      in_cdma0_vec.push_back(in_cdma0);
      in_cdma1_vec.push_back(in_cdma1);
      wt_cdma0_vec.push_back(wt_cdma0);
      wt_cdma1_vec.push_back(wt_cdma1);
      out_cdma_vec.push_back(out_cdma);

      // alloc sdma
      auto in_sdma0 = alloc_dma_(SDMAType(), cluster_id, s_id);
      auto in_sdma1 = alloc_dma_(SDMAType(), cluster_id, s_id);
      auto in_sdma2 = alloc_dma_(SDMAType(), cluster_id, s_id);
      auto in_sdma3 = alloc_dma_(SDMAType(), cluster_id, s_id);
      auto wt_sdma0 = alloc_dma_(SDMAType(), cluster_id, s_id);
      auto wt_sdma1 = alloc_dma_(SDMAType(), cluster_id, s_id);
      auto out_sdma = alloc_dma_(SDMAType(), cluster_id, s_id);
      in_sdma0_vec.push_back(in_sdma0);
      in_sdma1_vec.push_back(in_sdma1);
      in_sdma2_vec.push_back(in_sdma2);
      in_sdma3_vec.push_back(in_sdma3);
      wt_sdma0_vec.push_back(wt_sdma0);
      wt_sdma1_vec.push_back(wt_sdma1);
      out_sdma_vec.push_back(out_sdma);
    }

    EFLOG(DBG) << "xxxx n_loop_times : " << n_loop_times;
    EFLOG(DBG) << "xxxx d_loop_times : " << d_loop_times;
    EFLOG(DBG) << "xxxx h_loop_times : " << h_loop_times;
    EFLOG(DBG) << "xxxx w_loop_times : " << w_loop_times;
    EFLOG(DBG) << "xxxx ci_loop_times : " << ci_loop_times;
    EFLOG(DBG) << "xxxx t_loop_times : " << t_loop_times;
    EFLOG(DBG) << "xxxx r_loop_times : " << r_loop_times;
    EFLOG(DBG) << "xxxx s_loop_times : " << s_loop_times;
    EFLOG(DBG) << "xxxx co_loop_times : " << co_loop_times;

    // for debug
    auto pad_head = pad_stride_dilation_param.pad_head;
    auto pad_tail = pad_stride_dilation_param.pad_tail;
    auto pad_top = pad_stride_dilation_param.pad_top;
    auto pad_bot = pad_stride_dilation_param.pad_bot;
    auto pad_left = pad_stride_dilation_param.pad_left;
    auto pad_right = pad_stride_dilation_param.pad_right;
    auto stride_d = pad_stride_dilation_param.stride_d;
    auto stride_h = pad_stride_dilation_param.stride_h;
    auto stride_w = pad_stride_dilation_param.stride_w;
    auto base_dilation_d = pad_stride_dilation_param.base_dilation_d;
    auto base_dilation_h = pad_stride_dilation_param.base_dilation_h;
    auto base_dilation_w = pad_stride_dilation_param.base_dilation_w;
    auto window_dilation_d = pad_stride_dilation_param.window_dilation_d;
    auto window_dilation_h = pad_stride_dilation_param.window_dilation_h;
    auto window_dilation_w = pad_stride_dilation_param.window_dilation_w;
    auto T = rhs_hbm_shape_vec[1];
    auto R = rhs_hbm_shape_vec[2];
    auto S = rhs_hbm_shape_vec[3];

    EFLOG(DBG) << "xxxx sip_num_used : " << sip_num_used;
    EFLOG(DBG) << "xxxxx pad_top : " << pad_top;
    EFLOG(DBG) << "xxxxx pad_bot : " << pad_bot;
    EFLOG(DBG) << "xxxxx pad_left : " << pad_left;
    EFLOG(DBG) << "xxxxx pad_right : " << pad_right;
    EFLOG(DBG) << "xxxxx pad_head : " << pad_head;
    EFLOG(DBG) << "xxxxx pad_tail : " << pad_tail;
    EFLOG(DBG) << "xxxxx stride_d : " << stride_d;
    EFLOG(DBG) << "xxxxx stride_h : " << stride_h;
    EFLOG(DBG) << "xxxxx stride_w : " << stride_w;
    EFLOG(DBG) << "xxxx T : " << T;
    EFLOG(DBG) << "xxxx R : " << R;
    EFLOG(DBG) << "xxxx S : " << S;
    EFLOG(DBG) << "xxxx base_dilation_d : " << base_dilation_d;
    EFLOG(DBG) << "xxxx base_dilation_h : " << base_dilation_h;
    EFLOG(DBG) << "xxxx base_dilation_w : " << base_dilation_w;
    EFLOG(DBG) << "xxxxx window_dilation_d : " << window_dilation_d;
    EFLOG(DBG) << "xxxxx window_dilation_h : " << window_dilation_h;
    EFLOG(DBG) << "xxxxx window_dilation_w : " << window_dilation_w;

    EFLOG(DBG) << "=====launch_=======cluster_id========= " << cluster_id;

    auto task = async_([&]() {
      for (int32_t s_id = 0; s_id < sip_num_used; s_id++) {
        //   for (int32_t s_id = 0; s_id < 1; s_id++) {
        EFLOG(DBG) << "s_id : " << s_id;
        EFLOG(DBG) << "n_loop_offset_per_sip[s_id] : "
                   << n_loop_offset_per_sip[s_id];
        EFLOG(DBG) << "n_loop_len_per_sip[s_id] : "
                   << n_loop_len_per_sip[s_id];
        if (n_loop_len_per_sip[s_id] > 0) {
          launch_(func_name,
                  {lhs_hbm_tile,
                   rhs_hbm_tile,
                   csb_input0_vec[s_id],
                   csb_input1_vec[s_id],
                   csb_input2_vec[s_id],
                   csb_input3_vec[s_id],
                   csb_weight0_vec[s_id],
                   csb_weight1_vec[s_id],
                   csb_output0_vec[s_id],
                   csb_output1_vec[s_id],
                   sip_input0_vec[s_id],
                   sip_input1_vec[s_id],
                   sip_weight0_vec[s_id],
                   sip_weight1_vec[s_id],
                   sip_output0_vec[s_id],
                   sip_output1_vec[s_id],
                   in_cdma0_vec[s_id],
                   in_cdma1_vec[s_id],
                   wt_cdma0_vec[s_id],
                   wt_cdma1_vec[s_id],
                   out_cdma_vec[s_id],
                   in_sdma0_vec[s_id],
                   in_sdma1_vec[s_id],
                   in_sdma2_vec[s_id],
                   in_sdma3_vec[s_id],
                   wt_sdma0_vec[s_id],
                   wt_sdma1_vec[s_id],
                   out_sdma_vec[s_id],
                   Cast64To32Bit(n_loop_times),
                   Cast64To32Bit(d_loop_times),
                   Cast64To32Bit(h_loop_times),
                   Cast64To32Bit(w_loop_times),
                   Cast64To32Bit(ci_loop_times),
                   Cast64To32Bit(t_loop_times),
                   Cast64To32Bit(r_loop_times),
                   Cast64To32Bit(s_loop_times),
                   Cast64To32Bit(co_loop_times),
                   Cast64To32Bit(n_loop_offset_per_sip[s_id]),
                   Cast64To32Bit(n_loop_len_per_sip[s_id]),
                   Cast64To32Bit(s_id)},
                  {out_hbm_tile}, OnEngine::SIP, Cast64To32Bit(cluster_id),
                  Cast64To32Bit(s_id));
        }
      }
    });
    await_(task);
  }
  non4c::PostRun(context, split_params);

  EFLOG(DBG) << "Exit Common20Conv3dBPKImpl impl !!!" << std::endl;
}
}  // namespace factor
