/* Copyright 2018-2022 Enflame. All Rights Reserved.

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

#include "ops/common/dtu_conv3d_op_utils.h"
namespace factor {
namespace conv3d_utils {
Conv3dPadStrideDilationParam GetConv3dPadStrideDilationParam(
    const dtu::op::DtuOpContext& context) {
  auto op = context.op_desc.op;
  auto conv_attr = hlir::getConvAttributes(op);
  Conv3dPadStrideDilationParam pad_stride_dilation_param;
  pad_stride_dilation_param.pad_head = conv_attr.low_padding[0];
  pad_stride_dilation_param.pad_tail = conv_attr.high_padding[0];
  pad_stride_dilation_param.pad_top = conv_attr.low_padding[1];
  pad_stride_dilation_param.pad_bot = conv_attr.high_padding[1];
  pad_stride_dilation_param.pad_left = conv_attr.low_padding[2];
  pad_stride_dilation_param.pad_right = conv_attr.high_padding[2];
  pad_stride_dilation_param.stride_d = conv_attr.window_strides[0];
  pad_stride_dilation_param.stride_h = conv_attr.window_strides[1];
  pad_stride_dilation_param.stride_w = conv_attr.window_strides[2];
  pad_stride_dilation_param.window_dilation_d = conv_attr.rhs_dilation[0];
  pad_stride_dilation_param.window_dilation_h = conv_attr.rhs_dilation[1];
  pad_stride_dilation_param.window_dilation_w = conv_attr.rhs_dilation[2];
  pad_stride_dilation_param.base_dilation_d = conv_attr.lhs_dilation[0];
  pad_stride_dilation_param.base_dilation_h = conv_attr.lhs_dilation[1];
  pad_stride_dilation_param.base_dilation_w = conv_attr.lhs_dilation[2];
  return pad_stride_dilation_param;
}

Conv3dPartitionParam InitConv3dPartitionParam(
    const dtu::op::DtuOpContext& context, const int bpe) {
  Conv3dPartitionParam partition_params{};
  // auto partition record to lhs and rhs shape
  bool ef32_mode = false;
  if (::dtu::FLAGS_ENFLAME_ENABLE_EF32) {
    EFLOG(DBG) << "enable enflame ef32 flag";
    ef32_mode = true;
  } else {
    EFLOG(DBG) << "disable enflame ef32 flag";
    ef32_mode = false;
  }
  auto op = context.op_desc.op;
  auto conv_op = llvm::cast<DialectNS::ConvOp>(op);
  auto lhs = conv_op.lhs();
  auto rhs = conv_op.rhs();
  auto lhs_shape = hlir::ShapeUtil::GetShapeFromMlir(lhs);
  auto rhs_shape = hlir::ShapeUtil::GetShapeFromMlir(rhs);
  auto out_shape = hlir::ShapeUtil::GetShapeFromMlir(op->getResult(0));
  auto conv_attr = hlir::getConvAttributes(op);
  int64_t pad_head = conv_attr.low_padding[0];
  int64_t pad_tail = conv_attr.high_padding[0];
  int64_t pad_top = conv_attr.low_padding[1];
  int64_t pad_bot = conv_attr.high_padding[1];
  int64_t pad_left = conv_attr.low_padding[2];
  int64_t pad_right = conv_attr.high_padding[2];
  int64_t stride_d = conv_attr.window_strides[0];
  int64_t stride_h = conv_attr.window_strides[1];
  int64_t stride_w = conv_attr.window_strides[2];
  int64_t base_dilation_d = conv_attr.lhs_dilation[0];
  int64_t base_dilation_h = conv_attr.lhs_dilation[1];
  int64_t base_dilation_w = conv_attr.lhs_dilation[2];
  int64_t window_dilation_d = conv_attr.rhs_dilation[0];
  int64_t window_dilation_h = conv_attr.rhs_dilation[1];
  int64_t window_dilation_w = conv_attr.rhs_dilation[2];

  auto N = lhs_shape[0];
  auto Di = lhs_shape[1];
  auto Hi = lhs_shape[2];
  auto Wi = lhs_shape[3];
  auto Ci = lhs_shape[4];
  auto T = rhs_shape[0];
  auto R = rhs_shape[1];
  auto S = rhs_shape[2];
  auto Co = rhs_shape[4];
  auto Do = out_shape[1];
  auto Ho = out_shape[2];
  auto Wo = out_shape[3];
  auto ci_len = 16;

  auto actual_r = (R - 1) * window_dilation_h + 1;
  auto actual_s = (S - 1) * window_dilation_w + 1;
  int64_t n = 1;
  int64_t d = 1;
  int64_t ho = 1;
  int64_t wo = 1;
  int64_t co = 32;
  int64_t hi = actual_r + (ho - 1) * stride_h;
  int64_t wi = actual_s + (wo - 1) * stride_w;
  auto tr = dtu::op::TargetResource(context.target);
  int64_t ele_num_max = 0;
  if (bpe > 0) {
    ele_num_max = (tr.get_sip_dmem_size() - tr.get_sip_stack_size()) / 2 / bpe;
    ci_len = 64 / bpe;
    co = 128 / bpe;
  }
  if (bpe == 4 && ef32_mode == true) {
    co = 128;
  }

  int64_t ci = 0;
  auto CiPad = ((Ci + ci_len - 1) / ci_len) * ci_len;
  int64_t sip_size_used = 0;
  auto element_type =
      hlir::ShapeUtil::GetElementTypeFromMlir(conv_op.lhs().getType());

  // for ho20wo20
  if (bpe == 2 && Ho > 16 && Wo > 16) {
    ci = ci_len;
    ho = 20;
    wo = 20;
    hi = actual_r + (ho - 1) * stride_h;
    wi = actual_s + (wo - 1) * stride_w;
    while (1) {
      sip_size_used = hi * wi * ci + ho * wo * co + R * S * ci * co;
      if (sip_size_used > ele_num_max) {
        ci = ci - ci_len;
        break;
      }
      ci = ci + ci_len;
      if (ci > CiPad) {
        ci = ci - ci_len;
        break;
      }
    }
  }
  // for ho16wo16
  if (ci == 0) {
    ci = ci_len;
    ho = 16;
    wo = 16;
    hi = actual_r + (ho - 1) * stride_h;
    wi = actual_s + (wo - 1) * stride_w;
    while (1) {
      sip_size_used = hi * wi * ci + ho * wo * co + R * S * ci * co;
      if (sip_size_used > ele_num_max) {
        ci = ci - ci_len;
        break;
      }
      ci = ci + ci_len;
      if (ci > CiPad) {
        ci = ci - ci_len;
        break;
      }
    }
  }
  // for ho8wo8
  // ci = 0;
  if (ci == 0) {
    ci = ci_len;
    ho = 8;
    wo = 8;
    hi = actual_r + (ho - 1) * stride_h;
    wi = actual_s + (wo - 1) * stride_w;
    while (1) {
      sip_size_used = hi * wi * ci + ho * wo * co + R * S * ci * co;
      if (sip_size_used > ele_num_max) {
        ci = ci - ci_len;
        break;
      }
      ci = ci + ci_len;
      if (ci > CiPad) {
        ci = ci - ci_len;
        break;
      }
    }
  }
  // for ho3wo3
  // ci = 0;
  if (ci == 0) {
    ci = ci_len;
    ho = 3;
    wo = 3;
    hi = actual_r + (ho - 1) * stride_h;
    wi = actual_s + (wo - 1) * stride_w;
    while (1) {
      sip_size_used = hi * wi * ci + ho * wo * co + R * S * ci * co;
      if (sip_size_used > ele_num_max) {
        ci = ci - ci_len;
        break;
      }
      ci = ci + ci_len;
      if (ci > CiPad) {
        ci = ci - ci_len;
        break;
      }
    }
  }
  // for ho1wo1
  // ci = 0;
  if (ci == 0) {
    ci = ci_len;
    ho = 1;
    wo = 1;
    hi = actual_r + (ho - 1) * stride_h;
    wi = actual_s + (wo - 1) * stride_w;
    while (1) {
      sip_size_used = hi * wi * ci + ho * wo * co + R * S * ci * co;
      if (sip_size_used > ele_num_max) {
        ci = ci - ci_len;
        break;
      }
      ci = ci + ci_len;
      if (ci > CiPad) {
        ci = ci - ci_len;
        break;
      }
    }
  }
  EFLOG(DBG) << "ele_num_max:" << ele_num_max;
  EFLOG(DBG) << "conv3d_shape: ef32_mode:" << ef32_mode;
  EFLOG(DBG) << "conv3d_shape: bpe:" << bpe << "; ho:" << ho;
  EFLOG(DBG) << "wo:" << wo;
  EFLOG(DBG) << "hi:" << hi;
  EFLOG(DBG) << "wi:" << wi;
  EFLOG(DBG) << "R:" << R;
  EFLOG(DBG) << "S:" << S;
  EFLOG(DBG) << "co:" << co;

  partition_params.csb_out_shape = {n, d, ho, wo, co};
  partition_params.sip_out_shape = {n, d, ho, wo, co};
  if (conv_attr.input_feature_dimension ==
      conv_attr.kernel_input_feature_dimension + 1) {
    // ff
    EFLOG(DBG) << "InitPartitionParam ff mode !!!";
    partition_params.csb_rhs_shape = {1, R, S, ci, co};
    partition_params.sip_rhs_shape = {1, R, S, ci, co};
  } else {
    // bpi
    EFLOG(DBG) << "InitPartitionParam bpi mode !!!";
    partition_params.csb_rhs_shape = {1, R, S, co, ci};
    partition_params.sip_rhs_shape = {1, R, S, ci, co};
  }

  partition_params.sip_lhs_shape = {n, d, hi, wi, ci};
  if (base_dilation_h == 1 && base_dilation_w == 1) {
    partition_params.csb_lhs_shape = {n, d, hi, wi, ci};
  } else {
    partition_params.csb_lhs_shape = {n, d, hi / base_dilation_h + 1,
                                      wi / base_dilation_w + 1, ci};
  }

  return partition_params;
}

// bpk
Conv3dPartitionParam InitConv3dPartitionParamBPK(
    const dtu::op::DtuOpContext& context, const int bpe) {
  Conv3dPartitionParam partition_params{};
  // auto partition record to lhs and rhs shape
  bool ef32_mode = false;
  if (::dtu::FLAGS_ENFLAME_ENABLE_EF32) {
    EFLOG(DBG) << "enable enflame ef32 flag";
    ef32_mode = true;
  } else {
    EFLOG(DBG) << "disable enflame ef32 flag";
    ef32_mode = false;
  }
  auto op = context.op_desc.op;
  auto conv_op = llvm::cast<DialectNS::ConvOp>(op);
  auto lhs = conv_op.lhs();
  auto rhs = conv_op.rhs();
  auto lhs_shape = hlir::ShapeUtil::GetShapeFromMlir(lhs);
  auto rhs_shape = hlir::ShapeUtil::GetShapeFromMlir(rhs);
  auto out_shape = hlir::ShapeUtil::GetShapeFromMlir(op->getResult(0));
  auto conv_attr = hlir::getConvAttributes(op);
  int32_t pad_head = conv_attr.low_padding[0];
  int32_t pad_tail = conv_attr.high_padding[0];
  int32_t pad_top = conv_attr.low_padding[1];
  int32_t pad_bot = conv_attr.high_padding[1];
  int32_t pad_left = conv_attr.low_padding[2];
  int32_t pad_right = conv_attr.high_padding[2];
  int32_t stride_d = conv_attr.window_strides[0];
  int32_t stride_h = conv_attr.window_strides[1];
  int32_t stride_w = conv_attr.window_strides[2];
  int32_t base_dilation_d = conv_attr.lhs_dilation[0];
  int32_t base_dilation_h = conv_attr.lhs_dilation[1];
  int32_t base_dilation_w = conv_attr.lhs_dilation[2];
  int32_t window_dilation_d = conv_attr.rhs_dilation[0];
  int32_t window_dilation_h = conv_attr.rhs_dilation[1];
  int32_t window_dilation_w = conv_attr.rhs_dilation[2];
  auto tr = dtu::op::TargetResource(context.target);
  int32_t sip_cnt = Cast64To32Bit(tr.get_sip_number());
  // +++++++++++++ NCDHW*CTRSCo = NDHWCo->DHWNCo ++++++++++++ //
  auto N = lhs_shape[4];
  auto Ci = lhs_shape[0];
  auto Di = lhs_shape[1];
  auto Hi = lhs_shape[2];
  auto Wi = lhs_shape[3];
  auto T = rhs_shape[1];
  auto R = rhs_shape[2];
  auto S = rhs_shape[3];
  auto Co = rhs_shape[4];
  auto Do = out_shape[0];
  auto Ho = out_shape[1];
  auto Wo = out_shape[2];
  int32_t hom = (Ho > 3) ? 3 : Ho;
  int32_t wom = (Wo > 3) ? 3 : Wo;

  // out:(n,do,ho,wo,co)
  // int32_t t = 1;
  // int32_t r = 1;
  int32_t t = 1;
  int32_t r = R > 2 ? 2 : R;
  int32_t s = S > 2 ? 2 : S;
  auto actual_d = (t - 1) * window_dilation_d + 1;
  auto actual_h = (r - 1) * window_dilation_h + 1;
  auto actual_w = (s - 1) * window_dilation_w + 1;
  int32_t n6 = (N + sip_cnt - 1) / sip_cnt;
  int32_t n6_pad = 8;
  if (n6 > 8) {
    n6_pad = 16;
  }
  if (n6 > 16) {
    n6_pad = 32;
  }

  int32_t n = 8;
  int32_t ci_len = 16;
  auto CiPad = ((Ci + ci_len - 1) / ci_len) * ci_len;
  // --- for do1ho1wo1 ---
  int32_t dout = 1;
  int32_t ho = 1;
  int32_t wo = 1;
  // --- ---
  int32_t co = 32;
  int32_t di = 1;
  int32_t hi = actual_h + (ho - 1) * stride_h;
  int32_t wi = actual_w + (wo - 1) * stride_w;
  int32_t ele_num_max = 0;
  if (bpe > 0) {
    ele_num_max = (tr.get_sip_dmem_size() - tr.get_sip_stack_size()) / 2 / bpe;
    ci_len = 64 / bpe;
    co = 128 / bpe;
  }
  if (bpe == 4 && ef32_mode == true) {
    co = 128;
  }

  int32_t sip_size_used = 0;
  auto element_type =
      hlir::ShapeUtil::GetElementTypeFromMlir(conv_op.lhs().getType());
  int32_t ci = ci_len;

  // tune n
  while (1) {
    // sip_size_used = CiPad * di * hi * wi * n + n * dout * ho * wo * co +
    // CiPad *t * r * s * co;
    sip_size_used = ci * hi * wi * n + n * ho * wo * co + ci * r * s * co;
    if (sip_size_used > ele_num_max) {
      n = n >> 1;
      break;
    }
    n = n << 1;
    if (n > n6_pad) {
      n = n >> 1;
      break;
    }
  }

  // tune howo
  while (1) {
    // sip_size_used = CiPad * di * hi * wi * n + n * dout * ho * wo * co +
    // CiPad *t * r * s * co;
    sip_size_used = ci * hi * wi * n + n * ho * wo * co + ci * r * s * co;
    if (sip_size_used > ele_num_max) {
      ho = ho - 1;
      wo = wo - 1;
      break;
    }
    ho = ho + 1;
    wo = wo + 1;
    hi = actual_h + (ho - 1) * stride_h;
    wi = actual_w + (wo - 1) * stride_w;
    if (ho > hom || wo > wom) {
      ho = ho - 1;
      wo = wo - 1;
      break;
    }
    if ((ho == hom || ho == 2) && (ef32_mode == true) && (n == 32)) {
      ho = ho - 1;
      wo = wo - 1;
      break;
    }
  }
  hi = actual_h + (ho - 1) * stride_h;
  wi = actual_w + (wo - 1) * stride_w;

  // tune ci
  while (1) {
    // sip_size_used = CiPad * di * hi * wi * n + n * dout * ho * wo * co +
    // CiPad *t * r * s * co;
    sip_size_used = ci * hi * wi * n + n * ho * wo * co + ci * r * s * co;
    if (sip_size_used > ele_num_max) {
      ci = ci - ci_len;
      break;
    }
    ci = ci + ci_len;
    if (ci > CiPad) {
      ci = ci - ci_len;
      break;
    }
  }

  while (1) {
    // sip_size_used = CiPad * di * hi * wi * n + n * dout * ho * wo * co +
    // CiPad *t * r * s * co;
    sip_size_used = ci * hi * wi * n + n * ho * wo * co + ci * r * s * co;
    if (sip_size_used > ele_num_max) {
      s = s - 1;
      break;
    }
    s = s + 1;
    actual_w = (s - 1) * window_dilation_w + 1;
    wi = actual_w + (wo - 1) * stride_w;
    if (s > S) {
      s = s - 1;
      break;
    }
  }
  actual_w = (s - 1) * window_dilation_w + 1;
  wi = actual_w + (wo - 1) * stride_w;

  while (1) {
    // sip_size_used = CiPad * di * hi * wi * n + n * dout * ho * wo * co +
    // CiPad *t * r * s * co;
    sip_size_used = ci * hi * wi * n + n * ho * wo * co + ci * r * s * co;
    if (sip_size_used > ele_num_max) {
      r = r - 1;
      break;
    }
    r = r + 1;
    actual_h = (r - 1) * window_dilation_h + 1;
    hi = actual_h + (ho - 1) * stride_h;
    if (r > R) {
      r = r - 1;
      break;
    }
  }

  actual_h = (r - 1) * window_dilation_h + 1;
  hi = actual_h + (ho - 1) * stride_h;

  // for debug n8ho1wo1
  // ho = 2;
  // wo = 2;
  // n = 8;

  // ho = 3;
  // wo = 3;
  // n = 8;

  // ho = 3;
  // wo = 3;
  // n = 16;

  // ho = 3;
  // wo = 3;
  // n = 32;

  // r = R > 2 ? 2 : R;
  // s = S > 2 ? 2 : S;
  // actual_w = (s - 1) * window_dilation_w + 1;
  // wi = actual_w + (wo - 1) * stride_w;
  // actual_h = (r - 1) * window_dilation_h + 1;
  // hi = actual_h + (ho - 1) * stride_h;
  // ci = ci_len;

  // actual_d = (t - 1) * window_dilation_d + 1;
  // // tune ci
  // while (1) {
  //   // sip_size_used = CiPad * di * hi * wi * n + n * dout * ho * wo * co +
  //   // CiPad *t * r * s * co;
  //   sip_size_used = ci * hi * wi * n + n * ho * wo * co + ci * r * s * co;
  //   if (sip_size_used > ele_num_max) {
  //     ci = ci - ci_len;
  //     break;
  //   }
  //   ci = ci + ci_len;
  //   if (ci > CiPad) {
  //     ci = ci - ci_len;
  //     break;
  //   }
  // }

  // while (1) {
  //   // sip_size_used = CiPad * di * hi * wi * n + n * dout * ho * wo * co +
  //   // CiPad *t * r * s * co;
  //   sip_size_used = ci * hi * wi * n + n * ho * wo * co + ci * r * s * co;
  //   if (sip_size_used > ele_num_max) {
  //     s = s - 1;
  //     break;
  //   }
  //   s = s + 1;
  //   actual_w = (s - 1) * window_dilation_w + 1;
  //   wi = actual_w + (wo - 1) * stride_w;
  //   if (s > S) {
  //     s = s - 1;
  //     break;
  //   }
  // }

  // while (1) {
  //   // sip_size_used = CiPad * di * hi * wi * n + n * dout * ho * wo * co +
  //   // CiPad *t * r * s * co;
  //   sip_size_used = ci * hi * wi * n + n * ho * wo * co + ci * r * s * co;
  //   if (sip_size_used > ele_num_max) {
  //     r = r - 1;
  //     break;
  //   }
  //   r = r + 1;
  //   actual_h = (r - 1) * window_dilation_h + 1;
  //   hi = actual_h + (ho - 1) * stride_h;
  //   if (r > R) {
  //     r = r - 1;
  //     break;
  //   }
  // }
  // actual_w = (s - 1) * window_dilation_w + 1;
  // wi = actual_w + (wo - 1) * stride_w;
  // actual_h = (r - 1) * window_dilation_h + 1;
  // hi = actual_h + (ho - 1) * stride_h;

  // ci = 16;
  sip_size_used = ci * hi * wi * n + n * ho * wo * co + ci * r * s * co;
  EFLOG(DBG) << "conv3d_Param: n:" << n;
  EFLOG(DBG) << "conv3d_Param: ci:" << ci;
  EFLOG(DBG) << "conv3d_Param: do:" << dout;
  EFLOG(DBG) << "conv3d_Param: ho:" << ho;
  EFLOG(DBG) << "conv3d_Param: wo:" << wo;
  EFLOG(DBG) << "conv3d_Param: co:" << co;
  EFLOG(DBG) << "conv3d_Param: di:" << di;
  EFLOG(DBG) << "conv3d_Param: hi:" << hi;
  EFLOG(DBG) << "conv3d_Param: wi:" << wi;
  EFLOG(DBG) << "conv3d_Param: T:" << T;
  EFLOG(DBG) << "conv3d_Param: R:" << R;
  EFLOG(DBG) << "conv3d_Param: S:" << S;
  EFLOG(DBG) << "conv3d_Param: t:" << t;
  EFLOG(DBG) << "conv3d_Param: r:" << r;
  EFLOG(DBG) << "conv3d_Param: s:" << s;
  EFLOG(DBG) << "conv3d_shape ele_num_max:" << ele_num_max;
  EFLOG(DBG) << "conv3d_shape sip_size_used:" << sip_size_used;
  EFLOG(DBG) << "conv3d_shape: ef32_mode:" << ef32_mode;
  EFLOG(DBG) << "conv3d_shape: bpe:" << bpe;

  EFLOG(DBG) << "InitPartitionParam bpk mode !!!";
  // bpk
  partition_params.csb_out_shape = {1, ho, wo, n, co};  // {dout, ho, wo, n, co}
  partition_params.sip_out_shape = {1, n, ho, wo, co};  // {dout, n, ho, wo, co}

  partition_params.csb_rhs_shape = {ci, 1, r, s, co};  // {ci, t, r, s, co}
  partition_params.sip_rhs_shape = {1, r, s, ci, co};  // {t, r, s, ci, co}

  partition_params.csb_lhs_shape = {ci, 1, hi, wi, n};  // {ci, di, hi, wi, n}
  partition_params.sip_lhs_shape = {1, n, hi, wi, ci};  // {di, n, hi, wi, ci}

  return partition_params;
}

}  // namespace conv3d_utils
}  // namespace factor
