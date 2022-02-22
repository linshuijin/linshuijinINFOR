/* Copyright 2020-2021 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// #include "sip20intrin.h"
// struct memref {
//   char* addr;
//   int offset;
// };

typedef enum { FP16 = 0, BF16 = 1, FP32 = 2 } DataType;

extern "C" void conv3d_ef32_kernelho1wo1(
    memref* lhs_param, memref* rhs_param, memref* out_param, int N, int Hi,
    int Wi, int Ci, int R, int S, int Co, int Ho, int Wo, int stride_h,
    int stride_w, int base_dilation_h, int base_dilation_w,
    int window_dilation_h, int window_dilation_w, int ld_flag, int st_flag) {
  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);

  int addr = reinterpret_cast<int>(lhs_addr >> 6);
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  int in_off1 = ((16 * BPE) >> 6) & 0xffff;
  int ci_offset = (in_off1 << 16) | in_off1;
  tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);
  int in_off2 = ((Ci * (window_dilation_w - 1) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off2 << 16) | in_off2;
  tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);
  int in_off3 =
      (((Ci * Wi * (window_dilation_h - 1) - Ci * (window_dilation_w - 1)) *
        BPE) >>
       6) &
      0xffff;
  ci_offset = (in_off3 << 16) | in_off3;
  tar_t input_offset3 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // set up tar for weight
  addr = reinterpret_cast<int>(rhs_addr >> 6);
  addr = (addr + 4) << 16 | addr;  // thread 1 addr | thread 0 addr
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = (16 * BPE) >> 6;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  co_offset = (16 * 5 * BPE) >> 6;
  tar_t weight_offset1 =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // set up tar for output
  addr = reinterpret_cast<int>(out_addr >> 6);
  addr = (addr + 4) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset_2 = (16 * BPE) >> 6;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset_2 << 16) | co_offset_2, output_addr);
  tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

  smr_t smr;

  v16f32 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10, iv11, iv12,
      iv13, iv14, iv15, iv16, iv17, iv18, iv19, iv20, iv21, iv22, iv23, iv24,
      iv25, iv26, iv27, iv28, iv29, iv30, iv31;
  v16f32 iv_lhs;
  va16f32x4 va;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  if (ld_flag == 0) {
    __dtu_c_movsr2naccovr(0x10001);
  } else {
    __dtu_c_movsr2naccovr(0x1);
  }

#pragma clang loop unroll(disable)
  for (int r = 0; r < R; ++r) {
#pragma clang loop unroll(disable)
    for (int s = 0; s < S; ++s) {
#pragma clang loop unroll(disable)
      for (int ci = 0; ci < Ci; ci += 16) {
        // load weight
        iv_lhs = __dtu_s_tivld_itar(input_addr, input_offset1);
        // load weight
        iv0 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv1 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv2 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv3 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv4 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv5 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv6 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv7 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv8 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv9 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv10 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv11 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv12 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv13 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv14 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv15 = __dtu_s_tivld_itar(weight_addr, weight_offset1);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv0, 0);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv1, 1);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv2, 2);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv3, 3);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv4, 4);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv5, 5);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv6, 6);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv7, 7);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv8, 8);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv9, 9);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv10, 10);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv11, 11);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv12, 12);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv13, 13);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv14, 14);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv15, 15);

        iv16 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv17 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv18 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv19 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv20 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv21 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv22 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv23 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv24 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv25 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv26 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv27 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv28 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv29 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv30 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv31 = __dtu_s_tivld_itar(weight_addr, weight_offset1);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv16, 16);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv17, 17);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv18, 18);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv19, 19);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv20, 20);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv21, 21);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv22, 22);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv23, 23);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv24, 24);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv25, 25);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv26, 26);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv27, 27);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv28, 28);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv29, 29);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv30, 30);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv31, 31);

        iv0 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv1 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv2 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv3 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv4 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv5 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv6 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv7 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv8 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv9 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv10 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv11 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv12 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv13 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv14 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv15 = __dtu_s_tivld_itar(weight_addr, weight_offset1);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv0, 32);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv1, 33);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv2, 34);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv3, 35);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv4, 36);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv5, 37);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv6, 38);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv7, 39);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv8, 40);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv9, 41);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv10, 42);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv11, 43);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv12, 44);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv13, 45);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv14, 46);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv15, 47);

        iv16 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv17 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv18 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv19 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv20 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv21 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv22 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv23 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv24 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv25 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv26 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv27 = __dtu_s_tivld_itar(weight_addr, weight_offset1);
        iv28 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv29 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv30 = __dtu_s_tivld_itar(weight_addr, weight_offset);
        iv31 = __dtu_s_tivld_itar(weight_addr, weight_offset1);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv16, 48);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv17, 49);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv18, 50);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv19, 51);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv20, 52);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv21, 53);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv22, 54);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv23, 55);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv24, 56);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv25, 57);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv26, 58);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv27, 59);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv28, 60);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv29, 61);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv30, 62);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, iv31, 63);

        va = __dtu_m_vmm_mode11_f_vs0(va, iv_lhs, smr);
        __dtu_c_movsr2naccovr(0x1);
      }  // end ci
      iv_lhs = __dtu_s_tivld_itar(input_addr, input_offset2);
    }  // end s
    iv_lhs = __dtu_s_tivld_itar(input_addr, input_offset3);
  }  // end r
  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(va, output_addr, output_offset);
  }
  __dtu_c_movsr2naccovr(0);
}

extern "C" void conv3d_ef32_kernelho3wo3(
    memref* lhs_param, memref* rhs_param, memref* out_param, int N, int Hi,
    int Wi, int Ci, int R, int S, int Co, int Ho, int Wo, int stride_h,
    int stride_w, int base_dilation_h, int base_dilation_w,
    int window_dilation_h, int window_dilation_w, int ld_flag, int st_flag) {
  smr_t smr;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  v16f32 iv_lhs_1,iv_lhs_2,iv_lhs_3,iv_lhs_4,iv_lhs_5,iv_lhs_6,iv_lhs_7,iv_lhs_8,iv_lhs_9;
  va16f32x4 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7, vacc8;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  int actual_s = (S - 1) * window_dilation_w + 1;
  window_dilation_w = (S == 1) ? 1 : window_dilation_w;

  int st_flag_1=0;
  int* dbg_p = (int*)(496 * 1024);
  *dbg_p++ = st_flag_1 + 0X30000000;
  *dbg_p++ = ld_flag;
  *dbg_p++ = Ci;
  *dbg_p++ = lhs_addr;
  *dbg_p++ = rhs_addr;
  *dbg_p++ = out_addr;
  *dbg_p++ = window_dilation_h;
  *dbg_p++ = window_dilation_w;

   // set up tar for input feature
  int addr = lhs_addr >> 6;
  // thread 0 and 1 use the same address
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  // for next output on wo
  int in_off1 = ((Ci * stride_w * BPE) >> 6) & 0xffff;
  int ci_offset = (in_off1 << 16) | in_off1;
  tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);
  // for next output on ho
  int in_off2 =
      ((((Ci * Wi * stride_h) - (Ci * stride_w * 2)) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off2 << 16) | in_off2;
  tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off3 =
      (((16 - Ci * Wi * stride_h * 2 - Ci * stride_w * 2) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off3 << 16) | in_off3;
  tar_t input_offset3 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off4 = (((Ci * window_dilation_w - Ci) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off4 << 16) | in_off4;
  tar_t input_offset4 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // Ci * Wi * window_dilation_h -((actual_s - 1) + 1 + (window_dilation_w - 1))
  int in_off5 =
      (((Ci * Wi * window_dilation_h - Ci * S * window_dilation_w) * BPE) >>
       6) &
      0xffff;
  ci_offset = (in_off5 << 16) | in_off5;
  tar_t input_offset5 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // set up tar for weight
  addr = rhs_addr >> 6;
  addr = (addr + 1) << 16 | addr;
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = ((2*16 * BPE) >> 6) & 0xffff;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // co_offset = ((16 * 5 * BPE) >> 6) & 0xffff;
  // tar_t weight_offset1 =
  //     __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // set up tar for output
  addr = out_addr >> 6;
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  co_offset = ((16 *2* BPE) >> 6) & 0xffff;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
  // co_offset = ((16 * 5 * BPE) >> 6) & 0xffff;
  // tar_t output_offset1 =
  //     __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);


  if (ld_flag == 0) {
    __dtu_c_movsr2naccovr(0x10001);
  } else {
    __dtu_c_movsr2naccovr(0x1);
  }

  // load weight
  vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 0);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 1);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 2);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 3);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 4);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 5);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 6);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 7);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 8);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 9);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 10);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 11);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 12);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 13);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 14);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 15);
  

  vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 16);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 17);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 18);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 19);
  
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 20);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 21);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 22);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 23);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 24);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 25);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 26);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 27);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 28);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 29);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 30);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 31);

  vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 32);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 33);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 34);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 35);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 36);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 37);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 38);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 39);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 40);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 41);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 42);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 43);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 44);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 45);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 46);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 47);
  

  vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 48);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 49);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 50);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 51);
  
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 52);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 53);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 54);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 55);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 56);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 57);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 58);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 59);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 60);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 61);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 62);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 63);
#pragma clang loop unroll(disable)
  for (int r = 0; r < R; ++r) {
#pragma clang loop unroll(disable)
    for (int s = 0; s < S; ++s) {
#pragma clang loop unroll(disable)
      for (int ci = 0; ci < Ci; ci += 16) {
        // load input
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset2);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset2);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset3);
        // load weight
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        // VMM vacc0:8
        vacc0 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc0, iv_lhs_1, smr, vr0, 0);
        vacc1 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc1, iv_lhs_2, smr, vr1, 1);
        vacc2 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc2, iv_lhs_3, smr, vr2, 2);
        vacc3 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc3, iv_lhs_4, smr, vr3, 3);
        vacc4 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc4, iv_lhs_5, smr, vr4, 4);
        vacc5 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc5, iv_lhs_6, smr, vr5, 5);
        vacc6 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc6, iv_lhs_7, smr, vr6, 6);
        vacc7 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc7, iv_lhs_8, smr, vr7, 7);
        vacc8 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc8, iv_lhs_9, smr, vr8, 8);

        
        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 9);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 10);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 11);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 12);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 13);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 14);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 15);
        

        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);


        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 16);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 17);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 18);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 19);
        
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 20);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 21);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 22);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 23);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 24);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 25);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 26);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 27);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 28);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 29);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 30);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 31);

        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);


        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 32);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 33);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 34);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 35);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 36);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 37);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 38);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 39);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 40);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 41);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 42);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 43);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 44);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 45);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 46);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 47);
        

        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?


        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 48);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 49);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 50);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 51);
        
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 52);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 53);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 54);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 55);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 56);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 57);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 58);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 59);

        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 60);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 61);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 62);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 63);
      }  // end ci
      iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset4);
    }  // end s
    iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset5);
  }  // end r

  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(vacc0, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc1, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc2, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc3, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc4, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc5, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc6, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc7, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc8, output_addr, output_offset);
  }
  __dtu_c_movsr2naccovr(0);
}


extern "C" void conv3d_ef32_kernelho8wo8(
    memref* lhs_param, memref* rhs_param, memref* out_param, int N, int Hi,
    int Wi, int Ci, int R, int S, int Co, int Ho, int Wo, int stride_h,
    int stride_w, int base_dilation_h, int base_dilation_w,
    int window_dilation_h, int window_dilation_w, int ld_flag, int st_flag) {
  smr_t smr;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;

  v16f32 iv_lhs_0,iv_lhs_1,iv_lhs_2,iv_lhs_3,iv_lhs_4,iv_lhs_5,iv_lhs_6,iv_lhs_7,iv_lhs_8,iv_lhs_9,
         iv_lhs_10,iv_lhs_11,iv_lhs_12,iv_lhs_13,iv_lhs_14,iv_lhs_15,iv_lhs_16,iv_lhs_17,iv_lhs_18,iv_lhs_19,
         iv_lhs_20,iv_lhs_21,iv_lhs_22,iv_lhs_23,iv_lhs_24,iv_lhs_25,iv_lhs_26,iv_lhs_27,iv_lhs_28,iv_lhs_29,
         iv_lhs_30,iv_lhs_31;
  va16f32x4 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7, vacc8, vacc9,
      vacc10, vacc11, vacc12, vacc13, vacc14, vacc15, vacc16, vacc17, vacc18,
      vacc19, vacc20, vacc21, vacc22, vacc23, vacc24, vacc25, vacc26, vacc27,
      vacc28, vacc29, vacc30, vacc31, vacc32, vacc33, vacc34, vacc35, vacc36,
      vacc37, vacc38, vacc39, vacc40, vacc41, vacc42, vacc43, vacc44, vacc45,
      vacc46, vacc47, vacc48, vacc49, vacc50, vacc51, vacc52, vacc53, vacc54,
      vacc55, vacc56, vacc57, vacc58, vacc59, vacc60, vacc61, vacc62, vacc63;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  int actual_s = (S - 1) * window_dilation_w + 1;
  window_dilation_w = (S == 1) ? 1 : window_dilation_w;

  int st_flag_1=0;
  int* dbg_p = (int*)(496 * 1024);
  *dbg_p++ = st_flag_1 + 0X30000000;
  *dbg_p++ = ld_flag;
  *dbg_p++ = Ci;
  *dbg_p++ = lhs_addr;
  *dbg_p++ = rhs_addr;
  *dbg_p++ = out_addr;
  *dbg_p++ = window_dilation_h;
  *dbg_p++ = window_dilation_w;

 // set up tar for input feature
  int addr = lhs_addr >> 6;
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  // for next output on wo
  int in_off1 = ((Ci * stride_w * BPE) >> 6) & 0xffff;
  int ci_offset = (in_off1 << 16) | in_off1;
  tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);
  // for next output on ho
  int in_off2 =
      ((((Ci * Wi * stride_h) - (Ci * stride_w * 7)) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off2 << 16) | in_off2;
  tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next round on ci
  int in_off3 =
      (((16 - Ci * Wi * stride_h * 7 - Ci * stride_w * 7) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off3 << 16) | in_off3;
  tar_t input_offset3 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next round on s
  int in_off4 = (((Ci * window_dilation_w - Ci) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off4 << 16) | in_off4;
  tar_t input_offset4 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next round on r
  int in_off5 =
      (((Ci * Wi * window_dilation_h - Ci * S * window_dilation_w) * BPE) >>
       6) &
      0xffff;
  ci_offset = (in_off5 << 16) | in_off5;
  tar_t input_offset5 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // set up tar for weight
  addr = rhs_addr >> 6;
  addr = (addr + 1) << 16 | addr;
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = ((32 * BPE) >> 6) & 0xffff;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

   // set up tar for output
  addr = out_addr >> 6;
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  co_offset = ((32 * BPE) >> 6) & 0xffff;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);

  if (ld_flag == 0) {
    __dtu_c_movsr2naccovr(0x10001);
  } else {
    __dtu_c_movsr2naccovr(0x1);
  }

  // load weight
  vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 0);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 1);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 2);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 3);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 4);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 5);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 6);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 7);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 8);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 9);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 10);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 11);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 12);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 13);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 14);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 15);
  

  vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 16);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 17);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 18);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 19);
  
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 20);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 21);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 22);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 23);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 24);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 25);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 26);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 27);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 28);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 29);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 30);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 31);

  vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 32);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 33);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 34);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 35);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 36);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 37);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 38);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 39);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 40);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 41);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 42);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 43);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 44);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 45);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 46);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 47);
  

  vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 48);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 49);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 50);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 51);
  
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 52);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 53);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 54);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 55);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 56);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 57);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 58);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 59);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 60);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 61);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 62);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 63);

#pragma clang loop unroll(disable)
  for (int r = 0; r < R; ++r) {
#pragma clang loop unroll(disable)
    for (int s = 0; s < S; ++s) {
#pragma clang loop unroll(disable)
      for (int ci = 0; ci < Ci; ci += 16) {
        //load input:0-7
        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset2);
        // load weight 0-7
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        // VMM vacc0:7
        vacc0 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc0, iv_lhs_0, smr, vr0, 0);
        vacc1 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc1, iv_lhs_1, smr, vr1, 1);
        vacc2 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc2, iv_lhs_2, smr, vr2, 2);
        vacc3 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc3, iv_lhs_3, smr, vr3, 3);
        vacc4 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc4, iv_lhs_4, smr, vr4, 4);
        vacc5 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc5, iv_lhs_5, smr, vr5, 5);
        vacc6 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc6, iv_lhs_6, smr, vr6, 6);
        vacc7 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc7, iv_lhs_7, smr, vr7, 7);

        // load input:8-15
        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);
        // load weight
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc8 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc8, iv_lhs_8, smr, vr8, 8);
        vacc9 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc9, iv_lhs_9, smr, vr9, 9);
        vacc10 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc10, iv_lhs_10, smr, vr10, 10);
        vacc11 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc11, iv_lhs_11, smr, vr11, 11);
        vacc12 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc12, iv_lhs_12, smr, vr12, 12);
        vacc13 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc13, iv_lhs_13, smr, vr13, 13);
        vacc14 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc14, iv_lhs_14, smr, vr14, 14);
        vacc15 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc15, iv_lhs_15, smr, vr15, 15);
        // load input:16-23
        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc16 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc16, iv_lhs_16, smr, vr16, 16);
        vacc17 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc17, iv_lhs_17, smr, vr17, 17);
        vacc18 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc18, iv_lhs_18, smr, vr18, 18);
        vacc19 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc19, iv_lhs_19, smr, vr19, 19);
        vacc20 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc20, iv_lhs_20, smr, vr20, 20);
        vacc21 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc21, iv_lhs_21, smr, vr21, 21);
        vacc22 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc22, iv_lhs_22, smr, vr22, 22);
        vacc23 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc23, iv_lhs_23, smr, vr23, 23);
        // load input:
        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?

        // // VMM vacc0:31

        vacc24 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc24, iv_lhs_24, smr, vr24, 24);
        vacc25 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc25, iv_lhs_25, smr, vr25, 25);
        vacc26 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc26, iv_lhs_26, smr, vr26, 26);
        vacc27 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc27, iv_lhs_27, smr, vr27, 27);
        vacc28 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc28, iv_lhs_28, smr, vr28, 28);
        vacc29 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc29, iv_lhs_29, smr, vr29, 29);
        vacc30 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc30, iv_lhs_30, smr, vr30, 30);
        vacc31 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc31, iv_lhs_31, smr, vr31, 31);

         // load input:0-7
        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc32 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc32, iv_lhs_0, smr, vr0, 32);
        vacc33 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc33, iv_lhs_1, smr, vr1, 33);
        vacc34 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc34, iv_lhs_2, smr, vr2, 34);
        vacc35 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc35, iv_lhs_3, smr, vr3, 35);
        vacc36 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc36, iv_lhs_4, smr, vr4, 36);
        vacc37 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc37, iv_lhs_5, smr, vr5, 37);
        vacc38 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc38, iv_lhs_6, smr, vr6, 38);
        vacc39 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc39, iv_lhs_7, smr, vr7, 39);
        // load input:8-15
        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc40 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc40, iv_lhs_8, smr, vr8, 40);
        vacc41 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc41, iv_lhs_9, smr, vr9, 41);
        vacc42 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc42, iv_lhs_10, smr, vr10, 42);
        vacc43 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc43, iv_lhs_11, smr, vr11, 43);
        vacc44 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc44, iv_lhs_12, smr, vr12, 44);
        vacc45 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc45, iv_lhs_13, smr, vr13, 45);
        vacc46 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc46, iv_lhs_14, smr, vr14, 46);
        vacc47 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc47, iv_lhs_15, smr, vr15, 47);

        // load input:16-23
        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc48 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc48, iv_lhs_16, smr, vr16, 48);
        vacc49 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc49, iv_lhs_17, smr, vr17, 49);
        vacc50 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc50, iv_lhs_18, smr, vr18, 50);
        vacc51 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc51, iv_lhs_19, smr, vr19, 51);
        vacc52 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc52, iv_lhs_20, smr, vr20, 52);
        vacc53 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc53, iv_lhs_21, smr, vr21, 53);
        vacc54 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc54, iv_lhs_22, smr, vr22, 54);
        vacc55 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc55, iv_lhs_23, smr, vr23, 55);
        // load input:24-31
        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset3);

        // load weight
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?

        // VMM vacc0:31
        vacc56 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc56, iv_lhs_24, smr, vr24, 56);
        vacc57 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc57, iv_lhs_25, smr, vr25, 57);
        vacc58 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc58, iv_lhs_26, smr, vr26, 58);
        vacc59 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc59, iv_lhs_27, smr, vr27, 59);
        vacc60 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc60, iv_lhs_28, smr, vr28, 60);
        vacc61 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc61, iv_lhs_29, smr, vr29, 61);
        vacc62 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc62, iv_lhs_30, smr, vr30, 62);
        vacc63 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc63, iv_lhs_31, smr, vr31, 63);

        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);
      }  // end ci
      iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset4);
    }  // end s
    iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset5);
  }  // end r
  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(vacc0, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc1, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc2, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc3, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc4, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc5, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc6, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc7, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc8, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc9, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc10, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc11, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc12, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc13, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc14, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc15, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc16, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc17, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc18, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc19, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc20, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc21, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc22, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc23, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc24, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc25, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc26, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc27, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc28, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc29, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc30, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc31, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc32, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc33, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc34, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc35, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc36, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc37, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc38, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc39, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc40, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc41, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc42, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc43, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc44, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc45, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc46, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc47, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc48, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc49, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc50, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc51, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc52, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc53, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc54, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc55, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc56, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc57, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc58, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc59, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc60, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc61, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc62, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc63, output_addr, output_offset);
  }
  __dtu_c_movsr2naccovr(0);
}


// extern "C" void conv3d_ef32_kernelho12wo12(
//     memref* lhs_param, memref* rhs_param, memref* out_param, int N, int Hi,
//     int Wi, int Ci, int R, int S, int Co, int Ho, int Wo, int stride_h,
//     int stride_w, int base_dilation_h, int base_dilation_w,
//     int window_dilation_h, int window_dilation_w, int ld_flag, int st_flag) {
//   smr_t smr;
//   v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
//       vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
//       vr25, vr26, vr27, vr28, vr29, vr30, vr31;

//   v16f32 iv_lhs_0,iv_lhs_1,iv_lhs_2,iv_lhs_3,iv_lhs_4,iv_lhs_5,iv_lhs_6,iv_lhs_7,iv_lhs_8,iv_lhs_9,
//          iv_lhs_10,iv_lhs_11,iv_lhs_12,iv_lhs_13,iv_lhs_14,iv_lhs_15,iv_lhs_16,iv_lhs_17,iv_lhs_18,iv_lhs_19,
//          iv_lhs_20,iv_lhs_21,iv_lhs_22,iv_lhs_23,iv_lhs_24,iv_lhs_25,iv_lhs_26,iv_lhs_27,iv_lhs_28,iv_lhs_29,
//          iv_lhs_30,iv_lhs_31;
//   va16f32x4 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7, vacc8, vacc9,
//       vacc10, vacc11, vacc12, vacc13, vacc14, vacc15, vacc16, vacc17, vacc18,
//       vacc19, vacc20, vacc21, vacc22, vacc23, vacc24, vacc25, vacc26, vacc27,
//       vacc28, vacc29, vacc30, vacc31, vacc32, vacc33, vacc34, vacc35, vacc36,
//       vacc37, vacc38, vacc39, vacc40, vacc41, vacc42, vacc43, vacc44, vacc45,
//       vacc46, vacc47, vacc48, vacc49, vacc50, vacc51, vacc52, vacc53, vacc54,
//       vacc55, vacc56, vacc57, vacc58, vacc59, vacc60, vacc61, vacc62, vacc63,
//       vacc64, vacc65, vacc66, vacc67, vacc68, vacc69, vacc70, vacc71, vacc72,
//       vacc73, vacc74, vacc75, vacc76, vacc77, vacc78, vacc79, vacc80, vacc81,
//       vacc82, vacc83, vacc84, vacc85, vacc86, vacc87, vacc88, vacc89, vacc90, 
//       vacc91, vacc92, vacc93, vacc94, vacc95, vacc96, vacc97, vacc98, vacc99,
//       vacc100,vacc101, vacc102, vacc103, vacc104, vacc105, vacc106,  vacc107, vacc108,
//       vacc109,vacc110, vacc111, vacc112, vacc113, vacc114, vacc115,  vacc116, vacc117,
//       vacc118,vacc119, vacc120, vacc121, vacc122, vacc123,  vacc124, vacc125, vacc126,
//       vacc127,vacc128, vacc129, vacc130, vacc131, vacc132,  vacc133, vacc134, vacc135,
//       vacc136,vacc137, vacc138, vacc139, vacc140, vacc141,  vacc142, vacc143;

//   __dtu_c_movsr2vab_lv_s(0);
//   __dtu_c_movsr2vab_m_s1(0);
//   __dtu_c_movsr2vab_m_d(0);

//   int BPE = 4;
//   int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
//   int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
//   int out_addr = reinterpret_cast<int>(out_param->addr);
//   int actual_s = (S - 1) * window_dilation_w + 1;
//   window_dilation_w = (S == 1) ? 1 : window_dilation_w;

//   int st_flag_1=0;
//   int* dbg_p = (int*)(496 * 1024);
//   *dbg_p++ = st_flag_1 + 0X30000000;
//   *dbg_p++ = ld_flag;
//   *dbg_p++ = Ci;
//   *dbg_p++ = lhs_addr;
//   *dbg_p++ = rhs_addr;
//   *dbg_p++ = out_addr;
//   *dbg_p++ = window_dilation_h;
//   *dbg_p++ = window_dilation_w;

//  // set up tar for input feature
//   int addr = lhs_addr >> 6;
//   addr = addr << 16 | addr;
//   tar_t input_addr = __dtu_c_movsr2targ(addr);
//   // for next output on wo
//   int in_off1 = ((Ci * stride_w * BPE) >> 6) & 0xffff;
//   int ci_offset = (in_off1 << 16) | in_off1;
//   tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);
//   // for next output on ho
//   int in_off2 =
//       ((((Ci * Wi * stride_h) - (Ci * stride_w * 11)) * BPE) >> 6) & 0xffff;
//   ci_offset = (in_off2 << 16) | in_off2;
//   tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // for next round on ci
//   int in_off3 =
//       (((16 - Ci * Wi * stride_h * 11 - Ci * stride_w * 11) * BPE) >> 6) & 0xffff;
//   ci_offset = (in_off3 << 16) | in_off3;
//   tar_t input_offset3 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // for next round on s
//   int in_off4 = (((Ci * window_dilation_w - Ci) * BPE) >> 6) & 0xffff;
//   ci_offset = (in_off4 << 16) | in_off4;
//   tar_t input_offset4 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // for next round on r
//   int in_off5 =
//       (((Ci * Wi * window_dilation_h - Ci * S * window_dilation_w) * BPE) >>
//        6) &
//       0xffff;
//   ci_offset = (in_off5 << 16) | in_off5;
//   tar_t input_offset5 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // set up tar for weight
//   addr = rhs_addr >> 6;
//   addr = (addr + 1) << 16 | addr;
//   tar_t weight_addr = __dtu_c_movsr2targ(addr);
//   // Co * 2 >> 6
//   int co_offset = ((32 * BPE) >> 6) & 0xffff;
//   tar_t weight_offset =
//       __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

//    // set up tar for output
//   addr = out_addr >> 6;
//   addr = (addr + 1) << 16 | addr;
//   tar_t output_addr = __dtu_c_movsr2targ(addr);
//   // Co * 2 >> 6
//   co_offset = ((32 * BPE) >> 6) & 0xffff;
//   tar_t output_offset =
//       __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);

//   if (ld_flag == 0) {
//     __dtu_c_movsr2naccovr(0x10001);
//   } else {
//     __dtu_c_movsr2naccovr(0x1);
//   }

//   // load weight
//   vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);


//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 0);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 1);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 2);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 3);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 4);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 5);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 6);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 7);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 8);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 9);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 10);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 11);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 12);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 13);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 14);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 15);
  

//   vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);


//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 16);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 17);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 18);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 19);
  
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 20);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 21);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 22);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 23);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 24);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 25);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 26);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 27);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 28);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 29);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 30);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 31);

//   vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);


//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 32);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 33);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 34);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 35);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 36);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 37);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 38);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 39);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 40);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 41);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 42);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 43);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 44);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 45);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 46);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 47);
  

//   vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//   vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?


//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 48);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 49);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 50);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 51);
  
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 52);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 53);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 54);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 55);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 56);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 57);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 58);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 59);

//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 60);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 61);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 62);
//   smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 63);

// #pragma clang loop unroll(disable)
//   for (int r = 0; r < R; ++r) {
// #pragma clang loop unroll(disable)
//     for (int s = 0; s < S; ++s) {
// #pragma clang loop unroll(disable)
//       for (int ci = 0; ci < Ci; ci += 16) {
//         //load input:0-7
//         iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         // load weight 0-7
//         vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         // VMM vacc0:7
//         vacc0 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc0, iv_lhs_0, smr, vr0, 0);
//         vacc1 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc1, iv_lhs_1, smr, vr1, 1);
//         vacc2 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc2, iv_lhs_2, smr, vr2, 2);
//         vacc3 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc3, iv_lhs_3, smr, vr3, 3);
//         vacc4 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc4, iv_lhs_4, smr, vr4, 4);
//         vacc5 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc5, iv_lhs_5, smr, vr5, 5);
//         vacc6 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc6, iv_lhs_6, smr, vr6, 6);
//         vacc7 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc7, iv_lhs_7, smr, vr7, 7);

//         // load input:8-15
//         iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset2);
//         iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         // load weight
//         vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         vacc8 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc8, iv_lhs_8, smr, vr8, 8);
//         vacc9 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc9, iv_lhs_9, smr, vr9, 9);
//         vacc10 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc10, iv_lhs_10, smr, vr10, 10);
//         vacc11 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc11, iv_lhs_11, smr, vr11, 11);
//         vacc12 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc12, iv_lhs_12, smr, vr12, 12);
//         vacc13 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc13, iv_lhs_13, smr, vr13, 13);
//         vacc14 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc14, iv_lhs_14, smr, vr14, 14);
//         vacc15 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc15, iv_lhs_15, smr, vr15, 15);
//         // load input:16-23
//         iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset2);

//         vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         vacc16 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc16, iv_lhs_16, smr, vr16, 16);
//         vacc17 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc17, iv_lhs_17, smr, vr17, 17);
//         vacc18 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc18, iv_lhs_18, smr, vr18, 18);
//         vacc19 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc19, iv_lhs_19, smr, vr19, 19);
//         vacc20 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc20, iv_lhs_20, smr, vr20, 20);
//         vacc21 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc21, iv_lhs_21, smr, vr21, 21);
//         vacc22 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc22, iv_lhs_22, smr, vr22, 22);
//         vacc23 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc23, iv_lhs_23, smr, vr23, 23);
//         // load input:
//         iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?

//         // // VMM vacc0:31

//         vacc24 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc24, iv_lhs_24, smr, vr24, 24);
//         vacc25 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc25, iv_lhs_25, smr, vr25, 25);
//         vacc26 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc26, iv_lhs_26, smr, vr26, 26);
//         vacc27 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc27, iv_lhs_27, smr, vr27, 27);
//         vacc28 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc28, iv_lhs_28, smr, vr28, 28);
//         vacc29 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc29, iv_lhs_29, smr, vr29, 29);
//         vacc30 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc30, iv_lhs_30, smr, vr30, 30);
//         vacc31 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc31, iv_lhs_31, smr, vr31, 31);

//          // load input:0-7
//         iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset2);
//         iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         vacc32 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc32, iv_lhs_0, smr, vr0, 32);
//         vacc33 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc33, iv_lhs_1, smr, vr1, 33);
//         vacc34 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc34, iv_lhs_2, smr, vr2, 34);
//         vacc35 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc35, iv_lhs_3, smr, vr3, 35);
//         vacc36 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc36, iv_lhs_4, smr, vr4, 36);
//         vacc37 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc37, iv_lhs_5, smr, vr5, 37);
//         vacc38 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc38, iv_lhs_6, smr, vr6, 38);
//         vacc39 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc39, iv_lhs_7, smr, vr7, 39);
//         // load input:8-15
//         iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);

//         vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         vacc40 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc40, iv_lhs_8, smr, vr8, 40);
//         vacc41 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc41, iv_lhs_9, smr, vr9, 41);
//         vacc42 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc42, iv_lhs_10, smr, vr10, 42);
//         vacc43 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc43, iv_lhs_11, smr, vr11, 43);
//         vacc44 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc44, iv_lhs_12, smr, vr12, 44);
//         vacc45 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc45, iv_lhs_13, smr, vr13, 45);
//         vacc46 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc46, iv_lhs_14, smr, vr14, 46);
//         vacc47 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc47, iv_lhs_15, smr, vr15, 47);

//         // load input:16-23
//         iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         vacc48 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc48, iv_lhs_16, smr, vr16, 48);
//         vacc49 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc49, iv_lhs_17, smr, vr17, 49);
//         vacc50 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc50, iv_lhs_18, smr, vr18, 50);
//         vacc51 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc51, iv_lhs_19, smr, vr19, 51);
//         vacc52 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc52, iv_lhs_20, smr, vr20, 52);
//         vacc53 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc53, iv_lhs_21, smr, vr21, 53);
//         vacc54 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc54, iv_lhs_22, smr, vr22, 54);
//         vacc55 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc55, iv_lhs_23, smr, vr23, 55);
//         // load input:24-31
//         iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset2);
//         iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         // load weight
//         vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?

//         // VMM vacc0:31
//         vacc56 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc56, iv_lhs_24, smr, vr24, 56);
//         vacc57 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc57, iv_lhs_25, smr, vr25, 57);
//         vacc58 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc58, iv_lhs_26, smr, vr26, 58);
//         vacc59 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc59, iv_lhs_27, smr, vr27, 59);
//         vacc60 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc60, iv_lhs_28, smr, vr28, 60);
//         vacc61 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc61, iv_lhs_29, smr, vr29, 61);
//         vacc62 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc62, iv_lhs_30, smr, vr30, 62);
//         vacc63 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc63, iv_lhs_31, smr, vr31, 63);

//         iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset2);

//         vacc64 = __dtu_m_vmm_mode11_f_vs0(vacc64, iv_lhs_0, smr);
//         vacc65 = __dtu_m_vmm_mode11_f_vs0(vacc65, iv_lhs_1, smr);
//         vacc66 = __dtu_m_vmm_mode11_f_vs0(vacc66, iv_lhs_2, smr);
//         vacc67 = __dtu_m_vmm_mode11_f_vs0(vacc67, iv_lhs_3, smr);
//         vacc68 = __dtu_m_vmm_mode11_f_vs0(vacc68, iv_lhs_4, smr);
//         vacc69 = __dtu_m_vmm_mode11_f_vs0(vacc69, iv_lhs_5, smr);
//         vacc70 = __dtu_m_vmm_mode11_f_vs0(vacc70, iv_lhs_6, smr);
//         vacc71 = __dtu_m_vmm_mode11_f_vs0(vacc71, iv_lhs_7, smr);

//         iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         vacc72 = __dtu_m_vmm_mode11_f_vs0(vacc72, iv_lhs_8, smr);
//         vacc73 = __dtu_m_vmm_mode11_f_vs0(vacc73, iv_lhs_9, smr);
//         vacc74 = __dtu_m_vmm_mode11_f_vs0(vacc74, iv_lhs_10, smr);
//         vacc75 = __dtu_m_vmm_mode11_f_vs0(vacc75, iv_lhs_11, smr);
//         vacc76 = __dtu_m_vmm_mode11_f_vs0(vacc76, iv_lhs_12, smr);
//         vacc77 = __dtu_m_vmm_mode11_f_vs0(vacc77, iv_lhs_13, smr);
//         vacc78 = __dtu_m_vmm_mode11_f_vs0(vacc78, iv_lhs_14, smr);
//         vacc79 = __dtu_m_vmm_mode11_f_vs0(vacc79, iv_lhs_15, smr);

//         iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset2);
//         iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         vacc80 = __dtu_m_vmm_mode11_f_vs0(vacc80, iv_lhs_16, smr);
//         vacc81 = __dtu_m_vmm_mode11_f_vs0(vacc81, iv_lhs_17, smr);
//         vacc82 = __dtu_m_vmm_mode11_f_vs0(vacc82, iv_lhs_18, smr);
//         vacc83 = __dtu_m_vmm_mode11_f_vs0(vacc83, iv_lhs_19, smr);
//         vacc84 = __dtu_m_vmm_mode11_f_vs0(vacc84, iv_lhs_20, smr);
//         vacc85 = __dtu_m_vmm_mode11_f_vs0(vacc85, iv_lhs_21, smr);
//         vacc86 = __dtu_m_vmm_mode11_f_vs0(vacc86, iv_lhs_22, smr);
//         vacc87 = __dtu_m_vmm_mode11_f_vs0(vacc87, iv_lhs_23, smr);

//         iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset2);

//         vacc88 = __dtu_m_vmm_mode11_f_vs0(vacc88, iv_lhs_24, smr);
//         vacc89 = __dtu_m_vmm_mode11_f_vs0(vacc89, iv_lhs_25, smr);
//         vacc90 = __dtu_m_vmm_mode11_f_vs0(vacc90, iv_lhs_26, smr);
//         vacc91 = __dtu_m_vmm_mode11_f_vs0(vacc91, iv_lhs_27, smr);
//         vacc92 = __dtu_m_vmm_mode11_f_vs0(vacc92, iv_lhs_28, smr);
//         vacc93 = __dtu_m_vmm_mode11_f_vs0(vacc93, iv_lhs_29, smr);
//         vacc94 = __dtu_m_vmm_mode11_f_vs0(vacc94, iv_lhs_30, smr);
//         vacc95 = __dtu_m_vmm_mode11_f_vs0(vacc95, iv_lhs_31, smr);

//         iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         vacc96 = __dtu_m_vmm_mode11_f_vs0(vacc96, iv_lhs_0, smr);
//         vacc97 = __dtu_m_vmm_mode11_f_vs0(vacc97, iv_lhs_1, smr);
//         vacc98 = __dtu_m_vmm_mode11_f_vs0(vacc98, iv_lhs_2, smr);
//         vacc99 = __dtu_m_vmm_mode11_f_vs0(vacc99, iv_lhs_3, smr);
//         vacc100 = __dtu_m_vmm_mode11_f_vs0(vacc100, iv_lhs_4, smr);
//         vacc101 = __dtu_m_vmm_mode11_f_vs0(vacc101, iv_lhs_5, smr);
//         vacc102 = __dtu_m_vmm_mode11_f_vs0(vacc102, iv_lhs_6, smr);
//         vacc103 = __dtu_m_vmm_mode11_f_vs0(vacc103, iv_lhs_7, smr);

//         iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset2);
//         iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         vacc104 = __dtu_m_vmm_mode11_f_vs0(vacc104, iv_lhs_8, smr);
//         vacc105 = __dtu_m_vmm_mode11_f_vs0(vacc105, iv_lhs_9, smr);
//         vacc106 = __dtu_m_vmm_mode11_f_vs0(vacc106, iv_lhs_10, smr);
//         vacc107 = __dtu_m_vmm_mode11_f_vs0(vacc107, iv_lhs_11, smr);
//         vacc108 = __dtu_m_vmm_mode11_f_vs0(vacc108, iv_lhs_12, smr);
//         vacc109 = __dtu_m_vmm_mode11_f_vs0(vacc109, iv_lhs_13, smr);
//         vacc110 = __dtu_m_vmm_mode11_f_vs0(vacc110, iv_lhs_14, smr);
//         vacc111 = __dtu_m_vmm_mode11_f_vs0(vacc111, iv_lhs_15, smr);

//         iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset2);

//         vacc112 = __dtu_m_vmm_mode11_f_vs0(vacc112, iv_lhs_16, smr);
//         vacc113 = __dtu_m_vmm_mode11_f_vs0(vacc113, iv_lhs_17, smr);
//         vacc114 = __dtu_m_vmm_mode11_f_vs0(vacc114, iv_lhs_18, smr);
//         vacc115 = __dtu_m_vmm_mode11_f_vs0(vacc115, iv_lhs_19, smr);
//         vacc116 = __dtu_m_vmm_mode11_f_vs0(vacc116, iv_lhs_20, smr);
//         vacc117 = __dtu_m_vmm_mode11_f_vs0(vacc117, iv_lhs_21, smr);
//         vacc118 = __dtu_m_vmm_mode11_f_vs0(vacc118, iv_lhs_22, smr);
//         vacc119 = __dtu_m_vmm_mode11_f_vs0(vacc119, iv_lhs_23, smr);

//         iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         vacc120 = __dtu_m_vmm_mode11_f_vs0(vacc120, iv_lhs_24, smr);
//         vacc121 = __dtu_m_vmm_mode11_f_vs0(vacc121, iv_lhs_25, smr);
//         vacc122 = __dtu_m_vmm_mode11_f_vs0(vacc122, iv_lhs_26, smr);
//         vacc123 = __dtu_m_vmm_mode11_f_vs0(vacc123, iv_lhs_27, smr);
//         vacc124 = __dtu_m_vmm_mode11_f_vs0(vacc124, iv_lhs_28, smr);
//         vacc125 = __dtu_m_vmm_mode11_f_vs0(vacc125, iv_lhs_29, smr);
//         vacc126 = __dtu_m_vmm_mode11_f_vs0(vacc126, iv_lhs_30, smr);
//         vacc127 = __dtu_m_vmm_mode11_f_vs0(vacc127, iv_lhs_31, smr);

//         iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset2);
//         iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

//         vacc128 = __dtu_m_vmm_mode11_f_vs0(vacc128, iv_lhs_0, smr);
//         vacc129 = __dtu_m_vmm_mode11_f_vs0(vacc129, iv_lhs_1, smr);
//         vacc130 = __dtu_m_vmm_mode11_f_vs0(vacc130, iv_lhs_2, smr);
//         vacc131 = __dtu_m_vmm_mode11_f_vs0(vacc131, iv_lhs_3, smr);
//         vacc132 = __dtu_m_vmm_mode11_f_vs0(vacc132, iv_lhs_4, smr);
//         vacc133 = __dtu_m_vmm_mode11_f_vs0(vacc133, iv_lhs_5, smr);
//         vacc134 = __dtu_m_vmm_mode11_f_vs0(vacc134, iv_lhs_6, smr);
//         vacc135 = __dtu_m_vmm_mode11_f_vs0(vacc135, iv_lhs_7, smr);

//         iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset3);

//         vacc136 = __dtu_m_vmm_mode11_f_vs0(vacc136, iv_lhs_8, smr);
//         vacc137 = __dtu_m_vmm_mode11_f_vs0(vacc137, iv_lhs_9, smr);
//         vacc138 = __dtu_m_vmm_mode11_f_vs0(vacc138, iv_lhs_10, smr);
//         vacc139 = __dtu_m_vmm_mode11_f_vs0(vacc139, iv_lhs_11, smr);
//         vacc140 = __dtu_m_vmm_mode11_f_vs0(vacc140, iv_lhs_12, smr);
//         vacc141 = __dtu_m_vmm_mode11_f_vs0(vacc141, iv_lhs_13, smr);
//         vacc142 = __dtu_m_vmm_mode11_f_vs0(vacc142, iv_lhs_14, smr);
//         vacc143 = __dtu_m_vmm_mode11_f_vs0(vacc143, iv_lhs_15, smr);

//         smr = __dtu_v_swapsmr(smr);
//         __dtu_c_movsr2naccovr(0x1);
//       }  // end ci
//       iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset4);
//     }  // end s
//     iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset5);
//   }  // end r
//   if (st_flag == 1) {
//     __dtu_l_tvsta_w_q(vacc0, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc1, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc2, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc3, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc4, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc5, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc6, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc7, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc8, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc9, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc10, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc11, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc12, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc13, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc14, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc15, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc16, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc17, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc18, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc19, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc20, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc21, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc22, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc23, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc24, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc25, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc26, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc27, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc28, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc29, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc30, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc31, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc32, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc33, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc34, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc35, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc36, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc37, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc38, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc39, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc40, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc41, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc42, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc43, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc44, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc45, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc46, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc47, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc48, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc49, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc50, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc51, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc52, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc53, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc54, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc55, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc56, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc57, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc58, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc59, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc60, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc61, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc62, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc63, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc64, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc65, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc66, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc67, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc68, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc69, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc70, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc71, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc72, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc73, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc74, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc75, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc76, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc77, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc78, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc79, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc80, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc81, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc82, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc83, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc84, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc85, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc86, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc87, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc88, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc89, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc90, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc91, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc92, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc93, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc94, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc95, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc96, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc97, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc98, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc99, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc100, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc101, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc102, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc103, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc104, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc105, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc106, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc107, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc108, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc109, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc110, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc111, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc112, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc113, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc114, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc115, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc116, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc117, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc118, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc119, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc120, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc121, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc122, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc123, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc124, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc125, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc126, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc127, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc128, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc129, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc130, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc131, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc132, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc133, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc134, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc135, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc136, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc137, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc138, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc139, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc140, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc141, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc142, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(vacc143, output_addr, output_offset);
    
//   }
//   __dtu_c_movsr2naccovr(0);
// }

extern "C" void conv3d_ef32_kernelho16wo16(
    memref* lhs_param, memref* rhs_param, memref* out_param, int N, int Hi,
    int Wi, int Ci, int R, int S, int Co, int Ho, int Wo, int stride_h,
    int stride_w, int base_dilation_h, int base_dilation_w,
    int window_dilation_h, int window_dilation_w, int ld_flag, int st_flag) {
  smr_t smr;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;

  v16f32 iv_lhs_0,iv_lhs_1,iv_lhs_2,iv_lhs_3,iv_lhs_4,iv_lhs_5,iv_lhs_6,iv_lhs_7,iv_lhs_8,iv_lhs_9,
         iv_lhs_10,iv_lhs_11,iv_lhs_12,iv_lhs_13,iv_lhs_14,iv_lhs_15,iv_lhs_16,iv_lhs_17,iv_lhs_18,iv_lhs_19,
         iv_lhs_20,iv_lhs_21,iv_lhs_22,iv_lhs_23,iv_lhs_24,iv_lhs_25,iv_lhs_26,iv_lhs_27,iv_lhs_28,iv_lhs_29,
         iv_lhs_30,iv_lhs_31;
  va16f32x4 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7, vacc8, vacc9,
      vacc10, vacc11, vacc12, vacc13, vacc14, vacc15, vacc16, vacc17, vacc18,
      vacc19, vacc20, vacc21, vacc22, vacc23, vacc24, vacc25, vacc26, vacc27,
      vacc28, vacc29, vacc30, vacc31, vacc32, vacc33, vacc34, vacc35, vacc36,
      vacc37, vacc38, vacc39, vacc40, vacc41, vacc42, vacc43, vacc44, vacc45,
      vacc46, vacc47, vacc48, vacc49, vacc50, vacc51, vacc52, vacc53, vacc54,
      vacc55, vacc56, vacc57, vacc58, vacc59, vacc60, vacc61, vacc62, vacc63,
      vacc64, vacc65, vacc66, vacc67, vacc68, vacc69, vacc70, vacc71, vacc72,
      vacc73, vacc74, vacc75, vacc76, vacc77, vacc78, vacc79, vacc80, vacc81,
      vacc82, vacc83, vacc84, vacc85, vacc86, vacc87, vacc88, vacc89, vacc90, 
      vacc91, vacc92, vacc93, vacc94, vacc95, vacc96, vacc97, vacc98, vacc99,
      vacc100,vacc101, vacc102, vacc103, vacc104, vacc105, vacc106,  vacc107, vacc108,
      vacc109,vacc110, vacc111, vacc112, vacc113, vacc114, vacc115,  vacc116, vacc117,
      vacc118,vacc119, vacc120, vacc121, vacc122, vacc123,  vacc124, vacc125, vacc126,
      vacc127,vacc128, vacc129, vacc130, vacc131, vacc132,  vacc133, vacc134, vacc135,
      vacc136,vacc137, vacc138, vacc139, vacc140, vacc141,  vacc142, vacc143,
      vacc144,vacc145,vacc146, vacc147, vacc148, vacc149, vacc150,  vacc151, vacc152, vacc153,
      vacc154,vacc155,vacc156, vacc157, vacc158, vacc159, vacc160,  vacc161, vacc162, vacc163,
      vacc164,vacc165,vacc166, vacc167, vacc168, vacc169, vacc170,  vacc171, vacc172, vacc173,
      vacc174,vacc175,vacc176, vacc177, vacc178, vacc179, vacc180,  vacc181, vacc182, vacc183,
      vacc184,vacc185,vacc186, vacc187, vacc188, vacc189, vacc190,  vacc191, vacc192, vacc193,
      vacc194,vacc195,vacc196, vacc197, vacc198, vacc199, vacc200,  vacc201, vacc202, vacc203,
      vacc204,vacc205,vacc206, vacc207, vacc208, vacc209, vacc210,  vacc211, vacc212, vacc213,
      vacc214,vacc215,vacc216, vacc217, vacc218, vacc219, vacc220,  vacc221, vacc222, vacc223,
      vacc224,vacc225,vacc226, vacc227, vacc228, vacc229, vacc230,  vacc231, vacc232, vacc233,
      vacc234,vacc235,vacc236, vacc237, vacc238, vacc239, vacc240,  vacc241, vacc242, vacc243,
      vacc244,vacc245,vacc246, vacc247, vacc248, vacc249, vacc250,  vacc251, vacc252, vacc253,vacc254, vacc255;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  int actual_s = (S - 1) * window_dilation_w + 1;
  window_dilation_w = (S == 1) ? 1 : window_dilation_w;

  int st_flag_1=0;
  int* dbg_p = (int*)(496 * 1024);
  *dbg_p++ = st_flag_1 + 0X30000000;
  *dbg_p++ = ld_flag;
  *dbg_p++ = Ci;
  *dbg_p++ = lhs_addr;
  *dbg_p++ = rhs_addr;
  *dbg_p++ = out_addr;
  *dbg_p++ = window_dilation_h;
  *dbg_p++ = window_dilation_w;

 // set up tar for input feature
  int addr = lhs_addr >> 6;
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  // for next output on wo
  int in_off1 = ((Ci * stride_w * BPE) >> 6) & 0xffff;
  int ci_offset = (in_off1 << 16) | in_off1;
  tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);
  // for next output on ho
  int in_off2 =
      ((((Ci * Wi * stride_h) - (Ci * stride_w * 15)) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off2 << 16) | in_off2;
  tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next round on ci
  int in_off3 =
      (((16 - Ci * Wi * stride_h * 15 - Ci * stride_w * 15) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off3 << 16) | in_off3;
  tar_t input_offset3 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next round on s
  int in_off4 = (((Ci * window_dilation_w - Ci) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off4 << 16) | in_off4;
  tar_t input_offset4 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next round on r
  int in_off5 =
      (((Ci * Wi * window_dilation_h - Ci * S * window_dilation_w) * BPE) >>
       6) &
      0xffff;
  ci_offset = (in_off5 << 16) | in_off5;
  tar_t input_offset5 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // set up tar for weight
  addr = rhs_addr >> 6;
  addr = (addr + 1) << 16 | addr;
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = ((32 * BPE) >> 6) & 0xffff;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

   // set up tar for output
  addr = out_addr >> 6;
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  co_offset = ((32 * BPE) >> 6) & 0xffff;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);

  if (ld_flag == 0) {
    __dtu_c_movsr2naccovr(0x10001);
  } else {
    __dtu_c_movsr2naccovr(0x1);
  }

  // load weight
  vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 0);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 1);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 2);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 3);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 4);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 5);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 6);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 7);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 8);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 9);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 10);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 11);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 12);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 13);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 14);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 15);
  

  vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 16);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 17);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 18);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 19);
  
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 20);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 21);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 22);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 23);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 24);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 25);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 26);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 27);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 28);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 29);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 30);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 31);

  vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 32);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 33);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 34);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 35);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 36);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 37);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 38);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 39);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 40);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 41);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 42);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 43);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 44);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 45);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 46);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 47);
  

  vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?


  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 48);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 49);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 50);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 51);
  
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 52);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 53);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 54);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 55);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 56);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 57);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 58);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 59);

  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 60);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 61);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 62);
  smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 63);

#pragma clang loop unroll(disable)
  for (int r = 0; r < R; ++r) {
#pragma clang loop unroll(disable)
    for (int s = 0; s < S; ++s) {
#pragma clang loop unroll(disable)
      for (int ci = 0; ci < Ci; ci += 16) {
        //load input:0-7
        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);
        // load weight 0-7
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        // VMM vacc0:7
        vacc0 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc0, iv_lhs_0, smr, vr0, 0);
        vacc1 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc1, iv_lhs_1, smr, vr1, 1);
        vacc2 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc2, iv_lhs_2, smr, vr2, 2);
        vacc3 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc3, iv_lhs_3, smr, vr3, 3);
        vacc4 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc4, iv_lhs_4, smr, vr4, 4);
        vacc5 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc5, iv_lhs_5, smr, vr5, 5);
        vacc6 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc6, iv_lhs_6, smr, vr6, 6);
        vacc7 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc7, iv_lhs_7, smr, vr7, 7);

        // load input:8-15
        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);
        // load weight
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc8 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc8, iv_lhs_8, smr, vr8, 8);
        vacc9 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc9, iv_lhs_9, smr, vr9, 9);
        vacc10 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc10, iv_lhs_10, smr, vr10, 10);
        vacc11 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc11, iv_lhs_11, smr, vr11, 11);
        vacc12 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc12, iv_lhs_12, smr, vr12, 12);
        vacc13 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc13, iv_lhs_13, smr, vr13, 13);
        vacc14 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc14, iv_lhs_14, smr, vr14, 14);
        vacc15 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc15, iv_lhs_15, smr, vr15, 15);
        // load input:16-23
        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc16 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc16, iv_lhs_16, smr, vr16, 16);
        vacc17 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc17, iv_lhs_17, smr, vr17, 17);
        vacc18 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc18, iv_lhs_18, smr, vr18, 18);
        vacc19 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc19, iv_lhs_19, smr, vr19, 19);
        vacc20 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc20, iv_lhs_20, smr, vr20, 20);
        vacc21 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc21, iv_lhs_21, smr, vr21, 21);
        vacc22 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc22, iv_lhs_22, smr, vr22, 22);
        vacc23 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc23, iv_lhs_23, smr, vr23, 23);
        // load input:
        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?

        // // VMM vacc0:31

        vacc24 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc24, iv_lhs_24, smr, vr24, 24);
        vacc25 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc25, iv_lhs_25, smr, vr25, 25);
        vacc26 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc26, iv_lhs_26, smr, vr26, 26);
        vacc27 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc27, iv_lhs_27, smr, vr27, 27);
        vacc28 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc28, iv_lhs_28, smr, vr28, 28);
        vacc29 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc29, iv_lhs_29, smr, vr29, 29);
        vacc30 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc30, iv_lhs_30, smr, vr30, 30);
        vacc31 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc31, iv_lhs_31, smr, vr31, 31);

         // load input:0-7
        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc32 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc32, iv_lhs_0, smr, vr0, 32);
        vacc33 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc33, iv_lhs_1, smr, vr1, 33);
        vacc34 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc34, iv_lhs_2, smr, vr2, 34);
        vacc35 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc35, iv_lhs_3, smr, vr3, 35);
        vacc36 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc36, iv_lhs_4, smr, vr4, 36);
        vacc37 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc37, iv_lhs_5, smr, vr5, 37);
        vacc38 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc38, iv_lhs_6, smr, vr6, 38);
        vacc39 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc39, iv_lhs_7, smr, vr7, 39);
        // load input:8-15
        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc40 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc40, iv_lhs_8, smr, vr8, 40);
        vacc41 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc41, iv_lhs_9, smr, vr9, 41);
        vacc42 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc42, iv_lhs_10, smr, vr10, 42);
        vacc43 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc43, iv_lhs_11, smr, vr11, 43);
        vacc44 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc44, iv_lhs_12, smr, vr12, 44);
        vacc45 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc45, iv_lhs_13, smr, vr13, 45);
        vacc46 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc46, iv_lhs_14, smr, vr14, 46);
        vacc47 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc47, iv_lhs_15, smr, vr15, 47);

        // load input:16-23
        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        vacc48 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc48, iv_lhs_16, smr, vr16, 48);
        vacc49 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc49, iv_lhs_17, smr, vr17, 49);
        vacc50 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc50, iv_lhs_18, smr, vr18, 50);
        vacc51 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc51, iv_lhs_19, smr, vr19, 51);
        vacc52 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc52, iv_lhs_20, smr, vr20, 52);
        vacc53 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc53, iv_lhs_21, smr, vr21, 53);
        vacc54 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc54, iv_lhs_22, smr, vr22, 54);
        vacc55 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc55, iv_lhs_23, smr, vr23, 55);
        // load input:24-31
        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset2);

        // load weight
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);//?

        // VMM vacc0:31
        vacc56 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc56, iv_lhs_24, smr, vr24, 56);
        vacc57 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc57, iv_lhs_25, smr, vr25, 57);
        vacc58 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc58, iv_lhs_26, smr, vr26, 58);
        vacc59 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc59, iv_lhs_27, smr, vr27, 59);
        vacc60 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc60, iv_lhs_28, smr, vr28, 60);
        vacc61 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc61, iv_lhs_29, smr, vr29, 61);
        vacc62 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc62, iv_lhs_30, smr, vr30, 62);
        vacc63 = __dtu_m_vmm_mode11_f_vs0_ld_row(vacc63, iv_lhs_31, smr, vr31, 63);

        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc64 = __dtu_m_vmm_mode11_f_vs0(vacc64, iv_lhs_0, smr);
        vacc65 = __dtu_m_vmm_mode11_f_vs0(vacc65, iv_lhs_1, smr);
        vacc66 = __dtu_m_vmm_mode11_f_vs0(vacc66, iv_lhs_2, smr);
        vacc67 = __dtu_m_vmm_mode11_f_vs0(vacc67, iv_lhs_3, smr);
        vacc68 = __dtu_m_vmm_mode11_f_vs0(vacc68, iv_lhs_4, smr);
        vacc69 = __dtu_m_vmm_mode11_f_vs0(vacc69, iv_lhs_5, smr);
        vacc70 = __dtu_m_vmm_mode11_f_vs0(vacc70, iv_lhs_6, smr);
        vacc71 = __dtu_m_vmm_mode11_f_vs0(vacc71, iv_lhs_7, smr);

        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc72 = __dtu_m_vmm_mode11_f_vs0(vacc72, iv_lhs_8, smr);
        vacc73 = __dtu_m_vmm_mode11_f_vs0(vacc73, iv_lhs_9, smr);
        vacc74 = __dtu_m_vmm_mode11_f_vs0(vacc74, iv_lhs_10, smr);
        vacc75 = __dtu_m_vmm_mode11_f_vs0(vacc75, iv_lhs_11, smr);
        vacc76 = __dtu_m_vmm_mode11_f_vs0(vacc76, iv_lhs_12, smr);
        vacc77 = __dtu_m_vmm_mode11_f_vs0(vacc77, iv_lhs_13, smr);
        vacc78 = __dtu_m_vmm_mode11_f_vs0(vacc78, iv_lhs_14, smr);
        vacc79 = __dtu_m_vmm_mode11_f_vs0(vacc79, iv_lhs_15, smr);

        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc80 = __dtu_m_vmm_mode11_f_vs0(vacc80, iv_lhs_16, smr);
        vacc81 = __dtu_m_vmm_mode11_f_vs0(vacc81, iv_lhs_17, smr);
        vacc82 = __dtu_m_vmm_mode11_f_vs0(vacc82, iv_lhs_18, smr);
        vacc83 = __dtu_m_vmm_mode11_f_vs0(vacc83, iv_lhs_19, smr);
        vacc84 = __dtu_m_vmm_mode11_f_vs0(vacc84, iv_lhs_20, smr);
        vacc85 = __dtu_m_vmm_mode11_f_vs0(vacc85, iv_lhs_21, smr);
        vacc86 = __dtu_m_vmm_mode11_f_vs0(vacc86, iv_lhs_22, smr);
        vacc87 = __dtu_m_vmm_mode11_f_vs0(vacc87, iv_lhs_23, smr);

        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc88 = __dtu_m_vmm_mode11_f_vs0(vacc88, iv_lhs_24, smr);
        vacc89 = __dtu_m_vmm_mode11_f_vs0(vacc89, iv_lhs_25, smr);
        vacc90 = __dtu_m_vmm_mode11_f_vs0(vacc90, iv_lhs_26, smr);
        vacc91 = __dtu_m_vmm_mode11_f_vs0(vacc91, iv_lhs_27, smr);
        vacc92 = __dtu_m_vmm_mode11_f_vs0(vacc92, iv_lhs_28, smr);
        vacc93 = __dtu_m_vmm_mode11_f_vs0(vacc93, iv_lhs_29, smr);
        vacc94 = __dtu_m_vmm_mode11_f_vs0(vacc94, iv_lhs_30, smr);
        vacc95 = __dtu_m_vmm_mode11_f_vs0(vacc95, iv_lhs_31, smr);

        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc96 = __dtu_m_vmm_mode11_f_vs0(vacc96, iv_lhs_0, smr);
        vacc97 = __dtu_m_vmm_mode11_f_vs0(vacc97, iv_lhs_1, smr);
        vacc98 = __dtu_m_vmm_mode11_f_vs0(vacc98, iv_lhs_2, smr);
        vacc99 = __dtu_m_vmm_mode11_f_vs0(vacc99, iv_lhs_3, smr);
        vacc100 = __dtu_m_vmm_mode11_f_vs0(vacc100, iv_lhs_4, smr);
        vacc101 = __dtu_m_vmm_mode11_f_vs0(vacc101, iv_lhs_5, smr);
        vacc102 = __dtu_m_vmm_mode11_f_vs0(vacc102, iv_lhs_6, smr);
        vacc103 = __dtu_m_vmm_mode11_f_vs0(vacc103, iv_lhs_7, smr);

        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc104 = __dtu_m_vmm_mode11_f_vs0(vacc104, iv_lhs_8, smr);
        vacc105 = __dtu_m_vmm_mode11_f_vs0(vacc105, iv_lhs_9, smr);
        vacc106 = __dtu_m_vmm_mode11_f_vs0(vacc106, iv_lhs_10, smr);
        vacc107 = __dtu_m_vmm_mode11_f_vs0(vacc107, iv_lhs_11, smr);
        vacc108 = __dtu_m_vmm_mode11_f_vs0(vacc108, iv_lhs_12, smr);
        vacc109 = __dtu_m_vmm_mode11_f_vs0(vacc109, iv_lhs_13, smr);
        vacc110 = __dtu_m_vmm_mode11_f_vs0(vacc110, iv_lhs_14, smr);
        vacc111 = __dtu_m_vmm_mode11_f_vs0(vacc111, iv_lhs_15, smr);

        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc112 = __dtu_m_vmm_mode11_f_vs0(vacc112, iv_lhs_16, smr);
        vacc113 = __dtu_m_vmm_mode11_f_vs0(vacc113, iv_lhs_17, smr);
        vacc114 = __dtu_m_vmm_mode11_f_vs0(vacc114, iv_lhs_18, smr);
        vacc115 = __dtu_m_vmm_mode11_f_vs0(vacc115, iv_lhs_19, smr);
        vacc116 = __dtu_m_vmm_mode11_f_vs0(vacc116, iv_lhs_20, smr);
        vacc117 = __dtu_m_vmm_mode11_f_vs0(vacc117, iv_lhs_21, smr);
        vacc118 = __dtu_m_vmm_mode11_f_vs0(vacc118, iv_lhs_22, smr);
        vacc119 = __dtu_m_vmm_mode11_f_vs0(vacc119, iv_lhs_23, smr);

        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc120 = __dtu_m_vmm_mode11_f_vs0(vacc120, iv_lhs_24, smr);
        vacc121 = __dtu_m_vmm_mode11_f_vs0(vacc121, iv_lhs_25, smr);
        vacc122 = __dtu_m_vmm_mode11_f_vs0(vacc122, iv_lhs_26, smr);
        vacc123 = __dtu_m_vmm_mode11_f_vs0(vacc123, iv_lhs_27, smr);
        vacc124 = __dtu_m_vmm_mode11_f_vs0(vacc124, iv_lhs_28, smr);
        vacc125 = __dtu_m_vmm_mode11_f_vs0(vacc125, iv_lhs_29, smr);
        vacc126 = __dtu_m_vmm_mode11_f_vs0(vacc126, iv_lhs_30, smr);
        vacc127 = __dtu_m_vmm_mode11_f_vs0(vacc127, iv_lhs_31, smr);

        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc128 = __dtu_m_vmm_mode11_f_vs0(vacc128, iv_lhs_0, smr);
        vacc129 = __dtu_m_vmm_mode11_f_vs0(vacc129, iv_lhs_1, smr);
        vacc130 = __dtu_m_vmm_mode11_f_vs0(vacc130, iv_lhs_2, smr);
        vacc131 = __dtu_m_vmm_mode11_f_vs0(vacc131, iv_lhs_3, smr);
        vacc132 = __dtu_m_vmm_mode11_f_vs0(vacc132, iv_lhs_4, smr);
        vacc133 = __dtu_m_vmm_mode11_f_vs0(vacc133, iv_lhs_5, smr);
        vacc134 = __dtu_m_vmm_mode11_f_vs0(vacc134, iv_lhs_6, smr);
        vacc135 = __dtu_m_vmm_mode11_f_vs0(vacc135, iv_lhs_7, smr);

        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc136 = __dtu_m_vmm_mode11_f_vs0(vacc136, iv_lhs_8, smr);
        vacc137 = __dtu_m_vmm_mode11_f_vs0(vacc137, iv_lhs_9, smr);
        vacc138 = __dtu_m_vmm_mode11_f_vs0(vacc138, iv_lhs_10, smr);
        vacc139 = __dtu_m_vmm_mode11_f_vs0(vacc139, iv_lhs_11, smr);
        vacc140 = __dtu_m_vmm_mode11_f_vs0(vacc140, iv_lhs_12, smr);
        vacc141 = __dtu_m_vmm_mode11_f_vs0(vacc141, iv_lhs_13, smr);
        vacc142 = __dtu_m_vmm_mode11_f_vs0(vacc142, iv_lhs_14, smr);
        vacc143 = __dtu_m_vmm_mode11_f_vs0(vacc143, iv_lhs_15, smr);

        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc144 = __dtu_m_vmm_mode11_f_vs0(vacc144, iv_lhs_16, smr);
        vacc145 = __dtu_m_vmm_mode11_f_vs0(vacc145, iv_lhs_17, smr);
        vacc146 = __dtu_m_vmm_mode11_f_vs0(vacc146, iv_lhs_18, smr);
        vacc147 = __dtu_m_vmm_mode11_f_vs0(vacc147, iv_lhs_19, smr);
        vacc148 = __dtu_m_vmm_mode11_f_vs0(vacc148, iv_lhs_20, smr);
        vacc149 = __dtu_m_vmm_mode11_f_vs0(vacc149, iv_lhs_21, smr);
        vacc150 = __dtu_m_vmm_mode11_f_vs0(vacc150, iv_lhs_22, smr);
        vacc151 = __dtu_m_vmm_mode11_f_vs0(vacc151, iv_lhs_23, smr);

        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc152 = __dtu_m_vmm_mode11_f_vs0(vacc152, iv_lhs_24, smr);
        vacc153 = __dtu_m_vmm_mode11_f_vs0(vacc153, iv_lhs_25, smr);
        vacc154 = __dtu_m_vmm_mode11_f_vs0(vacc154, iv_lhs_26, smr);
        vacc155 = __dtu_m_vmm_mode11_f_vs0(vacc155, iv_lhs_27, smr);
        vacc156 = __dtu_m_vmm_mode11_f_vs0(vacc156, iv_lhs_28, smr);
        vacc157 = __dtu_m_vmm_mode11_f_vs0(vacc157, iv_lhs_29, smr);
        vacc158 = __dtu_m_vmm_mode11_f_vs0(vacc158, iv_lhs_30, smr);
        vacc159 = __dtu_m_vmm_mode11_f_vs0(vacc159, iv_lhs_31, smr);

        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc160 = __dtu_m_vmm_mode11_f_vs0(vacc160, iv_lhs_0, smr);
        vacc161 = __dtu_m_vmm_mode11_f_vs0(vacc161, iv_lhs_1, smr);
        vacc162 = __dtu_m_vmm_mode11_f_vs0(vacc162, iv_lhs_2, smr);
        vacc163 = __dtu_m_vmm_mode11_f_vs0(vacc163, iv_lhs_3, smr);
        vacc164 = __dtu_m_vmm_mode11_f_vs0(vacc164, iv_lhs_4, smr);
        vacc165 = __dtu_m_vmm_mode11_f_vs0(vacc165, iv_lhs_5, smr);
        vacc166 = __dtu_m_vmm_mode11_f_vs0(vacc166, iv_lhs_6, smr);
        vacc167 = __dtu_m_vmm_mode11_f_vs0(vacc167, iv_lhs_7, smr);

        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc168 = __dtu_m_vmm_mode11_f_vs0(vacc168, iv_lhs_8, smr);
        vacc169 = __dtu_m_vmm_mode11_f_vs0(vacc169, iv_lhs_9, smr);
        vacc170 = __dtu_m_vmm_mode11_f_vs0(vacc170, iv_lhs_10, smr);
        vacc171 = __dtu_m_vmm_mode11_f_vs0(vacc171, iv_lhs_11, smr);
        vacc172 = __dtu_m_vmm_mode11_f_vs0(vacc172, iv_lhs_12, smr);
        vacc173 = __dtu_m_vmm_mode11_f_vs0(vacc173, iv_lhs_13, smr);
        vacc174 = __dtu_m_vmm_mode11_f_vs0(vacc174, iv_lhs_14, smr);
        vacc175 = __dtu_m_vmm_mode11_f_vs0(vacc175, iv_lhs_15, smr);

        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc176 = __dtu_m_vmm_mode11_f_vs0(vacc176, iv_lhs_16, smr);
        vacc177 = __dtu_m_vmm_mode11_f_vs0(vacc177, iv_lhs_17, smr);
        vacc178 = __dtu_m_vmm_mode11_f_vs0(vacc178, iv_lhs_18, smr);
        vacc179 = __dtu_m_vmm_mode11_f_vs0(vacc179, iv_lhs_19, smr);
        vacc180 = __dtu_m_vmm_mode11_f_vs0(vacc180, iv_lhs_20, smr);
        vacc181 = __dtu_m_vmm_mode11_f_vs0(vacc181, iv_lhs_21, smr);
        vacc182 = __dtu_m_vmm_mode11_f_vs0(vacc182, iv_lhs_22, smr);
        vacc183 = __dtu_m_vmm_mode11_f_vs0(vacc183, iv_lhs_23, smr);

        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc184 = __dtu_m_vmm_mode11_f_vs0(vacc184, iv_lhs_24, smr);
        vacc185 = __dtu_m_vmm_mode11_f_vs0(vacc185, iv_lhs_25, smr);
        vacc186 = __dtu_m_vmm_mode11_f_vs0(vacc186, iv_lhs_26, smr);
        vacc187 = __dtu_m_vmm_mode11_f_vs0(vacc187, iv_lhs_27, smr);
        vacc188 = __dtu_m_vmm_mode11_f_vs0(vacc188, iv_lhs_28, smr);
        vacc189 = __dtu_m_vmm_mode11_f_vs0(vacc189, iv_lhs_29, smr);
        vacc190 = __dtu_m_vmm_mode11_f_vs0(vacc190, iv_lhs_30, smr);
        vacc191 = __dtu_m_vmm_mode11_f_vs0(vacc191, iv_lhs_31, smr);

        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc192 = __dtu_m_vmm_mode11_f_vs0(vacc192, iv_lhs_0, smr);
        vacc193 = __dtu_m_vmm_mode11_f_vs0(vacc193, iv_lhs_1, smr);
        vacc194 = __dtu_m_vmm_mode11_f_vs0(vacc194, iv_lhs_2, smr);
        vacc195 = __dtu_m_vmm_mode11_f_vs0(vacc195, iv_lhs_3, smr);
        vacc196 = __dtu_m_vmm_mode11_f_vs0(vacc196, iv_lhs_4, smr);
        vacc197 = __dtu_m_vmm_mode11_f_vs0(vacc197, iv_lhs_5, smr);
        vacc198 = __dtu_m_vmm_mode11_f_vs0(vacc198, iv_lhs_6, smr);
        vacc199 = __dtu_m_vmm_mode11_f_vs0(vacc199, iv_lhs_7, smr);

        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc200 = __dtu_m_vmm_mode11_f_vs0(vacc200, iv_lhs_8, smr);
        vacc201 = __dtu_m_vmm_mode11_f_vs0(vacc201, iv_lhs_9, smr);
        vacc202 = __dtu_m_vmm_mode11_f_vs0(vacc202, iv_lhs_10, smr);
        vacc203 = __dtu_m_vmm_mode11_f_vs0(vacc203, iv_lhs_11, smr);
        vacc204 = __dtu_m_vmm_mode11_f_vs0(vacc204, iv_lhs_12, smr);
        vacc205 = __dtu_m_vmm_mode11_f_vs0(vacc205, iv_lhs_13, smr);
        vacc206 = __dtu_m_vmm_mode11_f_vs0(vacc206, iv_lhs_14, smr);
        vacc207 = __dtu_m_vmm_mode11_f_vs0(vacc207, iv_lhs_15, smr);

        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc208 = __dtu_m_vmm_mode11_f_vs0(vacc208, iv_lhs_16, smr);
        vacc209 = __dtu_m_vmm_mode11_f_vs0(vacc209, iv_lhs_17, smr);
        vacc210 = __dtu_m_vmm_mode11_f_vs0(vacc210, iv_lhs_18, smr);
        vacc211 = __dtu_m_vmm_mode11_f_vs0(vacc211, iv_lhs_19, smr);
        vacc212 = __dtu_m_vmm_mode11_f_vs0(vacc212, iv_lhs_20, smr);
        vacc213 = __dtu_m_vmm_mode11_f_vs0(vacc213, iv_lhs_21, smr);
        vacc214 = __dtu_m_vmm_mode11_f_vs0(vacc214, iv_lhs_22, smr);
        vacc215 = __dtu_m_vmm_mode11_f_vs0(vacc215, iv_lhs_23, smr);

        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc216 = __dtu_m_vmm_mode11_f_vs0(vacc216, iv_lhs_24, smr);
        vacc217 = __dtu_m_vmm_mode11_f_vs0(vacc217, iv_lhs_25, smr);
        vacc218 = __dtu_m_vmm_mode11_f_vs0(vacc218, iv_lhs_26, smr);
        vacc219 = __dtu_m_vmm_mode11_f_vs0(vacc219, iv_lhs_27, smr);
        vacc220 = __dtu_m_vmm_mode11_f_vs0(vacc220, iv_lhs_28, smr);
        vacc221 = __dtu_m_vmm_mode11_f_vs0(vacc221, iv_lhs_29, smr);
        vacc222 = __dtu_m_vmm_mode11_f_vs0(vacc222, iv_lhs_30, smr);
        vacc223 = __dtu_m_vmm_mode11_f_vs0(vacc223, iv_lhs_31, smr);

        iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_1 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_2 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_3 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_4 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_5 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_6 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_7 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc224 = __dtu_m_vmm_mode11_f_vs0(vacc224, iv_lhs_0, smr);
        vacc225 = __dtu_m_vmm_mode11_f_vs0(vacc225, iv_lhs_1, smr);
        vacc226 = __dtu_m_vmm_mode11_f_vs0(vacc226, iv_lhs_2, smr);
        vacc227 = __dtu_m_vmm_mode11_f_vs0(vacc227, iv_lhs_3, smr);
        vacc228 = __dtu_m_vmm_mode11_f_vs0(vacc228, iv_lhs_4, smr);
        vacc229 = __dtu_m_vmm_mode11_f_vs0(vacc229, iv_lhs_5, smr);
        vacc230 = __dtu_m_vmm_mode11_f_vs0(vacc230, iv_lhs_6, smr);
        vacc231 = __dtu_m_vmm_mode11_f_vs0(vacc231, iv_lhs_7, smr);

        iv_lhs_8 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_9 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_10 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_11 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_12 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_13 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_14 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_15 = __dtu_s_tivld_itar(input_addr, input_offset2);

        vacc232 = __dtu_m_vmm_mode11_f_vs0(vacc232, iv_lhs_8, smr);
        vacc233 = __dtu_m_vmm_mode11_f_vs0(vacc233, iv_lhs_9, smr);
        vacc234 = __dtu_m_vmm_mode11_f_vs0(vacc234, iv_lhs_10, smr);
        vacc235 = __dtu_m_vmm_mode11_f_vs0(vacc235, iv_lhs_11, smr);
        vacc236 = __dtu_m_vmm_mode11_f_vs0(vacc236, iv_lhs_12, smr);
        vacc237 = __dtu_m_vmm_mode11_f_vs0(vacc237, iv_lhs_13, smr);
        vacc238 = __dtu_m_vmm_mode11_f_vs0(vacc238, iv_lhs_14, smr);
        vacc239 = __dtu_m_vmm_mode11_f_vs0(vacc239, iv_lhs_15, smr);

        iv_lhs_16 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_17 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_18 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_19 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_20 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_21 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_22 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_23 = __dtu_s_tivld_itar(input_addr, input_offset1);

        vacc240 = __dtu_m_vmm_mode11_f_vs0(vacc240, iv_lhs_16, smr);
        vacc241 = __dtu_m_vmm_mode11_f_vs0(vacc241, iv_lhs_17, smr);
        vacc242 = __dtu_m_vmm_mode11_f_vs0(vacc242, iv_lhs_18, smr);
        vacc243 = __dtu_m_vmm_mode11_f_vs0(vacc243, iv_lhs_19, smr);
        vacc244 = __dtu_m_vmm_mode11_f_vs0(vacc244, iv_lhs_20, smr);
        vacc245 = __dtu_m_vmm_mode11_f_vs0(vacc245, iv_lhs_21, smr);
        vacc246 = __dtu_m_vmm_mode11_f_vs0(vacc246, iv_lhs_22, smr);
        vacc247 = __dtu_m_vmm_mode11_f_vs0(vacc247, iv_lhs_23, smr);

        iv_lhs_24 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_25 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_26 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_27 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_28 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_29 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_30 = __dtu_s_tivld_itar(input_addr, input_offset1);
        iv_lhs_31 = __dtu_s_tivld_itar(input_addr, input_offset3);

        vacc248 = __dtu_m_vmm_mode11_f_vs0(vacc248, iv_lhs_24, smr);
        vacc249 = __dtu_m_vmm_mode11_f_vs0(vacc249, iv_lhs_25, smr);
        vacc250 = __dtu_m_vmm_mode11_f_vs0(vacc250, iv_lhs_26, smr);
        vacc251 = __dtu_m_vmm_mode11_f_vs0(vacc251, iv_lhs_27, smr);
        vacc252 = __dtu_m_vmm_mode11_f_vs0(vacc252, iv_lhs_28, smr);
        vacc253 = __dtu_m_vmm_mode11_f_vs0(vacc253, iv_lhs_29, smr);
        vacc254 = __dtu_m_vmm_mode11_f_vs0(vacc254, iv_lhs_30, smr);
        vacc255 = __dtu_m_vmm_mode11_f_vs0(vacc255, iv_lhs_31, smr);

        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);
      }  // end ci
      iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset4);
    }  // end s
    iv_lhs_0 = __dtu_s_tivld_itar(input_addr, input_offset5);
  }  // end r
  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(vacc0, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc1, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc2, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc3, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc4, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc5, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc6, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc7, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc8, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc9, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc10, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc11, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc12, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc13, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc14, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc15, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc16, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc17, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc18, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc19, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc20, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc21, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc22, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc23, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc24, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc25, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc26, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc27, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc28, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc29, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc30, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc31, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc32, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc33, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc34, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc35, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc36, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc37, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc38, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc39, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc40, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc41, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc42, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc43, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc44, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc45, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc46, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc47, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc48, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc49, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc50, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc51, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc52, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc53, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc54, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc55, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc56, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc57, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc58, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc59, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc60, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc61, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc62, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc63, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc64, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc65, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc66, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc67, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc68, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc69, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc70, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc71, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc72, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc73, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc74, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc75, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc76, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc77, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc78, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc79, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc80, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc81, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc82, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc83, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc84, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc85, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc86, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc87, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc88, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc89, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc90, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc91, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc92, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc93, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc94, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc95, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc96, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc97, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc98, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc99, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc100, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc101, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc102, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc103, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc104, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc105, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc106, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc107, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc108, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc109, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc110, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc111, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc112, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc113, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc114, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc115, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc116, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc117, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc118, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc119, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc120, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc121, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc122, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc123, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc124, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc125, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc126, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc127, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc128, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc129, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc130, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc131, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc132, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc133, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc134, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc135, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc136, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc137, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc138, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc139, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc140, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc141, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc142, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc143, output_addr, output_offset);
    
    __dtu_l_tvsta_w_q(vacc144, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc145, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc146, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc147, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc148, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc149, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc150, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc151, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc152, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc153, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc154, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc155, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc156, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc157, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc158, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc159, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc160, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc161, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc162, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc163, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc164, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc165, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc166, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc167, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc168, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc169, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc170, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc171, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc172, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc173, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc174, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc175, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc176, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc177, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc178, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc179, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc180, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc181, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc182, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc183, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc184, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc185, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc186, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc187, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc188, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc189, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc190, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc191, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc192, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc193, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc194, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc195, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc196, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc197, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc198, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc199, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc200, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc201, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc202, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc203, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc204, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc205, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc206, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc207, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc208, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc209, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc210, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc211, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc212, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc213, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc214, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc215, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc216, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc217, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc218, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc219, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc220, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc221, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc222, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc223, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc224, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc225, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc226, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc227, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc228, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc229, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc230, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc231, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc232, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc233, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc234, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc235, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc236, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc237, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc238, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc239, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc240, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc241, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc242, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc243, output_addr, output_offset);

    __dtu_l_tvsta_w_q(vacc244, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc245, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc246, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc247, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc248, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc249, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc250, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc251, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc252, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc253, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc254, output_addr, output_offset);
    __dtu_l_tvsta_w_q(vacc255, output_addr, output_offset);
    
  }
  __dtu_c_movsr2naccovr(0);
}


extern "C" void conv3d_ef32_kernel(memref* lhs_param, memref* rhs_param,
                                   memref* out_param, int N, int Hi, int Wi,
                                   int Ci, int R, int S, int Co, int Ho, int Wo,
                                   int stride_h, int stride_w,
                                   int base_dilation_h, int base_dilation_w,
                                   int window_dilation_h, int window_dilation_w,
                                   int ld_flag, int st_flag) {
  if (Ho == 8) {
     conv3d_ef32_kernelho8wo8(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                           Co, Ho, Wo, stride_h, stride_w, base_dilation_h,
                           base_dilation_w, window_dilation_h,
                           window_dilation_w, ld_flag, st_flag);
  } else if (Ho == 3) {
    conv3d_ef32_kernelho3wo3(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                           Co, Ho, Wo, stride_h, stride_w, base_dilation_h,
                           base_dilation_w, window_dilation_h,
                           window_dilation_w, ld_flag, st_flag);
  }else if (Ho == 1) {
    conv3d_ef32_kernelho1wo1(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                        Co, Ho, Wo, stride_h, stride_w, base_dilation_h,
                        base_dilation_w, window_dilation_h, window_dilation_w,
                        ld_flag, st_flag);
  }else if (Ho == 16) {
    conv3d_ef32_kernelho16wo16(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                        Co, Ho, Wo, stride_h, stride_w, base_dilation_h,
                        base_dilation_w, window_dilation_h, window_dilation_w,
                        ld_flag, st_flag);
  }

}

