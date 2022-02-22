/* Copyright 2018-2022 The Enflame Tech Company. All Rights Reserved.

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
extern "C" void conv3d_kerneln8ho1wo1(memref* lhs_param, memref* rhs_param,
                                      memref* out_param, int N, int Hi, int Wi,
                                      int Ci, int R, int S, int Co, int Ho,
                                      int Wo, int stride_h, int stride_w,
                                      int window_dilation_h,
                                      int window_dilation_w, int ld_flag,
                                      int st_flag) {
  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  smr_t smr;
  v16f32 iv_lhs1, iv_lhs2, iv_lhs3, iv_lhs4, iv_lhs5, iv_lhs6, iv_lhs7, iv_lhs8;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  va16f32x4 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;
  // for dtu-dgb debug
  int* dbg_p = (int*)(496 * 1024);
  *dbg_p++ = st_flag + 0X30000000;
  *dbg_p++ = ld_flag;
  *dbg_p++ = Ci;
  *dbg_p++ = lhs_addr;
  *dbg_p++ = rhs_addr;
  *dbg_p++ = out_addr;
  *dbg_p++ = window_dilation_h;
  *dbg_p++ = window_dilation_w;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  // set up tar for input feature
  int addr = reinterpret_cast<int>(lhs_addr >> 6);
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  int in_off0 = ((Hi * Wi * Ci * BPE) >> 6) & 0xffff;
  ;
  int ci_offset = (in_off0 << 16) | in_off0;
  tar_t input_offset0 = __dtu_c_movsr2tari(ci_offset, input_addr);
  int in_off1 = ((16 * BPE - Hi * Wi * Ci * (N - 1) * BPE) >> 6) & 0xffff;
  *dbg_p++ = N;
  *dbg_p++ = ((Hi * Wi * Ci * BPE) >> 6);
  *dbg_p++ = in_off0;
  *dbg_p++ = in_off1;

  ci_offset = (in_off1 << 16) | in_off1;
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
  addr = (addr + 1) << 16 | addr;  // thread 1 addr | thread 0 addr
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = (32 * BPE) >> 6;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // set up tar for output
  addr = reinterpret_cast<int>(out_addr >> 6);
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  // int co_offset = (32 * BPE) >> 6;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
  tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

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
        // load input
        iv_lhs1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv_lhs2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv_lhs3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv_lhs4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv_lhs5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv_lhs6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv_lhs7 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv_lhs8 = __dtu_s_tivld_itar(input_addr, input_offset1);
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
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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

        vacc0 = __dtu_m_vmm_mode11_f_vs0(vacc0, iv_lhs1, smr);
        vacc1 = __dtu_m_vmm_mode11_f_vs0(vacc1, iv_lhs2, smr);
        vacc2 = __dtu_m_vmm_mode11_f_vs0(vacc2, iv_lhs3, smr);
        vacc3 = __dtu_m_vmm_mode11_f_vs0(vacc3, iv_lhs4, smr);
        vacc4 = __dtu_m_vmm_mode11_f_vs0(vacc4, iv_lhs5, smr);
        vacc5 = __dtu_m_vmm_mode11_f_vs0(vacc5, iv_lhs6, smr);
        vacc6 = __dtu_m_vmm_mode11_f_vs0(vacc6, iv_lhs7, smr);
        vacc7 = __dtu_m_vmm_mode11_f_vs0(vacc7, iv_lhs8, smr);
        __dtu_c_movsr2naccovr(0x1);
      }  // end ci
      iv_lhs1 = __dtu_s_tivld_itar(input_addr, input_offset2);
    }  // end s
    iv_lhs1 = __dtu_s_tivld_itar(input_addr, input_offset3);
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
  }
  __dtu_c_movsr2naccovr(0);
}

extern "C" void conv3d_kerneln8ho2wo2(memref* lhs_param, memref* rhs_param,
                                      memref* out_param, int N, int Hi, int Wi,
                                      int Ci, int R, int S, int Co, int Ho,
                                      int Wo, int stride_h, int stride_w,
                                      int window_dilation_h,
                                      int window_dilation_w, int ld_flag,
                                      int st_flag) {
  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  smr_t smr;
  v16f32 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10, iv11, iv12,
      iv13, iv14, iv15;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  va16f32x4 va0000, va0001, va0002, va0003, va0004, va0005, va0006, va0007,
      va0100, va0101, va0102, va0103, va0104, va0105, va0106, va0107, va0200,
      va0201, va0202, va0203, va0204, va0205, va0206, va0207, va0300, va0301,
      va0302, va0303, va0304, va0305, va0306, va0307;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  // set up tar for input feature
  int addr = reinterpret_cast<int>(lhs_addr >> 6);
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  // for next output on n
  int in_off0 = ((Hi * Wi * Ci * BPE) >> 6) & 0xffff;
  int ci_offset = (in_off0 << 16) | in_off0;
  tar_t input_offset0 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next output on wo
  int in_off1 =
      ((Ci * stride_w * BPE - Hi * Wi * Ci * (N - 1) * BPE) >> 6) & 0xffff;

  ci_offset = (in_off1 << 16) | in_off1;
  tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next output on ho
  int in_off2 = (((Ci * Wi * stride_h - Ci * stride_w * (Wo - 1) -
                   Hi * Wi * Ci * (N - 1)) *
                  BPE) >>
                 6) &
                0xffff;
  ci_offset = (in_off2 << 16) | in_off2;
  tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off3 = (((16 - Ci * Wi * stride_h * (Ho - 1) -
                   Ci * stride_w * (Wo - 1) - Hi * Wi * Ci * (N - 1)) *
                  BPE) >>
                 6) &
                0xffff;
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
  addr = reinterpret_cast<int>(rhs_addr >> 6);
  addr = (addr + 1) << 16 | addr;  // thread 1 addr | thread 0 addr
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = (32 * BPE) >> 6;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // set up tar for output
  addr = reinterpret_cast<int>(out_addr >> 6);
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  // int co_offset = (32 * BPE) >> 6;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
  tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

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
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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
        // load input[n][ci][0][0]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv7 = __dtu_s_tivld_itar(input_addr, input_offset1);
        // load weight
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load input[n][ci][0][1]
        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0000 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0000, iv0, smr, vr0, 0);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0001 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0001, iv1, smr, vr1, 1);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0002 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0002, iv2, smr, vr2, 2);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0003 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0003, iv3, smr, vr3, 3);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0004 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0004, iv4, smr, vr4, 4);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0005 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0005, iv5, smr, vr5, 5);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0006 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0006, iv6, smr, vr6, 6);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset2);
        va0007 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0007, iv7, smr, vr7, 7);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load weight

        // load input[n][ci][1][0]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0100 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0100, iv8, smr, vr8, 8);
        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0101 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0101, iv9, smr, vr9, 9);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0102 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0102, iv10, smr, vr10, 10);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0103 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0103, iv11, smr, vr11, 11);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0104 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0104, iv12, smr, vr12, 12);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0105 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0105, iv13, smr, vr13, 13);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0106 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0106, iv14, smr, vr14, 14);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0107 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0107, iv15, smr, vr15, 15);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load input[n][ci][1][1]
        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0200 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0200, iv0, smr, vr16, 16);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0201 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0201, iv1, smr, vr17, 17);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0202 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0202, iv2, smr, vr18, 18);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0203 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0203, iv3, smr, vr19, 19);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0204 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0204, iv4, smr, vr20, 20);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0205 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0205, iv5, smr, vr21, 21);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0206 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0206, iv6, smr, vr22, 22);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset3);
        va0207 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0207, iv7, smr, vr23, 23);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        va0300 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0300, iv8, smr, vr24, 24);
        va0301 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0301, iv9, smr, vr25, 25);
        va0302 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0302, iv10, smr, vr26, 26);
        va0303 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0303, iv11, smr, vr27, 27);
        va0304 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0304, iv12, smr, vr28, 28);
        va0305 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0305, iv13, smr, vr29, 29);
        va0306 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0306, iv14, smr, vr30, 30);
        va0307 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0307, iv15, smr, vr31, 31);

        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);

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
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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
      iv0 = __dtu_s_tivld_itar(input_addr, input_offset4);
    }  // end s
    iv0 = __dtu_s_tivld_itar(input_addr, input_offset5);
  }  // end r
  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(va0000, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0100, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0200, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0300, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0001, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0101, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0201, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0301, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0002, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0102, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0202, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0302, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0003, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0103, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0203, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0303, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0004, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0104, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0204, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0304, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0005, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0105, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0205, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0305, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0006, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0106, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0206, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0306, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0007, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0107, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0207, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0307, output_addr, output_offset);
  }

  __dtu_c_movsr2naccovr(0);
}

extern "C" void conv3d_kerneln8ho3wo3(memref* lhs_param, memref* rhs_param,
                                      memref* out_param, int N, int Hi, int Wi,
                                      int Ci, int R, int S, int Co, int Ho,
                                      int Wo, int stride_h, int stride_w,
                                      int window_dilation_h,
                                      int window_dilation_w, int ld_flag,
                                      int st_flag) {
  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  smr_t smr;
  v16f32 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10, iv11, iv12,
      iv13, iv14, iv15;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  va16f32x4 va0000, va0001, va0002, va0003, va0004, va0005, va0006, va0007,
      va0100, va0101, va0102, va0103, va0104, va0105, va0106, va0107, va0200,
      va0201, va0202, va0203, va0204, va0205, va0206, va0207, va0300, va0301,
      va0302, va0303, va0304, va0305, va0306, va0307, va0400, va0401, va0402,
      va0403, va0404, va0405, va0406, va0407, va0500, va0501, va0502, va0503,
      va0504, va0505, va0506, va0507, va0600, va0601, va0602, va0603, va0604,
      va0605, va0606, va0607, va0700, va0701, va0702, va0703, va0704, va0705,
      va0706, va0707, va0800, va0801, va0802, va0803, va0804, va0805, va0806,
      va0807;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  // set up tar for input feature
  int addr = reinterpret_cast<int>(lhs_addr >> 6);
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  // for next output on n
  int in_off0 = ((Hi * Wi * Ci * BPE) >> 6) & 0xffff;
  int ci_offset = (in_off0 << 16) | in_off0;
  tar_t input_offset0 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next output on wo
  int in_off1 =
      ((Ci * stride_w * BPE - Hi * Wi * Ci * (N - 1) * BPE) >> 6) & 0xffff;

  ci_offset = (in_off1 << 16) | in_off1;
  tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next output on ho
  int in_off2 = (((Ci * Wi * stride_h - Ci * stride_w * (Wo - 1) -
                   Hi * Wi * Ci * (N - 1)) *
                  BPE) >>
                 6) &
                0xffff;
  ci_offset = (in_off2 << 16) | in_off2;
  tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off3 = (((16 - Ci * Wi * stride_h * (Ho - 1) -
                   Ci * stride_w * (Wo - 1) - Hi * Wi * Ci * (N - 1)) *
                  BPE) >>
                 6) &
                0xffff;
  ci_offset = (in_off3 << 16) | in_off3;
  tar_t input_offset3 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off4 = (((Ci * window_dilation_w - Ci) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off4 << 16) | in_off4;
  tar_t input_offset4 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off5 =
      (((Ci * Wi * window_dilation_h - Ci * S * window_dilation_w) * BPE) >>
       6) &
      0xffff;
  ci_offset = (in_off5 << 16) | in_off5;
  tar_t input_offset5 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // set up tar for weight
  addr = reinterpret_cast<int>(rhs_addr >> 6);
  addr = (addr + 1) << 16 | addr;  // thread 1 addr | thread 0 addr
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = (32 * BPE) >> 6;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // set up tar for output
  addr = reinterpret_cast<int>(out_addr >> 6);
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  // int co_offset = (32 * BPE) >> 6;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
  tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

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
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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
        // load input[n][ci][0][0]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv7 = __dtu_s_tivld_itar(input_addr, input_offset1);
        // load weight
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load input[n][ci][0][1]
        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0000 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0000, iv0, smr, vr0, 0);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0001 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0001, iv1, smr, vr1, 1);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0002 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0002, iv2, smr, vr2, 2);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0003 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0003, iv3, smr, vr3, 3);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0004 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0004, iv4, smr, vr4, 4);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0005 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0005, iv5, smr, vr5, 5);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0006 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0006, iv6, smr, vr6, 6);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0007 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0007, iv7, smr, vr7, 7);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load weight

        // load input[n][ci][0][2]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0100 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0100, iv8, smr, vr8, 8);
        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0101 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0101, iv9, smr, vr9, 9);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0102 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0102, iv10, smr, vr10, 10);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0103 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0103, iv11, smr, vr11, 11);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0104 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0104, iv12, smr, vr12, 12);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0105 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0105, iv13, smr, vr13, 13);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0106 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0106, iv14, smr, vr14, 14);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset2);
        va0107 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0107, iv15, smr, vr15, 15);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load input[n][ci][1][0]
        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0200 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0200, iv0, smr, vr16, 16);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0201 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0201, iv1, smr, vr17, 17);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0202 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0202, iv2, smr, vr18, 18);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0203 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0203, iv3, smr, vr19, 19);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0204 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0204, iv4, smr, vr20, 20);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0205 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0205, iv5, smr, vr21, 21);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0206 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0206, iv6, smr, vr22, 22);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0207 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0207, iv7, smr, vr23, 23);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load weight

        // load input[n][ci][1][1]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0300 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0300, iv8, smr, vr24, 24);
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0301 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0301, iv9, smr, vr25, 25);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0302 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0302, iv10, smr, vr26, 26);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0303 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0303, iv11, smr, vr27, 27);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0304 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0304, iv12, smr, vr28, 28);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0305 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0305, iv13, smr, vr29, 29);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0306 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0306, iv14, smr, vr30, 30);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0307 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0307, iv15, smr, vr31, 31);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load input[n][ci][1][2]
        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0400 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0400, iv0, smr, vr0, 32);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0401 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0401, iv1, smr, vr1, 33);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0402 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0402, iv2, smr, vr2, 34);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0403 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0403, iv3, smr, vr3, 35);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0404 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0404, iv4, smr, vr4, 36);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0405 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0405, iv5, smr, vr5, 37);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0406 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0406, iv6, smr, vr6, 38);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset2);
        va0407 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0407, iv7, smr, vr7, 39);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load input[n][ci][2][0]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0500 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0500, iv8, smr, vr8, 40);
        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0501 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0501, iv9, smr, vr9, 41);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0502 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0502, iv10, smr, vr10, 42);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0503 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0503, iv11, smr, vr11, 43);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0504 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0504, iv12, smr, vr12, 44);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0505 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0505, iv13, smr, vr13, 45);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0506 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0506, iv14, smr, vr14, 46);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0507 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0507, iv15, smr, vr15, 47);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load input[n][ci][2][1]
        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0600 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0600, iv0, smr, vr16, 48);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0601 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0601, iv1, smr, vr17, 49);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0602 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0602, iv2, smr, vr18, 50);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0603 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0603, iv3, smr, vr19, 51);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0604 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0604, iv4, smr, vr20, 52);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0605 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0605, iv5, smr, vr21, 53);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0606 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0606, iv6, smr, vr22, 54);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0607 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0607, iv7, smr, vr23, 55);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

        // load input[n][ci][2][2]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0700 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0700, iv8, smr, vr24, 56);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0701 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0701, iv9, smr, vr25, 57);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0702 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0702, iv10, smr, vr26, 58);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0703 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0703, iv11, smr, vr27, 59);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0704 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0704, iv12, smr, vr28, 60);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0705 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0705, iv13, smr, vr29, 61);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0706 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0706, iv14, smr, vr30, 62);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset3);
        va0707 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0707, iv15, smr, vr31, 63);

        va0800 = __dtu_m_vmm_mode11_f_vs0(va0800, iv0, smr);
        va0801 = __dtu_m_vmm_mode11_f_vs0(va0801, iv1, smr);
        va0802 = __dtu_m_vmm_mode11_f_vs0(va0802, iv2, smr);
        va0803 = __dtu_m_vmm_mode11_f_vs0(va0803, iv3, smr);
        va0804 = __dtu_m_vmm_mode11_f_vs0(va0804, iv4, smr);
        va0805 = __dtu_m_vmm_mode11_f_vs0(va0805, iv5, smr);
        va0806 = __dtu_m_vmm_mode11_f_vs0(va0806, iv6, smr);
        va0807 = __dtu_m_vmm_mode11_f_vs0(va0807, iv7, smr);

        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);
      }  // end ci
      iv0 = __dtu_s_tivld_itar(input_addr, input_offset4);
    }  // end s
    iv0 = __dtu_s_tivld_itar(input_addr, input_offset5);
  }  // end r
  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(va0000, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0100, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0200, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0300, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0400, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0500, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0600, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0700, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0800, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0001, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0101, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0201, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0301, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0401, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0501, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0601, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0701, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0801, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0002, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0102, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0202, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0302, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0402, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0502, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0602, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0702, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0802, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0003, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0103, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0203, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0303, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0403, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0503, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0603, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0703, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0803, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0004, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0104, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0204, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0304, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0404, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0504, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0604, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0704, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0804, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0005, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0105, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0205, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0305, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0405, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0505, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0605, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0705, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0805, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0006, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0106, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0206, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0306, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0406, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0506, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0606, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0706, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0806, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0007, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0107, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0207, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0307, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0407, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0507, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0607, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0707, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0807, output_addr, output_offset);
  }

  __dtu_c_movsr2naccovr(0);
}

extern "C" void conv3d_kerneln16ho1wo1(memref* lhs_param, memref* rhs_param,
                                       memref* out_param, int N, int Hi, int Wi,
                                       int Ci, int R, int S, int Co, int Ho,
                                       int Wo, int stride_h, int stride_w,
                                       int window_dilation_h,
                                       int window_dilation_w, int ld_flag,
                                       int st_flag) {
  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  smr_t smr;
  v16f32 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10, iv11, iv12,
      iv13, iv14, iv15;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  va16f32x4 va0, va1, va2, va3, va4, va5, va6, va7, va8, va9, va10, va11, va12,
      va13, va14, va15;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  // set up tar for input feature
  int addr = reinterpret_cast<int>(lhs_addr >> 6);
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  int in_off0 = ((Hi * Wi * Ci * BPE) >> 6) & 0xffff;
  ;
  int ci_offset = (in_off0 << 16) | in_off0;
  tar_t input_offset0 = __dtu_c_movsr2tari(ci_offset, input_addr);
  int in_off1 = ((16 * BPE - Hi * Wi * Ci * (N - 1) * BPE) >> 6) & 0xffff;

  ci_offset = (in_off1 << 16) | in_off1;
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
  addr = (addr + 1) << 16 | addr;  // thread 1 addr | thread 0 addr
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = (32 * BPE) >> 6;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // set up tar for output
  addr = reinterpret_cast<int>(out_addr >> 6);
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  // int co_offset = (32 * BPE) >> 6;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
  tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

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
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);

        // load weight
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0, iv0, smr, vr0, 0);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va1 = __dtu_m_vmm_mode11_f_vs0_ld_row(va1, iv1, smr, vr1, 1);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va2 = __dtu_m_vmm_mode11_f_vs0_ld_row(va2, iv2, smr, vr2, 2);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va3 = __dtu_m_vmm_mode11_f_vs0_ld_row(va3, iv3, smr, vr3, 3);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va4 = __dtu_m_vmm_mode11_f_vs0_ld_row(va4, iv4, smr, vr4, 4);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va5 = __dtu_m_vmm_mode11_f_vs0_ld_row(va5, iv5, smr, vr5, 5);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va6 = __dtu_m_vmm_mode11_f_vs0_ld_row(va6, iv6, smr, vr6, 6);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va7 = __dtu_m_vmm_mode11_f_vs0_ld_row(va7, iv7, smr, vr7, 7);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        va8 = __dtu_m_vmm_mode11_f_vs0_ld_row(va8, iv8, smr, vr8, 8);
        va9 = __dtu_m_vmm_mode11_f_vs0_ld_row(va9, iv9, smr, vr9, 9);
        va10 = __dtu_m_vmm_mode11_f_vs0_ld_row(va10, iv10, smr, vr10, 10);
        va11 = __dtu_m_vmm_mode11_f_vs0_ld_row(va11, iv11, smr, vr11, 11);
        va12 = __dtu_m_vmm_mode11_f_vs0_ld_row(va12, iv12, smr, vr12, 12);
        va13 = __dtu_m_vmm_mode11_f_vs0_ld_row(va13, iv13, smr, vr13, 13);
        va14 = __dtu_m_vmm_mode11_f_vs0_ld_row(va14, iv14, smr, vr14, 14);
        va15 = __dtu_m_vmm_mode11_f_vs0_ld_row(va15, iv15, smr, vr15, 15);

        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);

        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 16);

        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 17);

        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 18);

        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 19);

        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 20);

        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 21);

        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 22);

        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 23);

        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 24);

        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 25);

        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 26);

        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 27);

        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 28);

        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 29);

        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 30);

        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 31);

        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 32);

        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 33);

        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 34);

        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 35);

        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 36);

        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 37);

        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 38);

        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 39);

        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 40);

        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 41);

        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 42);

        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 43);

        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 44);

        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 45);

        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 46);

        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 47);

        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 48);

        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 49);

        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 50);

        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 51);

        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 52);

        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 53);

        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 54);

        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 55);

        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 56);

        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 57);

        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 58);

        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 59);

        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 60);

        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 61);

        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 62);

        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 63);

      }  // end ci
      iv0 = __dtu_s_tivld_itar(input_addr, input_offset2);
    }  // end s
    iv0 = __dtu_s_tivld_itar(input_addr, input_offset3);
  }  // end r
  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(va0, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va1, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va2, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va3, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va4, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va5, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va6, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va7, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va8, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va9, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va10, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va11, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va12, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va13, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va14, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va15, output_addr, output_offset);
  }
  __dtu_c_movsr2naccovr(0);
}

extern "C" void conv3d_kerneln16ho2wo2(memref* lhs_param, memref* rhs_param,
                                       memref* out_param, int N, int Hi, int Wi,
                                       int Ci, int R, int S, int Co, int Ho,
                                       int Wo, int stride_h, int stride_w,
                                       int window_dilation_h,
                                       int window_dilation_w, int ld_flag,
                                       int st_flag) {
  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  smr_t smr;
  v16f32 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10, iv11, iv12,
      iv13, iv14, iv15, iv16, iv17, iv18, iv19, iv20, iv21, iv22, iv23, iv24,
      iv25, iv26, iv27, iv28, iv29, iv30, iv31;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  va16f32x4 va0000, va0001, va0002, va0003, va0004, va0005, va0006, va0007,
      va0008, va0009, va0010, va0011, va0012, va0013, va0014, va0015, va0100,
      va0101, va0102, va0103, va0104, va0105, va0106, va0107, va0108, va0109,
      va0110, va0111, va0112, va0113, va0114, va0115, va0200, va0201, va0202,
      va0203, va0204, va0205, va0206, va0207, va0208, va0209, va0210, va0211,
      va0212, va0213, va0214, va0215, va0300, va0301, va0302, va0303, va0304,
      va0305, va0306, va0307, va0308, va0309, va0310, va0311, va0312, va0313,
      va0314, va0315;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  // set up tar for input feature
  int addr = reinterpret_cast<int>(lhs_addr >> 6);
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  // for next output on n
  int in_off0 = ((Hi * Wi * Ci * BPE) >> 6) & 0xffff;
  int ci_offset = (in_off0 << 16) | in_off0;
  tar_t input_offset0 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next output on wo
  int in_off1 =
      ((Ci * stride_w * BPE - Hi * Wi * Ci * (N - 1) * BPE) >> 6) & 0xffff;

  ci_offset = (in_off1 << 16) | in_off1;
  tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next output on ho
  int in_off2 = (((Ci * Wi * stride_h - Ci * stride_w * (Wo - 1) -
                   Hi * Wi * Ci * (N - 1)) *
                  BPE) >>
                 6) &
                0xffff;
  ci_offset = (in_off2 << 16) | in_off2;
  tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off3 = (((16 - Ci * Wi * stride_h * (Ho - 1) -
                   Ci * stride_w * (Wo - 1) - Hi * Wi * Ci * (N - 1)) *
                  BPE) >>
                 6) &
                0xffff;
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
  addr = reinterpret_cast<int>(rhs_addr >> 6);
  addr = (addr + 1) << 16 | addr;  // thread 1 addr | thread 0 addr
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = (32 * BPE) >> 6;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // set up tar for output
  addr = reinterpret_cast<int>(out_addr >> 6);
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  // int co_offset = (32 * BPE) >> 6;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
  tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

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
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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
        // load input[n][ci][0][0]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);

        // load weight
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0000 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0000, iv0, smr, vr0, 0);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0001 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0001, iv1, smr, vr1, 1);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0002 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0002, iv2, smr, vr2, 2);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0003 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0003, iv3, smr, vr3, 3);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0004 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0004, iv4, smr, vr4, 4);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0005 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0005, iv5, smr, vr5, 5);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0006 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0006, iv6, smr, vr6, 6);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0007 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0007, iv7, smr, vr7, 7);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load weight

        // load input[n][ci][0][1]
        iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0008 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0008, iv8, smr, vr8, 8);
        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0009 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0009, iv9, smr, vr9, 9);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0010 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0010, iv10, smr, vr10, 10);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0011 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0011, iv11, smr, vr11, 11);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0012 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0012, iv12, smr, vr12, 12);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0013 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0013, iv13, smr, vr13, 13);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0014 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0014, iv14, smr, vr14, 14);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0015 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0015, iv15, smr, vr15, 15);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0100 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0100, iv16, smr, vr16, 16);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0101 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0101, iv17, smr, vr17, 17);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0102 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0102, iv18, smr, vr18, 18);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0103 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0103, iv19, smr, vr19, 19);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0104 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0104, iv20, smr, vr20, 20);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0105 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0105, iv21, smr, vr21, 21);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0106 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0106, iv22, smr, vr22, 22);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv31 = __dtu_s_tivld_itar(input_addr, input_offset2);
        va0107 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0107, iv23, smr, vr23, 23);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load weight

        // load input[n][ci][1][0]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0108 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0108, iv24, smr, vr24, 24);
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0109 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0109, iv25, smr, vr25, 25);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0110 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0110, iv26, smr, vr26, 26);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0111 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0111, iv27, smr, vr27, 27);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0112 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0112, iv28, smr, vr28, 28);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0113 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0113, iv29, smr, vr29, 29);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0114 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0114, iv30, smr, vr30, 30);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0115 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0115, iv31, smr, vr31, 31);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0200 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0200, iv0, smr, vr0, 32);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0201 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0201, iv1, smr, vr1, 33);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0202 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0202, iv2, smr, vr2, 34);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0203 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0203, iv3, smr, vr3, 35);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0204 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0204, iv4, smr, vr4, 36);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0205 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0205, iv5, smr, vr5, 37);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0206 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0206, iv6, smr, vr6, 38);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0207 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0207, iv7, smr, vr7, 39);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load weight

        // load input[n][ci][1][1]
        iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0208 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0208, iv8, smr, vr8, 40);
        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0209 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0209, iv9, smr, vr9, 41);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0210 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0210, iv10, smr, vr10, 42);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0211 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0211, iv11, smr, vr11, 43);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0212 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0212, iv12, smr, vr12, 44);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0213 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0213, iv13, smr, vr13, 45);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0214 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0214, iv14, smr, vr14, 46);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0215 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0215, iv15, smr, vr15, 47);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0300 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0300, iv16, smr, vr16, 48);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0301 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0301, iv17, smr, vr17, 49);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0302 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0302, iv18, smr, vr18, 50);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0303 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0303, iv19, smr, vr19, 51);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0304 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0304, iv20, smr, vr20, 52);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0305 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0305, iv21, smr, vr21, 53);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0306 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0306, iv22, smr, vr22, 54);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv31 = __dtu_s_tivld_itar(input_addr, input_offset3);
        va0307 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0307, iv23, smr, vr23, 55);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        va0308 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0308, iv24, smr, vr24, 56);
        va0309 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0309, iv25, smr, vr25, 57);
        va0310 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0310, iv26, smr, vr26, 58);
        va0311 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0311, iv27, smr, vr27, 59);
        va0312 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0312, iv28, smr, vr28, 60);
        va0313 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0313, iv29, smr, vr29, 61);
        va0314 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0314, iv30, smr, vr30, 62);
        va0315 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0315, iv31, smr, vr31, 63);

        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);
      }  // end ci
      iv0 = __dtu_s_tivld_itar(input_addr, input_offset4);
    }  // end s
    iv0 = __dtu_s_tivld_itar(input_addr, input_offset5);
  }  // end r
  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(va0000, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0100, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0200, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0300, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0001, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0101, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0201, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0301, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0002, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0102, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0202, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0302, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0003, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0103, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0203, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0303, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0004, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0104, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0204, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0304, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0005, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0105, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0205, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0305, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0006, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0106, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0206, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0306, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0007, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0107, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0207, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0307, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0008, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0108, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0208, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0308, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0009, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0109, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0209, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0309, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0010, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0110, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0210, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0310, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0011, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0111, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0211, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0311, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0012, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0112, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0212, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0312, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0013, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0113, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0213, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0313, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0014, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0114, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0214, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0314, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0015, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0115, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0215, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0315, output_addr, output_offset);
  }

  __dtu_c_movsr2naccovr(0);
}

extern "C" void conv3d_kerneln16ho3wo3(memref* lhs_param, memref* rhs_param,
                                       memref* out_param, int N, int Hi, int Wi,
                                       int Ci, int R, int S, int Co, int Ho,
                                       int Wo, int stride_h, int stride_w,
                                       int window_dilation_h,
                                       int window_dilation_w, int ld_flag,
                                       int st_flag) {
  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  smr_t smr;
  v16f32 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10, iv11, iv12,
      iv13, iv14, iv15, iv16, iv17, iv18, iv19, iv20, iv21, iv22, iv23, iv24,
      iv25, iv26, iv27, iv28, iv29, iv30, iv31;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  va16f32x4 va0000, va0001, va0002, va0003, va0004, va0005, va0006, va0007,
      va0008, va0009, va0010, va0011, va0012, va0013, va0014, va0015, va0100,
      va0101, va0102, va0103, va0104, va0105, va0106, va0107, va0108, va0109,
      va0110, va0111, va0112, va0113, va0114, va0115, va0200, va0201, va0202,
      va0203, va0204, va0205, va0206, va0207, va0208, va0209, va0210, va0211,
      va0212, va0213, va0214, va0215, va0300, va0301, va0302, va0303, va0304,
      va0305, va0306, va0307, va0308, va0309, va0310, va0311, va0312, va0313,
      va0314, va0315, va0400, va0401, va0402, va0403, va0404, va0405, va0406,
      va0407, va0408, va0409, va0410, va0411, va0412, va0413, va0414, va0415,
      va0500, va0501, va0502, va0503, va0504, va0505, va0506, va0507, va0508,
      va0509, va0510, va0511, va0512, va0513, va0514, va0515, va0600, va0601,
      va0602, va0603, va0604, va0605, va0606, va0607, va0608, va0609, va0610,
      va0611, va0612, va0613, va0614, va0615, va0700, va0701, va0702, va0703,
      va0704, va0705, va0706, va0707, va0708, va0709, va0710, va0711, va0712,
      va0713, va0714, va0715, va0800, va0801, va0802, va0803, va0804, va0805,
      va0806, va0807, va0808, va0809, va0810, va0811, va0812, va0813, va0814,
      va0815, va0900, va0901, va0902, va0903, va0904, va0905, va0906, va0907,
      va0908, va0909, va0910, va0911, va0912, va0913, va0914, va0915;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  // set up tar for input feature
  int addr = reinterpret_cast<int>(lhs_addr >> 6);
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  // for next output on n
  int in_off0 = ((Hi * Wi * Ci * BPE) >> 6) & 0xffff;
  int ci_offset = (in_off0 << 16) | in_off0;
  tar_t input_offset0 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next output on wo
  int in_off1 =
      ((Ci * stride_w * BPE - Hi * Wi * Ci * (N - 1) * BPE) >> 6) & 0xffff;

  ci_offset = (in_off1 << 16) | in_off1;
  tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // for next output on ho
  int in_off2 = (((Ci * Wi * stride_h - Ci * stride_w * (Wo - 1) -
                   Hi * Wi * Ci * (N - 1)) *
                  BPE) >>
                 6) &
                0xffff;
  ci_offset = (in_off2 << 16) | in_off2;
  tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off3 = (((16 - Ci * Wi * stride_h * (Ho - 1) -
                   Ci * stride_w * (Wo - 1) - Hi * Wi * Ci * (N - 1)) *
                  BPE) >>
                 6) &
                0xffff;
  ci_offset = (in_off3 << 16) | in_off3;
  tar_t input_offset3 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off4 = (((Ci * window_dilation_w - Ci) * BPE) >> 6) & 0xffff;
  ci_offset = (in_off4 << 16) | in_off4;
  tar_t input_offset4 = __dtu_c_movsr2tari(ci_offset, input_addr);

  int in_off5 =
      (((Ci * Wi * window_dilation_h - Ci * S * window_dilation_w) * BPE) >>
       6) &
      0xffff;
  ci_offset = (in_off5 << 16) | in_off5;
  tar_t input_offset5 = __dtu_c_movsr2tari(ci_offset, input_addr);

  // set up tar for weight
  addr = reinterpret_cast<int>(rhs_addr >> 6);
  addr = (addr + 1) << 16 | addr;  // thread 1 addr | thread 0 addr
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = (32 * BPE) >> 6;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // set up tar for output
  addr = reinterpret_cast<int>(out_addr >> 6);
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  // int co_offset = (32 * BPE) >> 6;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
  tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

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
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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
        // load input[n][ci][0][0]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);

        // load weight
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0000 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0000, iv0, smr, vr0, 0);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0001 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0001, iv1, smr, vr1, 1);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0002 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0002, iv2, smr, vr2, 2);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0003 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0003, iv3, smr, vr3, 3);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0004 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0004, iv4, smr, vr4, 4);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0005 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0005, iv5, smr, vr5, 5);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0006 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0006, iv6, smr, vr6, 6);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0007 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0007, iv7, smr, vr7, 7);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load weight

        // load input[n][ci][0][1]
        iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0008 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0008, iv8, smr, vr8, 8);
        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0009 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0009, iv9, smr, vr9, 9);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0010 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0010, iv10, smr, vr10, 10);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0011 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0011, iv11, smr, vr11, 11);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0012 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0012, iv12, smr, vr12, 12);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0013 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0013, iv13, smr, vr13, 13);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0014 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0014, iv14, smr, vr14, 14);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0015 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0015, iv15, smr, vr15, 15);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0100 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0100, iv16, smr, vr16, 16);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0101 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0101, iv17, smr, vr17, 17);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0102 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0102, iv18, smr, vr18, 18);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0103 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0103, iv19, smr, vr19, 19);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0104 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0104, iv20, smr, vr20, 20);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0105 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0105, iv21, smr, vr21, 21);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0106 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0106, iv22, smr, vr22, 22);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0107 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0107, iv23, smr, vr23, 23);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load weight

        // load input[n][ci][0][2]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0108 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0108, iv24, smr, vr24, 24);
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0109 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0109, iv25, smr, vr25, 25);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0110 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0110, iv26, smr, vr26, 26);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0111 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0111, iv27, smr, vr27, 27);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0112 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0112, iv28, smr, vr28, 28);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0113 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0113, iv29, smr, vr29, 29);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0114 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0114, iv30, smr, vr30, 30);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0115 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0115, iv31, smr, vr31, 31);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0200 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0200, iv0, smr, vr0, 32);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0201 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0201, iv1, smr, vr1, 33);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0202 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0202, iv2, smr, vr2, 34);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0203 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0203, iv3, smr, vr3, 35);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0204 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0204, iv4, smr, vr4, 36);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0205 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0205, iv5, smr, vr5, 37);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0206 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0206, iv6, smr, vr6, 38);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset2);
        va0207 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0207, iv7, smr, vr7, 39);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load weight

        // load input[n][ci][1][0]
        iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0208 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0208, iv8, smr, vr8, 40);
        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0209 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0209, iv9, smr, vr9, 41);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0210 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0210, iv10, smr, vr10, 42);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0211 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0211, iv11, smr, vr11, 43);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0212 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0212, iv12, smr, vr12, 44);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0213 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0213, iv13, smr, vr13, 45);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0214 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0214, iv14, smr, vr14, 46);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0215 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0215, iv15, smr, vr15, 47);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0300 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0300, iv16, smr, vr16, 48);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0301 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0301, iv17, smr, vr17, 49);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0302 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0302, iv18, smr, vr18, 50);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0303 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0303, iv19, smr, vr19, 51);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0304 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0304, iv20, smr, vr20, 52);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0305 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0305, iv21, smr, vr21, 53);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0306 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0306, iv22, smr, vr22, 54);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0307 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0307, iv23, smr, vr23, 55);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        // load input[n][ci][1][1]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0308 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0308, iv24, smr, vr24, 56);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0309 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0309, iv25, smr, vr25, 57);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0310 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0310, iv26, smr, vr26, 58);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0311 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0311, iv27, smr, vr27, 59);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0312 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0312, iv28, smr, vr28, 60);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0313 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0313, iv29, smr, vr29, 61);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0314 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0314, iv30, smr, vr30, 62);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0315 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0315, iv31, smr, vr31, 63);

        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0400 = __dtu_m_vmm_mode11_f_vs0(va0400, iv0, smr);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0401 = __dtu_m_vmm_mode11_f_vs0(va0401, iv1, smr);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0402 = __dtu_m_vmm_mode11_f_vs0(va0402, iv2, smr);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0403 = __dtu_m_vmm_mode11_f_vs0(va0403, iv3, smr);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0404 = __dtu_m_vmm_mode11_f_vs0(va0404, iv4, smr);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0405 = __dtu_m_vmm_mode11_f_vs0(va0405, iv5, smr);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0406 = __dtu_m_vmm_mode11_f_vs0(va0406, iv6, smr);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0407 = __dtu_m_vmm_mode11_f_vs0(va0407, iv7, smr);

        // load input[n][ci][1][2]
        iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0408 = __dtu_m_vmm_mode11_f_vs0(va0408, iv8, smr);

        iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0409 = __dtu_m_vmm_mode11_f_vs0(va0409, iv9, smr);

        iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0410 = __dtu_m_vmm_mode11_f_vs0(va0410, iv10, smr);

        iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0411 = __dtu_m_vmm_mode11_f_vs0(va0411, iv11, smr);

        iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0412 = __dtu_m_vmm_mode11_f_vs0(va0412, iv12, smr);

        iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0413 = __dtu_m_vmm_mode11_f_vs0(va0413, iv13, smr);

        iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0414 = __dtu_m_vmm_mode11_f_vs0(va0414, iv14, smr);

        iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0415 = __dtu_m_vmm_mode11_f_vs0(va0415, iv15, smr);

        iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0500 = __dtu_m_vmm_mode11_f_vs0(va0500, iv16, smr);

        iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0501 = __dtu_m_vmm_mode11_f_vs0(va0501, iv17, smr);

        iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0502 = __dtu_m_vmm_mode11_f_vs0(va0502, iv18, smr);

        iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0503 = __dtu_m_vmm_mode11_f_vs0(va0503, iv19, smr);

        iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0504 = __dtu_m_vmm_mode11_f_vs0(va0504, iv20, smr);

        iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0505 = __dtu_m_vmm_mode11_f_vs0(va0505, iv21, smr);

        iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0506 = __dtu_m_vmm_mode11_f_vs0(va0506, iv22, smr);

        iv31 = __dtu_s_tivld_itar(input_addr, input_offset2);
        va0507 = __dtu_m_vmm_mode11_f_vs0(va0507, iv23, smr);

        // load input[n][ci][2][0]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0508 = __dtu_m_vmm_mode11_f_vs0(va0508, iv24, smr);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0509 = __dtu_m_vmm_mode11_f_vs0(va0509, iv25, smr);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0510 = __dtu_m_vmm_mode11_f_vs0(va0510, iv26, smr);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0511 = __dtu_m_vmm_mode11_f_vs0(va0511, iv27, smr);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0512 = __dtu_m_vmm_mode11_f_vs0(va0512, iv28, smr);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0513 = __dtu_m_vmm_mode11_f_vs0(va0513, iv29, smr);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0514 = __dtu_m_vmm_mode11_f_vs0(va0514, iv30, smr);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0515 = __dtu_m_vmm_mode11_f_vs0(va0515, iv31, smr);

        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0600 = __dtu_m_vmm_mode11_f_vs0(va0600, iv0, smr);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0601 = __dtu_m_vmm_mode11_f_vs0(va0601, iv1, smr);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0602 = __dtu_m_vmm_mode11_f_vs0(va0602, iv2, smr);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0603 = __dtu_m_vmm_mode11_f_vs0(va0603, iv3, smr);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0604 = __dtu_m_vmm_mode11_f_vs0(va0604, iv4, smr);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0605 = __dtu_m_vmm_mode11_f_vs0(va0605, iv5, smr);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0606 = __dtu_m_vmm_mode11_f_vs0(va0606, iv6, smr);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0607 = __dtu_m_vmm_mode11_f_vs0(va0607, iv7, smr);

        // load input[n][ci][2][1]
        iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0608 = __dtu_m_vmm_mode11_f_vs0(va0608, iv8, smr);

        iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0609 = __dtu_m_vmm_mode11_f_vs0(va0609, iv9, smr);

        iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0610 = __dtu_m_vmm_mode11_f_vs0(va0610, iv10, smr);

        iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0611 = __dtu_m_vmm_mode11_f_vs0(va0611, iv11, smr);

        iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0612 = __dtu_m_vmm_mode11_f_vs0(va0612, iv12, smr);

        iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0613 = __dtu_m_vmm_mode11_f_vs0(va0613, iv13, smr);

        iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0614 = __dtu_m_vmm_mode11_f_vs0(va0614, iv14, smr);

        iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0615 = __dtu_m_vmm_mode11_f_vs0(va0615, iv15, smr);

        iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0700 = __dtu_m_vmm_mode11_f_vs0(va0700, iv16, smr);

        iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0701 = __dtu_m_vmm_mode11_f_vs0(va0701, iv17, smr);

        iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0702 = __dtu_m_vmm_mode11_f_vs0(va0702, iv18, smr);

        iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0703 = __dtu_m_vmm_mode11_f_vs0(va0703, iv19, smr);

        iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0704 = __dtu_m_vmm_mode11_f_vs0(va0704, iv20, smr);

        iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0705 = __dtu_m_vmm_mode11_f_vs0(va0705, iv21, smr);

        iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0706 = __dtu_m_vmm_mode11_f_vs0(va0706, iv22, smr);

        iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va0707 = __dtu_m_vmm_mode11_f_vs0(va0707, iv23, smr);

        // load input[n][ci][2][2]
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0708 = __dtu_m_vmm_mode11_f_vs0(va0708, iv24, smr);

        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0709 = __dtu_m_vmm_mode11_f_vs0(va0709, iv25, smr);

        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0710 = __dtu_m_vmm_mode11_f_vs0(va0710, iv26, smr);

        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0711 = __dtu_m_vmm_mode11_f_vs0(va0711, iv27, smr);

        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0712 = __dtu_m_vmm_mode11_f_vs0(va0712, iv28, smr);

        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0713 = __dtu_m_vmm_mode11_f_vs0(va0713, iv29, smr);

        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0714 = __dtu_m_vmm_mode11_f_vs0(va0714, iv30, smr);

        iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0715 = __dtu_m_vmm_mode11_f_vs0(va0715, iv31, smr);

        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0800 = __dtu_m_vmm_mode11_f_vs0(va0800, iv0, smr);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0801 = __dtu_m_vmm_mode11_f_vs0(va0801, iv1, smr);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0802 = __dtu_m_vmm_mode11_f_vs0(va0802, iv2, smr);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0803 = __dtu_m_vmm_mode11_f_vs0(va0803, iv3, smr);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0804 = __dtu_m_vmm_mode11_f_vs0(va0804, iv4, smr);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0805 = __dtu_m_vmm_mode11_f_vs0(va0805, iv5, smr);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0806 = __dtu_m_vmm_mode11_f_vs0(va0806, iv6, smr);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset3);
        va0807 = __dtu_m_vmm_mode11_f_vs0(va0807, iv7, smr);

        va0808 = __dtu_m_vmm_mode11_f_vs0(va0808, iv8, smr);
        va0809 = __dtu_m_vmm_mode11_f_vs0(va0809, iv9, smr);
        va0810 = __dtu_m_vmm_mode11_f_vs0(va0810, iv10, smr);
        va0811 = __dtu_m_vmm_mode11_f_vs0(va0811, iv11, smr);
        va0812 = __dtu_m_vmm_mode11_f_vs0(va0812, iv12, smr);
        va0813 = __dtu_m_vmm_mode11_f_vs0(va0813, iv13, smr);
        va0814 = __dtu_m_vmm_mode11_f_vs0(va0814, iv14, smr);
        va0815 = __dtu_m_vmm_mode11_f_vs0(va0815, iv15, smr);

        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);
      }  // end ci
      iv0 = __dtu_s_tivld_itar(input_addr, input_offset4);
    }  // end s
    iv0 = __dtu_s_tivld_itar(input_addr, input_offset5);
  }  // end r
  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(va0000, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0100, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0200, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0300, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0400, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0500, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0600, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0700, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0800, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0001, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0101, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0201, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0301, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0401, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0501, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0601, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0701, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0801, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0002, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0102, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0202, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0302, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0402, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0502, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0602, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0702, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0802, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0003, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0103, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0203, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0303, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0403, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0503, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0603, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0703, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0803, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0004, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0104, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0204, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0304, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0404, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0504, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0604, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0704, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0804, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0005, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0105, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0205, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0305, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0405, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0505, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0605, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0705, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0805, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0006, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0106, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0206, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0306, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0406, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0506, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0606, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0706, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0806, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0007, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0107, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0207, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0307, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0407, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0507, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0607, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0707, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0807, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0008, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0108, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0208, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0308, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0408, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0508, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0608, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0708, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0808, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0009, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0109, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0209, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0309, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0409, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0509, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0609, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0709, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0809, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0010, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0110, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0210, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0310, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0410, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0510, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0610, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0710, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0810, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0011, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0111, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0211, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0311, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0411, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0511, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0611, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0711, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0811, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0012, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0112, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0212, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0312, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0412, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0512, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0612, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0712, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0812, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0013, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0113, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0213, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0313, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0413, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0513, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0613, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0713, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0813, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0014, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0114, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0214, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0314, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0414, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0514, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0614, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0714, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0814, output_addr, output_offset);

    __dtu_l_tvsta_w_q(va0015, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0115, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0215, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0315, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0415, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0515, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0615, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0715, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va0815, output_addr, output_offset);
  }

  __dtu_c_movsr2naccovr(0);
}

extern "C" void conv3d_kerneln32ho1wo1(memref* lhs_param, memref* rhs_param,
                                       memref* out_param, int N, int Hi, int
                                       Wi,
                                       int Ci, int R, int S, int Co, int Ho,
                                       int Wo, int stride_h, int stride_w,
                                       int window_dilation_h,
                                       int window_dilation_w, int ld_flag,
                                       int st_flag) {
  int BPE = 4;
  int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
  int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
  int out_addr = reinterpret_cast<int>(out_param->addr);
  smr_t smr;
  v16f32 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10, iv11, iv12,
      iv13, iv14, iv15, iv16, iv17, iv18, iv19, iv20, iv21, iv22, iv23, iv24,
      iv25, iv26, iv27, iv28, iv29, iv30, iv31;
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  va16f32x4 va0, va1, va2, va3, va4, va5, va6, va7, va8, va9, va10, va11,
  va12,
      va13, va14, va15, va16, va17, va18, va19, va20, va21, va22, va23, va24,
      va25, va26, va27, va28, va29, va30, va31;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  // set up tar for input feature
  int addr = reinterpret_cast<int>(lhs_addr >> 6);
  addr = addr << 16 | addr;
  tar_t input_addr = __dtu_c_movsr2targ(addr);
  int in_off0 = ((Hi * Wi * Ci * BPE) >> 6) & 0xffff;
  ;
  int ci_offset = (in_off0 << 16) | in_off0;
  tar_t input_offset0 = __dtu_c_movsr2tari(ci_offset, input_addr);
  int in_off1 = ((16 * BPE - Hi * Wi * Ci * (N - 1) * BPE) >> 6) & 0xffff;

  ci_offset = (in_off1 << 16) | in_off1;
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
  addr = (addr + 1) << 16 | addr;  // thread 1 addr | thread 0 addr
  tar_t weight_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  int co_offset = (32 * BPE) >> 6;
  tar_t weight_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

  // set up tar for output
  addr = reinterpret_cast<int>(out_addr >> 6);
  addr = (addr + 1) << 16 | addr;
  tar_t output_addr = __dtu_c_movsr2targ(addr);
  // Co * 2 >> 6
  // int co_offset = (32 * BPE) >> 6;
  tar_t output_offset =
      __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
  tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

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
  vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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
        iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
        iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);

        // load weight
        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va0 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0, iv0, smr, vr0, 0);
        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va1 = __dtu_m_vmm_mode11_f_vs0_ld_row(va1, iv1, smr, vr1, 1);
        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va2 = __dtu_m_vmm_mode11_f_vs0_ld_row(va2, iv2, smr, vr2, 2);
        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va3 = __dtu_m_vmm_mode11_f_vs0_ld_row(va3, iv3, smr, vr3, 3);
        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va4 = __dtu_m_vmm_mode11_f_vs0_ld_row(va4, iv4, smr, vr4, 4);
        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va5 = __dtu_m_vmm_mode11_f_vs0_ld_row(va5, iv5, smr, vr5, 5);
        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va6 = __dtu_m_vmm_mode11_f_vs0_ld_row(va6, iv6, smr, vr6, 6);
        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va7 = __dtu_m_vmm_mode11_f_vs0_ld_row(va7, iv7, smr, vr7, 7);
        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va8 = __dtu_m_vmm_mode11_f_vs0_ld_row(va8, iv8, smr, vr8, 8);
        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va9 = __dtu_m_vmm_mode11_f_vs0_ld_row(va9, iv9, smr, vr9, 9);
        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va10 = __dtu_m_vmm_mode11_f_vs0_ld_row(va10, iv10, smr, vr10, 10);
        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va11 = __dtu_m_vmm_mode11_f_vs0_ld_row(va11, iv11, smr, vr11, 11);
        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va12 = __dtu_m_vmm_mode11_f_vs0_ld_row(va12, iv12, smr, vr12, 12);
        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va13 = __dtu_m_vmm_mode11_f_vs0_ld_row(va13, iv13, smr, vr13, 13);
        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va14 = __dtu_m_vmm_mode11_f_vs0_ld_row(va14, iv14, smr, vr14, 14);
        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va15 = __dtu_m_vmm_mode11_f_vs0_ld_row(va15, iv15, smr, vr15, 15);
        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va16 = __dtu_m_vmm_mode11_f_vs0_ld_row(va16, iv16, smr, vr16, 16);
        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va17 = __dtu_m_vmm_mode11_f_vs0_ld_row(va17, iv17, smr, vr17, 17);
        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va18 = __dtu_m_vmm_mode11_f_vs0_ld_row(va18, iv18, smr, vr18, 18);
        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va19 = __dtu_m_vmm_mode11_f_vs0_ld_row(va19, iv19, smr, vr19, 19);
        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va20 = __dtu_m_vmm_mode11_f_vs0_ld_row(va20, iv20, smr, vr20, 20);
        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va21 = __dtu_m_vmm_mode11_f_vs0_ld_row(va21, iv21, smr, vr21, 21);
        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
        va22 = __dtu_m_vmm_mode11_f_vs0_ld_row(va22, iv22, smr, vr22, 22);
        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
        va23 = __dtu_m_vmm_mode11_f_vs0_ld_row(va23, iv23, smr, vr23, 23);
        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

        va24 = __dtu_m_vmm_mode11_f_vs0_ld_row(va24, iv24, smr, vr24, 24);
        va25 = __dtu_m_vmm_mode11_f_vs0_ld_row(va25, iv25, smr, vr25, 25);
        va26 = __dtu_m_vmm_mode11_f_vs0_ld_row(va26, iv26, smr, vr26, 26);
        va27 = __dtu_m_vmm_mode11_f_vs0_ld_row(va27, iv27, smr, vr27, 27);
        va28 = __dtu_m_vmm_mode11_f_vs0_ld_row(va28, iv28, smr, vr28, 28);
        va29 = __dtu_m_vmm_mode11_f_vs0_ld_row(va29, iv29, smr, vr29, 29);
        va30 = __dtu_m_vmm_mode11_f_vs0_ld_row(va30, iv30, smr, vr30, 30);
        va31 = __dtu_m_vmm_mode11_f_vs0_ld_row(va31, iv31, smr, vr31, 31);

        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);

        vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr0, 32);

        vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr1, 33);

        vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr2, 34);

        vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr3, 35);

        vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr4, 36);

        vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr5, 37);

        vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr6, 38);

        vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr7, 39);

        vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr8, 40);

        vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr9, 41);

        vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr10, 42);

        vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr11, 43);

        vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr12, 44);

        vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr13, 45);

        vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr14, 46);

        vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr15, 47);

        vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr16, 48);

        vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr17, 49);

        vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr18, 50);

        vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr19, 51);

        vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr20, 52);

        vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr21, 53);

        vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr22, 54);

        vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr23, 55);

        vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr24, 56);

        vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr25, 57);

        vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr26, 58);

        vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr27, 59);

        vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr28, 60);

        vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr29, 61);

        vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr30, 62);

        vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?
        smr = __dtu_m_ldsmr_mode11_f_row(smr, vr31, 63);
      }  // end ci
      iv0 = __dtu_s_tivld_itar(input_addr, input_offset2);
    }  // end s
    iv0 = __dtu_s_tivld_itar(input_addr, input_offset3);
  }  // end r
  if (st_flag == 1) {
    __dtu_l_tvsta_w_q(va0, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va1, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va2, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va3, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va4, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va5, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va6, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va7, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va8, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va9, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va10, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va11, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va12, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va13, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va14, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va15, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va16, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va17, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va18, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va19, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va20, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va21, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va22, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va23, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va24, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va25, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va26, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va27, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va28, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va29, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va30, output_addr, output_offset);
    __dtu_l_tvsta_w_q(va31, output_addr, output_offset);
  }
  __dtu_c_movsr2naccovr(0);
}

// extern "C" void conv3d_kerneln32ho2wo2(memref* lhs_param, memref* rhs_param,
//                                        memref* out_param, int N, int Hi, int
//                                        Wi,
//                                        int Ci, int R, int S, int Co, int Ho,
//                                        int Wo, int stride_h, int stride_w,
//                                        int window_dilation_h,
//                                        int window_dilation_w, int ld_flag,
//                                        int st_flag) {
//   int BPE = 4;
//   int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
//   int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
//   int out_addr = reinterpret_cast<int>(out_param->addr);
//   smr_t smr;
//   v16f32 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10, iv11, iv12,
//       iv13, iv14, iv15, iv16, iv17, iv18, iv19, iv20, iv21, iv22, iv23, iv24,
//       iv25, iv26, iv27, iv28, iv29, iv30, iv31;
//   v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
//       vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
//       vr25, vr26, vr27, vr28, vr29, vr30, vr31;
//   va16f32x4 va0000, va0001, va0002, va0003, va0004, va0005, va0006, va0007,
//       va0008, va0009, va0010, va0011, va0012, va0013, va0014, va0015, va0016,
//       va0017, va0018, va0019, va0020, va0021, va0022, va0023, va0024, va0025,
//       va0026, va0027, va0028, va0029, va0030, va0031, va0100, va0101, va0102,
//       va0103, va0104, va0105, va0106, va0107, va0108, va0109, va0110, va0111,
//       va0112, va0113, va0114, va0115, va0116, va0117, va0118, va0119, va0120,
//       va0121, va0122, va0123, va0124, va0125, va0126, va0127, va0128, va0129,
//       va0130, va0131, va0200, va0201, va0202, va0203, va0204, va0205, va0206,
//       va0207, va0208, va0209, va0210, va0211, va0212, va0213, va0214, va0215,
//       va0216, va0217, va0218, va0219, va0220, va0221, va0222, va0223, va0224,
//       va0225, va0226, va0227, va0228, va0229, va0230, va0231, va0300, va0301,
//       va0302, va0303, va0304, va0305, va0306, va0307, va0308, va0309, va0310,
//       va0311, va0312, va0313, va0314, va0315, va0316, va0317, va0318, va0319,
//       va0320, va0321, va0322, va0323, va0324, va0325, va0326, va0327, va0328,
//       va0329, va0330, va0331;

//   __dtu_c_movsr2vab_lv_s(0);
//   __dtu_c_movsr2vab_m_s1(0);
//   __dtu_c_movsr2vab_m_d(0);

//   // set up tar for input feature
//   int addr = reinterpret_cast<int>(lhs_addr >> 6);
//   addr = addr << 16 | addr;
//   tar_t input_addr = __dtu_c_movsr2targ(addr);
//   // for next output on n
//   int in_off0 = ((Hi * Wi * Ci * BPE) >> 6) & 0xffff;
//   int ci_offset = (in_off0 << 16) | in_off0;
//   tar_t input_offset0 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // for next output on wo
//   int in_off1 =
//       ((Ci * stride_w * BPE - Hi * Wi * Ci * (N - 1) * BPE) >> 6) & 0xffff;

//   ci_offset = (in_off1 << 16) | in_off1;
//   tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // for next output on ho
//   int in_off2 = (((Ci * Wi * stride_h - Ci * stride_w * (Wo - 1) -
//                    Hi * Wi * Ci * (N - 1)) *
//                   BPE) >>
//                  6) &
//                 0xffff;
//   ci_offset = (in_off2 << 16) | in_off2;
//   tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   int in_off3 = (((16 - Ci * Wi * stride_h * (Ho - 1) -
//                    Ci * stride_w * (Wo - 1) - Hi * Wi * Ci * (N - 1)) *
//                   BPE) >>
//                  6) &
//                 0xffff;
//   ci_offset = (in_off3 << 16) | in_off3;
//   tar_t input_offset3 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   int in_off4 = (((Ci * window_dilation_w - Ci) * BPE) >> 6) & 0xffff;
//   ci_offset = (in_off4 << 16) | in_off4;
//   tar_t input_offset4 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   int in_off5 =
//       (((Ci * Wi * window_dilation_h - Ci * S * window_dilation_w) * BPE) >>
//        6) &
//       0xffff;
//   ci_offset = (in_off5 << 16) | in_off5;
//   tar_t input_offset5 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // set up tar for weight
//   addr = reinterpret_cast<int>(rhs_addr >> 6);
//   addr = (addr + 1) << 16 | addr;  // thread 1 addr | thread 0 addr
//   tar_t weight_addr = __dtu_c_movsr2targ(addr);
//   // Co * 2 >> 6
//   int co_offset = (32 * BPE) >> 6;
//   tar_t weight_offset =
//       __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

//   // set up tar for output
//   addr = reinterpret_cast<int>(out_addr >> 6);
//   addr = (addr + 1) << 16 | addr;
//   tar_t output_addr = __dtu_c_movsr2targ(addr);
//   // Co * 2 >> 6
//   // int co_offset = (32 * BPE) >> 6;
//   tar_t output_offset =
//       __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
//   tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

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
//   vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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
//         // load input[n][ci][0][0]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);

//         // load weight
//         vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0000 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0000, iv0, smr, vr0, 0);
//         vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0001 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0001, iv1, smr, vr1, 1);
//         vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0002 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0002, iv2, smr, vr2, 2);
//         vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0003 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0003, iv3, smr, vr3, 3);
//         vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0004 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0004, iv4, smr, vr4, 4);
//         vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0005 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0005, iv5, smr, vr5, 5);
//         vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0006 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0006, iv6, smr, vr6, 6);
//         vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0007 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0007, iv7, smr, vr7, 7);
//         vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0008 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0008, iv8, smr, vr8, 8);
//         vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0009 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0009, iv9, smr, vr9, 9);
//         vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0010 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0010, iv10, smr, vr10,
//         10);
//         vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0011 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0011, iv11, smr, vr11,
//         11);
//         vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0012 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0012, iv12, smr, vr12,
//         12);
//         vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0013 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0013, iv13, smr, vr13,
//         13);
//         vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0014 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0014, iv14, smr, vr14,
//         14);
//         vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0015 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0015, iv15, smr, vr15,
//         15);
//         vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0016 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0016, iv16, smr, vr16,
//         16);
//         vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0017 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0017, iv17, smr, vr17,
//         17);
//         vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0018 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0018, iv18, smr, vr18,
//         18);
//         vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0019 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0019, iv19, smr, vr19,
//         19);
//         vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0020 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0020, iv20, smr, vr20,
//         20);
//         vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0021 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0021, iv21, smr, vr21,
//         21);
//         vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0022 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0022, iv22, smr, vr22,
//         22);
//         vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         va0023 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0023, iv23, smr, vr23,
//         23);
//         vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         // load weight

//         // load input[n][ci][0][1]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0024 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0024, iv24, smr, vr24,
//         24);
//         vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0025 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0025, iv25, smr, vr25,
//         25);
//         vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0026 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0026, iv26, smr, vr26,
//         26);
//         vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0027 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0027, iv27, smr, vr27,
//         27);
//         vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0028 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0028, iv28, smr, vr28,
//         28);
//         vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0029 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0029, iv29, smr, vr29,
//         29);
//         vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0030 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0030, iv30, smr, vr30,
//         30);
//         vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0031 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0031, iv31, smr, vr31,
//         31);
//         vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0100 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0100, iv0, smr, vr0, 32);
//         vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0101 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0101, iv1, smr, vr1, 33);
//         vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0102 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0102, iv2, smr, vr2, 34);
//         vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0103 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0103, iv3, smr, vr3, 35);
//         vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0104 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0104, iv4, smr, vr4, 36);
//         vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0105 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0105, iv5, smr, vr5, 37);
//         vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0106 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0106, iv6, smr, vr6, 38);
//         vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0107 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0107, iv7, smr, vr7, 39);
//         vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0108 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0108, iv8, smr, vr8, 40);
//         vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0109 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0109, iv9, smr, vr9, 41);
//         vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0110 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0110, iv10, smr, vr10,
//         42);
//         vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0111 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0111, iv11, smr, vr11,
//         43);
//         vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0112 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0112, iv12, smr, vr12,
//         44);
//         vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0113 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0113, iv13, smr, vr13,
//         45);
//         vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0114 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0114, iv14, smr, vr14,
//         46);
//         vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0115 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0115, iv15, smr, vr15,
//         47);
//         vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0116 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0116, iv16, smr, vr16,
//         48);
//         vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0117 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0117, iv17, smr, vr17,
//         49);
//         vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0118 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0118, iv18, smr, vr18,
//         50);
//         vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0119 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0119, iv19, smr, vr19,
//         51);
//         vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0120 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0120, iv20, smr, vr20,
//         52);
//         vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0121 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0121, iv21, smr, vr21,
//         53);
//         vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0122 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0122, iv22, smr, vr22,
//         54);
//         vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset2);
//         va0123 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0123, iv23, smr, vr23,
//         55);
//         vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         // load input[n][ci][1][0]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0124 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0124, iv24, smr, vr24,
//         56);

//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0125 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0125, iv25, smr, vr25,
//         57);

//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0126 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0126, iv26, smr, vr26,
//         58);

//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0127 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0127, iv27, smr, vr27,
//         59);

//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0128 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0128, iv28, smr, vr28,
//         60);

//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0129 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0129, iv29, smr, vr29,
//         61);

//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0130 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0130, iv30, smr, vr30,
//         62);

//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0131 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0131, iv31, smr, vr31,
//         63);

//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0200 = __dtu_m_vmm_mode11_f_vs0(va0200, iv0, smr);

//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0201 = __dtu_m_vmm_mode11_f_vs0(va0201, iv1, smr);

//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0202 = __dtu_m_vmm_mode11_f_vs0(va0202, iv2, smr);

//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0203 = __dtu_m_vmm_mode11_f_vs0(va0203, iv3, smr);

//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0204 = __dtu_m_vmm_mode11_f_vs0(va0204, iv4, smr);

//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0205 = __dtu_m_vmm_mode11_f_vs0(va0205, iv5, smr);

//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0206 = __dtu_m_vmm_mode11_f_vs0(va0206, iv6, smr);

//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0207 = __dtu_m_vmm_mode11_f_vs0(va0207, iv7, smr);

//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0208 = __dtu_m_vmm_mode11_f_vs0(va0208, iv8, smr);

//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0209 = __dtu_m_vmm_mode11_f_vs0(va0209, iv9, smr);

//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0210 = __dtu_m_vmm_mode11_f_vs0(va0210, iv10, smr);

//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0211 = __dtu_m_vmm_mode11_f_vs0(va0211, iv11, smr);

//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0212 = __dtu_m_vmm_mode11_f_vs0(va0212, iv12, smr);

//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0213 = __dtu_m_vmm_mode11_f_vs0(va0213, iv13, smr);

//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0214 = __dtu_m_vmm_mode11_f_vs0(va0214, iv14, smr);

//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0215 = __dtu_m_vmm_mode11_f_vs0(va0215, iv15, smr);

//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0216 = __dtu_m_vmm_mode11_f_vs0(va0216, iv16, smr);

//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0217 = __dtu_m_vmm_mode11_f_vs0(va0217, iv17, smr);

//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0218 = __dtu_m_vmm_mode11_f_vs0(va0218, iv18, smr);

//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0219 = __dtu_m_vmm_mode11_f_vs0(va0219, iv19, smr);

//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0220 = __dtu_m_vmm_mode11_f_vs0(va0220, iv20, smr);

//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0221 = __dtu_m_vmm_mode11_f_vs0(va0221, iv21, smr);

//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0222 = __dtu_m_vmm_mode11_f_vs0(va0222, iv22, smr);

//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         va0223 = __dtu_m_vmm_mode11_f_vs0(va0223, iv23, smr);

//         // load input[n][ci][1][1]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0224 = __dtu_m_vmm_mode11_f_vs0(va0224, iv24, smr);

//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0225 = __dtu_m_vmm_mode11_f_vs0(va0225, iv25, smr);

//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0226 = __dtu_m_vmm_mode11_f_vs0(va0226, iv26, smr);

//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0227 = __dtu_m_vmm_mode11_f_vs0(va0227, iv27, smr);

//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0228 = __dtu_m_vmm_mode11_f_vs0(va0228, iv28, smr);

//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0229 = __dtu_m_vmm_mode11_f_vs0(va0229, iv29, smr);

//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0230 = __dtu_m_vmm_mode11_f_vs0(va0230, iv30, smr);

//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0231 = __dtu_m_vmm_mode11_f_vs0(va0231, iv31, smr);

//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0300 = __dtu_m_vmm_mode11_f_vs0(va0300, iv0, smr);

//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0301 = __dtu_m_vmm_mode11_f_vs0(va0301, iv1, smr);

//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0302 = __dtu_m_vmm_mode11_f_vs0(va0302, iv2, smr);

//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0303 = __dtu_m_vmm_mode11_f_vs0(va0303, iv3, smr);

//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0304 = __dtu_m_vmm_mode11_f_vs0(va0304, iv4, smr);

//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0305 = __dtu_m_vmm_mode11_f_vs0(va0305, iv5, smr);

//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0306 = __dtu_m_vmm_mode11_f_vs0(va0306, iv6, smr);

//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0307 = __dtu_m_vmm_mode11_f_vs0(va0307, iv7, smr);

//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0308 = __dtu_m_vmm_mode11_f_vs0(va0308, iv8, smr);

//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0309 = __dtu_m_vmm_mode11_f_vs0(va0309, iv9, smr);

//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0310 = __dtu_m_vmm_mode11_f_vs0(va0310, iv10, smr);

//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0311 = __dtu_m_vmm_mode11_f_vs0(va0311, iv11, smr);

//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0312 = __dtu_m_vmm_mode11_f_vs0(va0312, iv12, smr);

//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0313 = __dtu_m_vmm_mode11_f_vs0(va0313, iv13, smr);

//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0314 = __dtu_m_vmm_mode11_f_vs0(va0314, iv14, smr);

//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0315 = __dtu_m_vmm_mode11_f_vs0(va0315, iv15, smr);

//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0316 = __dtu_m_vmm_mode11_f_vs0(va0316, iv16, smr);

//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0317 = __dtu_m_vmm_mode11_f_vs0(va0317, iv17, smr);

//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0318 = __dtu_m_vmm_mode11_f_vs0(va0318, iv18, smr);

//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0319 = __dtu_m_vmm_mode11_f_vs0(va0319, iv19, smr);

//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0320 = __dtu_m_vmm_mode11_f_vs0(va0320, iv20, smr);

//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0321 = __dtu_m_vmm_mode11_f_vs0(va0321, iv21, smr);

//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         va0322 = __dtu_m_vmm_mode11_f_vs0(va0322, iv22, smr);

//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset3);
//         va0323 = __dtu_m_vmm_mode11_f_vs0(va0323, iv23, smr);

//         va0324 = __dtu_m_vmm_mode11_f_vs0(va0324, iv24, smr);
//         va0325 = __dtu_m_vmm_mode11_f_vs0(va0325, iv25, smr);
//         va0326 = __dtu_m_vmm_mode11_f_vs0(va0326, iv26, smr);
//         va0327 = __dtu_m_vmm_mode11_f_vs0(va0327, iv27, smr);
//         va0328 = __dtu_m_vmm_mode11_f_vs0(va0328, iv28, smr);
//         va0329 = __dtu_m_vmm_mode11_f_vs0(va0329, iv29, smr);
//         va0330 = __dtu_m_vmm_mode11_f_vs0(va0330, iv30, smr);
//         va0331 = __dtu_m_vmm_mode11_f_vs0(va0331, iv31, smr);

//         smr = __dtu_v_swapsmr(smr);
//         __dtu_c_movsr2naccovr(0x1);
//       }  // end ci
//       iv0 = __dtu_s_tivld_itar(input_addr, input_offset4);
//     }  // end s
//     iv0 = __dtu_s_tivld_itar(input_addr, input_offset5);
//   }  // end r
//   if (st_flag == 1) {
//     __dtu_l_tvsta_w_q(va0000, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0100, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0200, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0300, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0001, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0101, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0201, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0301, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0002, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0102, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0202, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0302, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0003, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0103, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0203, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0303, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0004, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0104, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0204, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0304, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0005, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0105, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0205, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0305, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0006, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0106, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0206, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0306, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0007, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0107, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0207, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0307, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0008, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0108, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0208, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0308, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0009, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0109, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0209, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0309, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0010, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0110, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0210, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0310, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0011, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0111, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0211, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0311, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0012, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0112, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0212, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0312, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0013, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0113, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0213, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0313, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0014, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0114, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0214, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0314, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0015, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0115, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0215, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0315, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0016, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0116, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0216, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0316, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0017, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0117, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0217, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0317, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0018, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0118, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0218, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0318, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0019, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0119, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0219, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0319, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0020, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0120, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0220, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0320, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0021, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0121, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0221, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0321, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0022, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0122, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0222, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0322, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0023, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0123, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0223, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0323, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0024, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0124, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0224, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0324, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0025, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0125, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0225, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0325, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0026, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0126, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0226, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0326, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0027, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0127, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0227, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0327, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0028, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0128, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0228, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0328, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0029, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0129, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0229, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0329, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0030, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0130, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0230, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0330, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0031, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0131, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0231, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0331, output_addr, output_offset);
//   }

//   __dtu_c_movsr2naccovr(0);
// }

// extern "C" void conv3d_kerneln32ho3wo3(memref* lhs_param, memref* rhs_param,
//                                        memref* out_param, int N, int Hi, int
//                                        Wi,
//                                        int Ci, int R, int S, int Co, int Ho,
//                                        int Wo, int stride_h, int stride_w,
//                                        int window_dilation_h,
//                                        int window_dilation_w, int ld_flag,
//                                        int st_flag) {
//   int BPE = 4;
//   int lhs_addr = reinterpret_cast<int>(lhs_param->addr);
//   int rhs_addr = reinterpret_cast<int>(rhs_param->addr);
//   int out_addr = reinterpret_cast<int>(out_param->addr);
//   smr_t smr;
//   v16f32 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10, iv11, iv12,
//       iv13, iv14, iv15, iv16, iv17, iv18, iv19, iv20, iv21, iv22, iv23, iv24,
//       iv25, iv26, iv27, iv28, iv29, iv30, iv31;
//   v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
//       vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
//       vr25, vr26, vr27, vr28, vr29, vr30, vr31;
//   va16f32x4 va0000, va0001, va0002, va0003, va0004, va0005, va0006, va0007,
//       va0008, va0009, va0010, va0011, va0012, va0013, va0014, va0015, va0016,
//       va0017, va0018, va0019, va0020, va0021, va0022, va0023, va0024, va0025,
//       va0026, va0027, va0028, va0029, va0030, va0031, va0100, va0101, va0102,
//       va0103, va0104, va0105, va0106, va0107, va0108, va0109, va0110, va0111,
//       va0112, va0113, va0114, va0115, va0116, va0117, va0118, va0119, va0120,
//       va0121, va0122, va0123, va0124, va0125, va0126, va0127, va0128, va0129,
//       va0130, va0131, va0200, va0201, va0202, va0203, va0204, va0205, va0206,
//       va0207, va0208, va0209, va0210, va0211, va0212, va0213, va0214, va0215,
//       va0216, va0217, va0218, va0219, va0220, va0221, va0222, va0223, va0224,
//       va0225, va0226, va0227, va0228, va0229, va0230, va0231, va0300, va0301,
//       va0302, va0303, va0304, va0305, va0306, va0307, va0308, va0309, va0310,
//       va0311, va0312, va0313, va0314, va0315, va0316, va0317, va0318, va0319,
//       va0320, va0321, va0322, va0323, va0324, va0325, va0326, va0327, va0328,
//       va0329, va0330, va0331, va0400, va0401, va0402, va0403, va0404, va0405,
//       va0406, va0407, va0408, va0409, va0410, va0411, va0412, va0413, va0414,
//       va0415, va0416, va0417, va0418, va0419, va0420, va0421, va0422, va0423,
//       va0424, va0425, va0426, va0427, va0428, va0429, va0430, va0431, va0500,
//       va0501, va0502, va0503, va0504, va0505, va0506, va0507, va0508, va0509,
//       va0510, va0511, va0512, va0513, va0514, va0515, va0516, va0517, va0518,
//       va0519, va0520, va0521, va0522, va0523, va0524, va0525, va0526, va0527,
//       va0528, va0529, va0530, va0531, va0600, va0601, va0602, va0603, va0604,
//       va0605, va0606, va0607, va0608, va0609, va0610, va0611, va0612, va0613,
//       va0614, va0615, va0616, va0617, va0618, va0619, va0620, va0621, va0622,
//       va0623, va0624, va0625, va0626, va0627, va0628, va0629, va0630, va0631,
//       va0700, va0701, va0702, va0703, va0704, va0705, va0706, va0707, va0708,
//       va0709, va0710, va0711, va0712, va0713, va0714, va0715, va0716, va0717,
//       va0718, va0719, va0720, va0721, va0722, va0723, va0724, va0725, va0726,
//       va0727, va0728, va0729, va0730, va0731, va0800, va0801, va0802, va0803,
//       va0804, va0805, va0806, va0807, va0808, va0809, va0810, va0811, va0812,
//       va0813, va0814, va0815, va0816, va0817, va0818, va0819, va0820, va0821,
//       va0822, va0823, va0824, va0825, va0826, va0827, va0828, va0829, va0830,
//       va0831;

//   __dtu_c_movsr2vab_lv_s(0);
//   __dtu_c_movsr2vab_m_s1(0);
//   __dtu_c_movsr2vab_m_d(0);

//   // set up tar for input feature
//   int addr = reinterpret_cast<int>(lhs_addr >> 6);
//   addr = addr << 16 | addr;
//   tar_t input_addr = __dtu_c_movsr2targ(addr);
//   // for next output on n
//   int in_off0 = ((Hi * Wi * Ci * BPE) >> 6) & 0xffff;
//   int ci_offset = (in_off0 << 16) | in_off0;
//   tar_t input_offset0 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // for next output on wo
//   int in_off1 =
//       ((Ci * stride_w * BPE - Hi * Wi * Ci * (N - 1) * BPE) >> 6) & 0xffff;

//   ci_offset = (in_off1 << 16) | in_off1;
//   tar_t input_offset1 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // for next output on ho
//   int in_off2 = (((Ci * Wi * stride_h - Ci * stride_w * (Wo - 1) -
//                    Hi * Wi * Ci * (N - 1)) *
//                   BPE) >>
//                  6) &
//                 0xffff;
//   ci_offset = (in_off2 << 16) | in_off2;
//   tar_t input_offset2 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   int in_off3 = (((16 - Ci * Wi * stride_h * (Ho - 1) -
//                    Ci * stride_w * (Wo - 1) - Hi * Wi * Ci * (N - 1)) *
//                   BPE) >>
//                  6) &
//                 0xffff;
//   ci_offset = (in_off3 << 16) | in_off3;
//   tar_t input_offset3 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   int in_off4 = (((Ci * window_dilation_w - Ci) * BPE) >> 6) & 0xffff;
//   ci_offset = (in_off4 << 16) | in_off4;
//   tar_t input_offset4 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // Ci * Wi * window_dilation_h -((actual_s - 1) + 1 + (window_dilation_w -
//   // 1))
//   int in_off5 =
//       (((Ci * Wi * window_dilation_h - Ci * S * window_dilation_w) * BPE) >>
//        6) &
//       0xffff;
//   ci_offset = (in_off5 << 16) | in_off5;
//   tar_t input_offset5 = __dtu_c_movsr2tari(ci_offset, input_addr);

//   // set up tar for weight
//   addr = reinterpret_cast<int>(rhs_addr >> 6);
//   addr = (addr + 1) << 16 | addr;  // thread 1 addr | thread 0 addr
//   tar_t weight_addr = __dtu_c_movsr2targ(addr);
//   // Co * 2 >> 6
//   int co_offset = (32 * BPE) >> 6;
//   tar_t weight_offset =
//       __dtu_c_movsr2tari((co_offset << 16) | co_offset, weight_addr);

//   // set up tar for output
//   addr = reinterpret_cast<int>(out_addr >> 6);
//   addr = (addr + 1) << 16 | addr;
//   tar_t output_addr = __dtu_c_movsr2targ(addr);
//   // Co * 2 >> 6
//   // int co_offset = (32 * BPE) >> 6;
//   tar_t output_offset =
//       __dtu_c_movsr2tari((co_offset << 16) | co_offset, output_addr);
//   tar_t output_offset0 = __dtu_c_movsr2tari((0 << 16) | 0, output_addr);

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
//   vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);  //?

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
//         // load input[n][ci][0][0]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         // load weight
//         vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         va0000 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0000, iv0, smr, vr0, 0);
//         va0001 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0001, iv1, smr, vr1, 1);
//         va0002 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0002, iv2, smr, vr2, 2);
//         va0003 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0003, iv3, smr, vr3, 3);
//         va0004 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0004, iv4, smr, vr4, 4);
//         va0005 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0005, iv5, smr, vr5, 5);
//         va0006 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0006, iv6, smr, vr6, 6);
//         va0007 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0007, iv7, smr, vr7, 7);
//         va0008 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0008, iv8, smr, vr8, 8);
//         va0009 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0009, iv9, smr, vr9, 9);
//         va0010 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0010, iv10, smr, vr10,
//         10);
//         va0011 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0011, iv11, smr, vr11,
//         11);
//         va0012 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0012, iv12, smr, vr12,
//         12);
//         va0013 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0013, iv13, smr, vr13,
//         13);
//         va0014 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0014, iv14, smr, vr14,
//         14);
//         va0015 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0015, iv15, smr, vr15,
//         15);
//         va0016 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0016, iv16, smr, vr16,
//         16);
//         va0017 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0017, iv17, smr, vr17,
//         17);
//         va0018 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0018, iv18, smr, vr18,
//         18);
//         va0019 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0019, iv19, smr, vr19,
//         19);
//         va0020 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0020, iv20, smr, vr20,
//         20);
//         va0021 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0021, iv21, smr, vr21,
//         21);
//         va0022 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0022, iv22, smr, vr22,
//         22);
//         va0023 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0023, iv23, smr, vr23,
//         23);
//         va0024 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0024, iv24, smr, vr24,
//         24);
//         va0025 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0025, iv25, smr, vr25,
//         25);
//         va0026 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0026, iv26, smr, vr26,
//         26);
//         va0027 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0027, iv27, smr, vr27,
//         27);
//         va0028 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0028, iv28, smr, vr28,
//         28);
//         va0029 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0029, iv29, smr, vr29,
//         29);
//         va0030 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0030, iv30, smr, vr30,
//         30);
//         va0031 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0031, iv31, smr, vr31,
//         31);

//         // load weight
//         vr0 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr1 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr2 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr3 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr4 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr5 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr6 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr7 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr8 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr9 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr10 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr11 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr12 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr13 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr14 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr15 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr16 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr17 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr18 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr19 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr20 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr21 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr22 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr23 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr24 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr25 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr26 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr27 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr28 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr29 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr30 = __dtu_s_tvld_itar(weight_addr, weight_offset);
//         vr31 = __dtu_s_tvld_itar(weight_addr, weight_offset);

//         // load input[n][ci][0][1]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         va0100 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0100, iv0, smr, vr0, 32);
//         va0101 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0101, iv1, smr, vr1, 33);
//         va0102 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0102, iv2, smr, vr2, 34);
//         va0103 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0103, iv3, smr, vr3, 35);
//         va0104 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0104, iv4, smr, vr4, 36);
//         va0105 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0105, iv5, smr, vr5, 37);
//         va0106 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0106, iv6, smr, vr6, 38);
//         va0107 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0107, iv7, smr, vr7, 39);
//         va0108 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0108, iv8, smr, vr8, 40);
//         va0109 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0109, iv9, smr, vr9, 41);
//         va0110 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0110, iv10, smr, vr10,
//         42);
//         va0111 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0111, iv11, smr, vr11,
//         43);
//         va0112 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0112, iv12, smr, vr12,
//         44);
//         va0113 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0113, iv13, smr, vr13,
//         45);
//         va0114 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0114, iv14, smr, vr14,
//         46);
//         va0115 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0115, iv15, smr, vr15,
//         47);
//         va0116 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0116, iv16, smr, vr16,
//         48);
//         va0117 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0117, iv17, smr, vr17,
//         49);
//         va0118 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0118, iv18, smr, vr18,
//         50);
//         va0119 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0119, iv19, smr, vr19,
//         51);
//         va0120 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0120, iv20, smr, vr20,
//         52);
//         va0121 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0121, iv21, smr, vr21,
//         53);
//         va0122 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0122, iv22, smr, vr22,
//         54);
//         va0123 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0123, iv23, smr, vr23,
//         55);
//         va0124 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0124, iv24, smr, vr24,
//         56);
//         va0125 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0125, iv25, smr, vr25,
//         57);
//         va0126 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0126, iv26, smr, vr26,
//         58);
//         va0127 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0127, iv27, smr, vr27,
//         59);
//         va0128 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0128, iv28, smr, vr28,
//         60);
//         va0129 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0129, iv29, smr, vr29,
//         61);
//         va0130 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0130, iv30, smr, vr30,
//         62);
//         va0131 = __dtu_m_vmm_mode11_f_vs0_ld_row(va0131, iv31, smr, vr31,
//         63);

//         // load input[n][ci][0][2]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset2);
//         va0200 = __dtu_m_vmm_mode11_f_vs0(va0200, iv0, smr);
//         va0201 = __dtu_m_vmm_mode11_f_vs0(va0201, iv1, smr);
//         va0202 = __dtu_m_vmm_mode11_f_vs0(va0202, iv2, smr);
//         va0203 = __dtu_m_vmm_mode11_f_vs0(va0203, iv3, smr);
//         va0204 = __dtu_m_vmm_mode11_f_vs0(va0204, iv4, smr);
//         va0205 = __dtu_m_vmm_mode11_f_vs0(va0205, iv5, smr);
//         va0206 = __dtu_m_vmm_mode11_f_vs0(va0206, iv6, smr);
//         va0207 = __dtu_m_vmm_mode11_f_vs0(va0207, iv7, smr);
//         va0208 = __dtu_m_vmm_mode11_f_vs0(va0208, iv8, smr);
//         va0209 = __dtu_m_vmm_mode11_f_vs0(va0209, iv9, smr);
//         va0210 = __dtu_m_vmm_mode11_f_vs0(va0210, iv10, smr);
//         va0211 = __dtu_m_vmm_mode11_f_vs0(va0211, iv11, smr);
//         va0212 = __dtu_m_vmm_mode11_f_vs0(va0212, iv12, smr);
//         va0213 = __dtu_m_vmm_mode11_f_vs0(va0213, iv13, smr);
//         va0214 = __dtu_m_vmm_mode11_f_vs0(va0214, iv14, smr);
//         va0215 = __dtu_m_vmm_mode11_f_vs0(va0215, iv15, smr);
//         va0216 = __dtu_m_vmm_mode11_f_vs0(va0216, iv16, smr);
//         va0217 = __dtu_m_vmm_mode11_f_vs0(va0217, iv17, smr);
//         va0218 = __dtu_m_vmm_mode11_f_vs0(va0218, iv18, smr);
//         va0219 = __dtu_m_vmm_mode11_f_vs0(va0219, iv19, smr);
//         va0220 = __dtu_m_vmm_mode11_f_vs0(va0220, iv20, smr);
//         va0221 = __dtu_m_vmm_mode11_f_vs0(va0221, iv21, smr);
//         va0222 = __dtu_m_vmm_mode11_f_vs0(va0222, iv22, smr);
//         va0223 = __dtu_m_vmm_mode11_f_vs0(va0223, iv23, smr);
//         va0224 = __dtu_m_vmm_mode11_f_vs0(va0224, iv24, smr);
//         va0225 = __dtu_m_vmm_mode11_f_vs0(va0225, iv25, smr);
//         va0226 = __dtu_m_vmm_mode11_f_vs0(va0226, iv26, smr);
//         va0227 = __dtu_m_vmm_mode11_f_vs0(va0227, iv27, smr);
//         va0228 = __dtu_m_vmm_mode11_f_vs0(va0228, iv28, smr);
//         va0229 = __dtu_m_vmm_mode11_f_vs0(va0229, iv29, smr);
//         va0230 = __dtu_m_vmm_mode11_f_vs0(va0230, iv30, smr);
//         va0231 = __dtu_m_vmm_mode11_f_vs0(va0231, iv31, smr);

//         // load input[n][ci][1][0]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         va0300 = __dtu_m_vmm_mode11_f_vs0(va0300, iv0, smr);
//         va0301 = __dtu_m_vmm_mode11_f_vs0(va0301, iv1, smr);
//         va0302 = __dtu_m_vmm_mode11_f_vs0(va0302, iv2, smr);
//         va0303 = __dtu_m_vmm_mode11_f_vs0(va0303, iv3, smr);
//         va0304 = __dtu_m_vmm_mode11_f_vs0(va0304, iv4, smr);
//         va0305 = __dtu_m_vmm_mode11_f_vs0(va0305, iv5, smr);
//         va0306 = __dtu_m_vmm_mode11_f_vs0(va0306, iv6, smr);
//         va0307 = __dtu_m_vmm_mode11_f_vs0(va0307, iv7, smr);
//         va0308 = __dtu_m_vmm_mode11_f_vs0(va0308, iv8, smr);
//         va0309 = __dtu_m_vmm_mode11_f_vs0(va0309, iv9, smr);
//         va0310 = __dtu_m_vmm_mode11_f_vs0(va0310, iv10, smr);
//         va0311 = __dtu_m_vmm_mode11_f_vs0(va0311, iv11, smr);
//         va0312 = __dtu_m_vmm_mode11_f_vs0(va0312, iv12, smr);
//         va0313 = __dtu_m_vmm_mode11_f_vs0(va0313, iv13, smr);
//         va0314 = __dtu_m_vmm_mode11_f_vs0(va0314, iv14, smr);
//         va0315 = __dtu_m_vmm_mode11_f_vs0(va0315, iv15, smr);
//         va0316 = __dtu_m_vmm_mode11_f_vs0(va0316, iv16, smr);
//         va0317 = __dtu_m_vmm_mode11_f_vs0(va0317, iv17, smr);
//         va0318 = __dtu_m_vmm_mode11_f_vs0(va0318, iv18, smr);
//         va0319 = __dtu_m_vmm_mode11_f_vs0(va0319, iv19, smr);
//         va0320 = __dtu_m_vmm_mode11_f_vs0(va0320, iv20, smr);
//         va0321 = __dtu_m_vmm_mode11_f_vs0(va0321, iv21, smr);
//         va0322 = __dtu_m_vmm_mode11_f_vs0(va0322, iv22, smr);
//         va0323 = __dtu_m_vmm_mode11_f_vs0(va0323, iv23, smr);
//         va0324 = __dtu_m_vmm_mode11_f_vs0(va0324, iv24, smr);
//         va0325 = __dtu_m_vmm_mode11_f_vs0(va0325, iv25, smr);
//         va0326 = __dtu_m_vmm_mode11_f_vs0(va0326, iv26, smr);
//         va0327 = __dtu_m_vmm_mode11_f_vs0(va0327, iv27, smr);
//         va0328 = __dtu_m_vmm_mode11_f_vs0(va0328, iv28, smr);
//         va0329 = __dtu_m_vmm_mode11_f_vs0(va0329, iv29, smr);
//         va0330 = __dtu_m_vmm_mode11_f_vs0(va0330, iv30, smr);
//         va0331 = __dtu_m_vmm_mode11_f_vs0(va0331, iv31, smr);

//         // load input[n][ci][1][1]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         va0400 = __dtu_m_vmm_mode11_f_vs0(va0400, iv0, smr);
//         va0401 = __dtu_m_vmm_mode11_f_vs0(va0401, iv1, smr);
//         va0402 = __dtu_m_vmm_mode11_f_vs0(va0402, iv2, smr);
//         va0403 = __dtu_m_vmm_mode11_f_vs0(va0403, iv3, smr);
//         va0404 = __dtu_m_vmm_mode11_f_vs0(va0404, iv4, smr);
//         va0405 = __dtu_m_vmm_mode11_f_vs0(va0405, iv5, smr);
//         va0406 = __dtu_m_vmm_mode11_f_vs0(va0406, iv6, smr);
//         va0407 = __dtu_m_vmm_mode11_f_vs0(va0407, iv7, smr);
//         va0408 = __dtu_m_vmm_mode11_f_vs0(va0408, iv8, smr);
//         va0409 = __dtu_m_vmm_mode11_f_vs0(va0409, iv9, smr);
//         va0410 = __dtu_m_vmm_mode11_f_vs0(va0410, iv10, smr);
//         va0411 = __dtu_m_vmm_mode11_f_vs0(va0411, iv11, smr);
//         va0412 = __dtu_m_vmm_mode11_f_vs0(va0412, iv12, smr);
//         va0413 = __dtu_m_vmm_mode11_f_vs0(va0413, iv13, smr);
//         va0414 = __dtu_m_vmm_mode11_f_vs0(va0414, iv14, smr);
//         va0415 = __dtu_m_vmm_mode11_f_vs0(va0415, iv15, smr);
//         va0416 = __dtu_m_vmm_mode11_f_vs0(va0416, iv16, smr);
//         va0417 = __dtu_m_vmm_mode11_f_vs0(va0417, iv17, smr);
//         va0418 = __dtu_m_vmm_mode11_f_vs0(va0418, iv18, smr);
//         va0419 = __dtu_m_vmm_mode11_f_vs0(va0419, iv19, smr);
//         va0420 = __dtu_m_vmm_mode11_f_vs0(va0420, iv20, smr);
//         va0421 = __dtu_m_vmm_mode11_f_vs0(va0421, iv21, smr);
//         va0422 = __dtu_m_vmm_mode11_f_vs0(va0422, iv22, smr);
//         va0423 = __dtu_m_vmm_mode11_f_vs0(va0423, iv23, smr);
//         va0424 = __dtu_m_vmm_mode11_f_vs0(va0424, iv24, smr);
//         va0425 = __dtu_m_vmm_mode11_f_vs0(va0425, iv25, smr);
//         va0426 = __dtu_m_vmm_mode11_f_vs0(va0426, iv26, smr);
//         va0427 = __dtu_m_vmm_mode11_f_vs0(va0427, iv27, smr);
//         va0428 = __dtu_m_vmm_mode11_f_vs0(va0428, iv28, smr);
//         va0429 = __dtu_m_vmm_mode11_f_vs0(va0429, iv29, smr);
//         va0430 = __dtu_m_vmm_mode11_f_vs0(va0430, iv30, smr);
//         va0431 = __dtu_m_vmm_mode11_f_vs0(va0431, iv31, smr);

//         // load input[n][ci][1][2]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset2);
//         va0500 = __dtu_m_vmm_mode11_f_vs0(va0500, iv0, smr);
//         va0501 = __dtu_m_vmm_mode11_f_vs0(va0501, iv1, smr);
//         va0502 = __dtu_m_vmm_mode11_f_vs0(va0502, iv2, smr);
//         va0503 = __dtu_m_vmm_mode11_f_vs0(va0503, iv3, smr);
//         va0504 = __dtu_m_vmm_mode11_f_vs0(va0504, iv4, smr);
//         va0505 = __dtu_m_vmm_mode11_f_vs0(va0505, iv5, smr);
//         va0506 = __dtu_m_vmm_mode11_f_vs0(va0506, iv6, smr);
//         va0507 = __dtu_m_vmm_mode11_f_vs0(va0507, iv7, smr);
//         va0508 = __dtu_m_vmm_mode11_f_vs0(va0508, iv8, smr);
//         va0509 = __dtu_m_vmm_mode11_f_vs0(va0509, iv9, smr);
//         va0510 = __dtu_m_vmm_mode11_f_vs0(va0510, iv10, smr);
//         va0511 = __dtu_m_vmm_mode11_f_vs0(va0511, iv11, smr);
//         va0512 = __dtu_m_vmm_mode11_f_vs0(va0512, iv12, smr);
//         va0513 = __dtu_m_vmm_mode11_f_vs0(va0513, iv13, smr);
//         va0514 = __dtu_m_vmm_mode11_f_vs0(va0514, iv14, smr);
//         va0515 = __dtu_m_vmm_mode11_f_vs0(va0515, iv15, smr);
//         va0516 = __dtu_m_vmm_mode11_f_vs0(va0516, iv16, smr);
//         va0517 = __dtu_m_vmm_mode11_f_vs0(va0517, iv17, smr);
//         va0518 = __dtu_m_vmm_mode11_f_vs0(va0518, iv18, smr);
//         va0519 = __dtu_m_vmm_mode11_f_vs0(va0519, iv19, smr);
//         va0520 = __dtu_m_vmm_mode11_f_vs0(va0520, iv20, smr);
//         va0521 = __dtu_m_vmm_mode11_f_vs0(va0521, iv21, smr);
//         va0522 = __dtu_m_vmm_mode11_f_vs0(va0522, iv22, smr);
//         va0523 = __dtu_m_vmm_mode11_f_vs0(va0523, iv23, smr);
//         va0524 = __dtu_m_vmm_mode11_f_vs0(va0524, iv24, smr);
//         va0525 = __dtu_m_vmm_mode11_f_vs0(va0525, iv25, smr);
//         va0526 = __dtu_m_vmm_mode11_f_vs0(va0526, iv26, smr);
//         va0527 = __dtu_m_vmm_mode11_f_vs0(va0527, iv27, smr);
//         va0528 = __dtu_m_vmm_mode11_f_vs0(va0528, iv28, smr);
//         va0529 = __dtu_m_vmm_mode11_f_vs0(va0529, iv29, smr);
//         va0530 = __dtu_m_vmm_mode11_f_vs0(va0530, iv30, smr);
//         va0531 = __dtu_m_vmm_mode11_f_vs0(va0531, iv31, smr);

//         // load input[n][ci][2][0]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         va0600 = __dtu_m_vmm_mode11_f_vs0(va0600, iv0, smr);
//         va0601 = __dtu_m_vmm_mode11_f_vs0(va0601, iv1, smr);
//         va0602 = __dtu_m_vmm_mode11_f_vs0(va0602, iv2, smr);
//         va0603 = __dtu_m_vmm_mode11_f_vs0(va0603, iv3, smr);
//         va0604 = __dtu_m_vmm_mode11_f_vs0(va0604, iv4, smr);
//         va0605 = __dtu_m_vmm_mode11_f_vs0(va0605, iv5, smr);
//         va0606 = __dtu_m_vmm_mode11_f_vs0(va0606, iv6, smr);
//         va0607 = __dtu_m_vmm_mode11_f_vs0(va0607, iv7, smr);
//         va0608 = __dtu_m_vmm_mode11_f_vs0(va0608, iv8, smr);
//         va0609 = __dtu_m_vmm_mode11_f_vs0(va0609, iv9, smr);
//         va0610 = __dtu_m_vmm_mode11_f_vs0(va0610, iv10, smr);
//         va0611 = __dtu_m_vmm_mode11_f_vs0(va0611, iv11, smr);
//         va0612 = __dtu_m_vmm_mode11_f_vs0(va0612, iv12, smr);
//         va0613 = __dtu_m_vmm_mode11_f_vs0(va0613, iv13, smr);
//         va0614 = __dtu_m_vmm_mode11_f_vs0(va0614, iv14, smr);
//         va0615 = __dtu_m_vmm_mode11_f_vs0(va0615, iv15, smr);
//         va0616 = __dtu_m_vmm_mode11_f_vs0(va0616, iv16, smr);
//         va0617 = __dtu_m_vmm_mode11_f_vs0(va0617, iv17, smr);
//         va0618 = __dtu_m_vmm_mode11_f_vs0(va0618, iv18, smr);
//         va0619 = __dtu_m_vmm_mode11_f_vs0(va0619, iv19, smr);
//         va0620 = __dtu_m_vmm_mode11_f_vs0(va0620, iv20, smr);
//         va0621 = __dtu_m_vmm_mode11_f_vs0(va0621, iv21, smr);
//         va0622 = __dtu_m_vmm_mode11_f_vs0(va0622, iv22, smr);
//         va0623 = __dtu_m_vmm_mode11_f_vs0(va0623, iv23, smr);
//         va0624 = __dtu_m_vmm_mode11_f_vs0(va0624, iv24, smr);
//         va0625 = __dtu_m_vmm_mode11_f_vs0(va0625, iv25, smr);
//         va0626 = __dtu_m_vmm_mode11_f_vs0(va0626, iv26, smr);
//         va0627 = __dtu_m_vmm_mode11_f_vs0(va0627, iv27, smr);
//         va0628 = __dtu_m_vmm_mode11_f_vs0(va0628, iv28, smr);
//         va0629 = __dtu_m_vmm_mode11_f_vs0(va0629, iv29, smr);
//         va0630 = __dtu_m_vmm_mode11_f_vs0(va0630, iv30, smr);
//         va0631 = __dtu_m_vmm_mode11_f_vs0(va0631, iv31, smr);

//         // load input[n][ci][2][1]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset1);
//         va0700 = __dtu_m_vmm_mode11_f_vs0(va0700, iv0, smr);
//         va0701 = __dtu_m_vmm_mode11_f_vs0(va0701, iv1, smr);
//         va0702 = __dtu_m_vmm_mode11_f_vs0(va0702, iv2, smr);
//         va0703 = __dtu_m_vmm_mode11_f_vs0(va0703, iv3, smr);
//         va0704 = __dtu_m_vmm_mode11_f_vs0(va0704, iv4, smr);
//         va0705 = __dtu_m_vmm_mode11_f_vs0(va0705, iv5, smr);
//         va0706 = __dtu_m_vmm_mode11_f_vs0(va0706, iv6, smr);
//         va0707 = __dtu_m_vmm_mode11_f_vs0(va0707, iv7, smr);
//         va0708 = __dtu_m_vmm_mode11_f_vs0(va0708, iv8, smr);
//         va0709 = __dtu_m_vmm_mode11_f_vs0(va0709, iv9, smr);
//         va0710 = __dtu_m_vmm_mode11_f_vs0(va0710, iv10, smr);
//         va0711 = __dtu_m_vmm_mode11_f_vs0(va0711, iv11, smr);
//         va0712 = __dtu_m_vmm_mode11_f_vs0(va0712, iv12, smr);
//         va0713 = __dtu_m_vmm_mode11_f_vs0(va0713, iv13, smr);
//         va0714 = __dtu_m_vmm_mode11_f_vs0(va0714, iv14, smr);
//         va0715 = __dtu_m_vmm_mode11_f_vs0(va0715, iv15, smr);
//         va0716 = __dtu_m_vmm_mode11_f_vs0(va0716, iv16, smr);
//         va0717 = __dtu_m_vmm_mode11_f_vs0(va0717, iv17, smr);
//         va0718 = __dtu_m_vmm_mode11_f_vs0(va0718, iv18, smr);
//         va0719 = __dtu_m_vmm_mode11_f_vs0(va0719, iv19, smr);
//         va0720 = __dtu_m_vmm_mode11_f_vs0(va0720, iv20, smr);
//         va0721 = __dtu_m_vmm_mode11_f_vs0(va0721, iv21, smr);
//         va0722 = __dtu_m_vmm_mode11_f_vs0(va0722, iv22, smr);
//         va0723 = __dtu_m_vmm_mode11_f_vs0(va0723, iv23, smr);
//         va0724 = __dtu_m_vmm_mode11_f_vs0(va0724, iv24, smr);
//         va0725 = __dtu_m_vmm_mode11_f_vs0(va0725, iv25, smr);
//         va0726 = __dtu_m_vmm_mode11_f_vs0(va0726, iv26, smr);
//         va0727 = __dtu_m_vmm_mode11_f_vs0(va0727, iv27, smr);
//         va0728 = __dtu_m_vmm_mode11_f_vs0(va0728, iv28, smr);
//         va0729 = __dtu_m_vmm_mode11_f_vs0(va0729, iv29, smr);
//         va0730 = __dtu_m_vmm_mode11_f_vs0(va0730, iv30, smr);
//         va0731 = __dtu_m_vmm_mode11_f_vs0(va0731, iv31, smr);

//         // load input[n][ci][2][2]
//         iv0 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv1 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv2 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv3 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv4 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv5 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv6 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv7 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv8 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv9 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv10 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv11 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv12 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv13 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv14 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv15 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv16 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv17 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv18 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv19 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv20 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv21 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv22 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv23 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv24 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv25 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv26 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv27 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv28 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv29 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv30 = __dtu_s_tivld_itar(input_addr, input_offset0);
//         iv31 = __dtu_s_tivld_itar(input_addr, input_offset3);
//         va0800 = __dtu_m_vmm_mode11_f_vs0(va0800, iv0, smr);
//         va0801 = __dtu_m_vmm_mode11_f_vs0(va0801, iv1, smr);
//         va0802 = __dtu_m_vmm_mode11_f_vs0(va0802, iv2, smr);
//         va0803 = __dtu_m_vmm_mode11_f_vs0(va0803, iv3, smr);
//         va0804 = __dtu_m_vmm_mode11_f_vs0(va0804, iv4, smr);
//         va0805 = __dtu_m_vmm_mode11_f_vs0(va0805, iv5, smr);
//         va0806 = __dtu_m_vmm_mode11_f_vs0(va0806, iv6, smr);
//         va0807 = __dtu_m_vmm_mode11_f_vs0(va0807, iv7, smr);
//         va0808 = __dtu_m_vmm_mode11_f_vs0(va0808, iv8, smr);
//         va0809 = __dtu_m_vmm_mode11_f_vs0(va0809, iv9, smr);
//         va0810 = __dtu_m_vmm_mode11_f_vs0(va0810, iv10, smr);
//         va0811 = __dtu_m_vmm_mode11_f_vs0(va0811, iv11, smr);
//         va0812 = __dtu_m_vmm_mode11_f_vs0(va0812, iv12, smr);
//         va0813 = __dtu_m_vmm_mode11_f_vs0(va0813, iv13, smr);
//         va0814 = __dtu_m_vmm_mode11_f_vs0(va0814, iv14, smr);
//         va0815 = __dtu_m_vmm_mode11_f_vs0(va0815, iv15, smr);
//         va0816 = __dtu_m_vmm_mode11_f_vs0(va0816, iv16, smr);
//         va0817 = __dtu_m_vmm_mode11_f_vs0(va0817, iv17, smr);
//         va0818 = __dtu_m_vmm_mode11_f_vs0(va0818, iv18, smr);
//         va0819 = __dtu_m_vmm_mode11_f_vs0(va0819, iv19, smr);
//         va0820 = __dtu_m_vmm_mode11_f_vs0(va0820, iv20, smr);
//         va0821 = __dtu_m_vmm_mode11_f_vs0(va0821, iv21, smr);
//         va0822 = __dtu_m_vmm_mode11_f_vs0(va0822, iv22, smr);
//         va0823 = __dtu_m_vmm_mode11_f_vs0(va0823, iv23, smr);
//         va0824 = __dtu_m_vmm_mode11_f_vs0(va0824, iv24, smr);
//         va0825 = __dtu_m_vmm_mode11_f_vs0(va0825, iv25, smr);
//         va0826 = __dtu_m_vmm_mode11_f_vs0(va0826, iv26, smr);
//         va0827 = __dtu_m_vmm_mode11_f_vs0(va0827, iv27, smr);
//         va0828 = __dtu_m_vmm_mode11_f_vs0(va0828, iv28, smr);
//         va0829 = __dtu_m_vmm_mode11_f_vs0(va0829, iv29, smr);
//         va0830 = __dtu_m_vmm_mode11_f_vs0(va0830, iv30, smr);
//         va0831 = __dtu_m_vmm_mode11_f_vs0(va0831, iv31, smr);

//         smr = __dtu_v_swapsmr(smr);
//         __dtu_c_movsr2naccovr(0x1);
//       }  // end ci
//       iv0 = __dtu_s_tivld_itar(input_addr, input_offset4);
//     }  // end s
//     iv0 = __dtu_s_tivld_itar(input_addr, input_offset5);
//   }  // end r
//   if (st_flag == 1) {
//     __dtu_l_tvsta_w_q(va0000, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0100, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0200, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0300, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0400, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0500, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0600, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0700, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0800, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0001, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0101, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0201, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0301, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0401, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0501, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0601, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0701, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0801, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0002, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0102, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0202, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0302, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0402, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0502, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0602, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0702, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0802, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0003, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0103, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0203, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0303, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0403, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0503, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0603, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0703, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0803, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0004, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0104, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0204, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0304, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0404, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0504, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0604, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0704, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0804, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0005, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0105, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0205, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0305, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0405, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0505, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0605, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0705, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0805, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0006, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0106, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0206, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0306, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0406, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0506, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0606, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0706, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0806, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0007, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0107, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0207, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0307, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0407, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0507, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0607, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0707, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0807, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0008, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0108, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0208, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0308, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0408, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0508, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0608, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0708, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0808, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0009, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0109, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0209, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0309, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0409, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0509, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0609, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0709, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0809, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0010, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0110, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0210, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0310, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0410, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0510, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0610, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0710, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0810, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0011, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0111, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0211, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0311, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0411, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0511, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0611, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0711, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0811, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0012, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0112, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0212, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0312, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0412, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0512, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0612, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0712, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0812, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0013, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0113, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0213, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0313, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0413, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0513, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0613, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0713, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0813, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0014, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0114, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0214, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0314, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0414, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0514, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0614, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0714, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0814, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0015, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0115, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0215, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0315, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0415, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0515, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0615, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0715, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0815, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0016, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0116, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0216, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0316, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0416, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0516, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0616, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0716, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0816, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0017, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0117, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0217, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0317, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0417, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0517, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0617, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0717, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0817, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0018, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0118, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0218, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0318, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0418, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0518, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0618, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0718, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0818, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0019, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0119, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0219, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0319, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0419, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0519, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0619, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0719, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0819, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0020, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0120, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0220, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0320, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0420, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0520, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0620, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0720, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0820, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0021, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0121, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0221, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0321, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0421, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0521, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0621, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0721, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0821, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0022, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0122, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0222, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0322, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0422, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0522, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0622, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0722, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0822, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0023, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0123, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0223, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0323, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0423, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0523, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0623, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0723, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0823, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0024, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0124, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0224, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0324, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0424, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0524, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0624, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0724, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0824, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0025, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0125, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0225, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0325, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0425, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0525, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0625, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0725, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0825, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0026, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0126, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0226, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0326, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0426, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0526, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0626, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0726, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0826, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0027, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0127, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0227, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0327, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0427, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0527, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0627, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0727, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0827, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0028, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0128, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0228, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0328, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0428, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0528, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0628, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0728, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0828, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0029, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0129, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0229, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0329, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0429, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0529, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0629, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0729, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0829, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0030, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0130, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0230, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0330, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0430, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0530, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0630, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0730, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0830, output_addr, output_offset);

//     __dtu_l_tvsta_w_q(va0031, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0131, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0231, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0331, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0431, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0531, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0631, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0731, output_addr, output_offset);
//     __dtu_l_tvsta_w_q(va0831, output_addr, output_offset);
//   }

//   __dtu_c_movsr2naccovr(0);
// }

extern "C" void conv3d_bpk_ef32_kernel(memref* lhs_param, memref* rhs_param,
                                       memref* out_param, int N, int Hi, int Wi,
                                       int Ci, int R, int S, int Co, int Ho,
                                       int Wo, int stride_h, int stride_w,
                                       int window_dilation_h,
                                       int window_dilation_w, int ld_flag,
                                       int st_flag) {
  if (N == 8 && Ho == 1 && Wo == 1) {
    conv3d_kerneln8ho1wo1(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                          Co, Ho, Wo, stride_h, stride_w, window_dilation_h,
                          window_dilation_w, ld_flag, st_flag);
  }
  if (N == 8 && Ho == 2 && Wo == 2) {
    conv3d_kerneln8ho2wo2(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                          Co, Ho, Wo, stride_h, stride_w, window_dilation_h,
                          window_dilation_w, ld_flag, st_flag);
  }
  if (N == 8 && Ho == 3 && Wo == 3) {
    conv3d_kerneln8ho3wo3(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                          Co, Ho, Wo, stride_h, stride_w, window_dilation_h,
                          window_dilation_w, ld_flag, st_flag);
  }
  if (N == 16 && Ho == 1 && Wo == 1) {
    conv3d_kerneln16ho1wo1(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                           Co, Ho, Wo, stride_h, stride_w, window_dilation_h,
                           window_dilation_w, ld_flag, st_flag);
  }
  if (N == 16 && Ho == 2 && Wo == 2) {
    conv3d_kerneln16ho2wo2(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                           Co, Ho, Wo, stride_h, stride_w, window_dilation_h,
                           window_dilation_w, ld_flag, st_flag);
  }
  if (N == 16 && Ho == 3 && Wo == 3) {
    conv3d_kerneln16ho3wo3(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                           Co, Ho, Wo, stride_h, stride_w, window_dilation_h,
                           window_dilation_w, ld_flag, st_flag);
  }
  if (N == 32 && Ho == 1 && Wo == 1) {
    conv3d_kerneln32ho1wo1(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R, S,
                           Co, Ho, Wo, stride_h, stride_w, window_dilation_h,
                           window_dilation_w, ld_flag, st_flag);
  }
  // if (N == 32 && Ho == 2 && Wo == 2) {
  //   conv3d_kerneln32ho2wo2(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R,
  //   S,
  //                          Co, Ho, Wo, stride_h, stride_w, window_dilation_h,
  //                          window_dilation_w, ld_flag, st_flag);
  // }
  // if (N == 32 && Ho == 3 && Wo == 3) {
  //   conv3d_kerneln32ho3wo3(lhs_param, rhs_param, out_param, N, Hi, Wi, Ci, R,
  //   S,
  //                          Co, Ho, Wo, stride_h, stride_w, window_dilation_h,
  //                          window_dilation_w, ld_flag, st_flag);
  // }
}