/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cstdint>
#include <stddef.h>
#include <arpa/inet.h>
#include <holoscan/holoscan.hpp>

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                         #stmt,                                                                 \
                         __LINE__,                                                              \
                         __FILE__,                                                              \
                         cudaGetErrorString(_holoscan_cuda_err),                                \
                         static_cast<int>(_holoscan_cuda_err));                                 \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })

// this code is taken from rivermax media_receiver app:
// https://github.com/Mellanox/rivermax/blob/e9d6294346e5b100d718641ac7dbdbfff1effdbe/tests/media_receiver/viewer.cpp#L1032
#ifdef _MSC_VER
#define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#elif defined(__GNUC__)
#define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

PACK(struct RTP_EX_SRDS {
  uint32_t cc : 4;
  uint32_t x : 1;
  uint32_t p : 1;
  uint32_t v : 2;
  uint32_t pt : 7;
  uint32_t m : 1;
  uint32_t seq : 16;
  uint32_t timeStamp : 32;
  uint32_t ssrc : 32;
  uint32_t extSeqNum : 16;
  // SRD 1
  uint32_t srdLength1 : 16;
  uint32_t srdRowNumHi1 : 7;
  uint32_t f1 : 1;
  uint32_t srdRowNumLo1 : 8;
  uint32_t srdOffsetHi1 : 7;
  uint32_t c1 : 1;
  uint32_t srdOffsetLo1 : 8;

  uint64_t get_sequence() {
    return (static_cast<uint32_t>(ntohs(seq)) << 16) | static_cast<uint32_t>(extSeqNum);
  }
});
