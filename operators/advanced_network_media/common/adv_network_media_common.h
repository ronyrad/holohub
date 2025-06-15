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
#include <holoscan/holoscan.hpp>
#include "rtp_params.h"

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