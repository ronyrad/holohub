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
#include "adv_network_media_rx_common.h"
#include "../common/frame_buffer.h"

namespace holoscan::ops {
class PacketsToFramesConsumerUser {
 public:
  /**
   * @brief Called when a new frame has been fully constructed.
   * 
   * @param frame The completed frame.
   */
  virtual void on_new_frame(std::shared_ptr<FrameBufferBase> frame) = 0;

  /**
   * @brief Retrieves an allocated frame from the pool.
   * 
   * @return Shared pointer to an allocated frame.
   */
  virtual std::shared_ptr<FrameBufferBase> get_allocated_frame() = 0;
};

class PacketsToFramesConsumer {
 public:
  explicit PacketsToFramesConsumer(PacketsToFramesConsumerUser *user);
  void process_incoming_packet(const RTP_EX_SRDS *header, uint8_t *payload);
  void reset_frame_state();
  bool is_processing_copy() { return contiguous_mem_to_copy_ > 0; }

 private:
  PacketsToFramesConsumerUser *user_;
  std::shared_ptr<FrameBufferBase> frame_;
  size_t contiguous_mem_to_copy_;
  size_t current_byte_in_frame_;
  bool waiting_for_end_of_frame_;
  uint8_t* current_payload_start_ptr_;

 private:
  std::pair<bool, std::string> validate_packet_integrity(const RTP_EX_SRDS* header);
};

}  // namespace holoscan::ops
