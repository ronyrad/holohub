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

#include <cuda_runtime.h>
#include "advanced_network/common.h"
#include "packets_to_frames_consumer.h"

namespace holoscan::ops {
PacketsToFramesConsumer::PacketsToFramesConsumer(PacketsToFramesConsumerUser* user)
    : user_(user),
      frame_(user->get_allocated_frame()),
      contiguous_mem_to_copy_(0),
      waiting_for_end_of_frame_(false),
      current_byte_in_frame_(0),
      current_payload_start_ptr_(nullptr) {}

void PacketsToFramesConsumer::process_incoming_packet(const RTP_EX_SRDS* header, uint8_t* payload) {
  if (waiting_for_end_of_frame_) {
    if (header->m) {
      HOLOSCAN_LOG_INFO("End of frame received restarting");
      waiting_for_end_of_frame_ = false;
      reset_frame_state();
    }
    return;
  }

  if (current_payload_start_ptr_ == nullptr) { current_payload_start_ptr_ = payload; }

  bool is_contiguous_memory = current_payload_start_ptr_ + contiguous_mem_to_copy_ == payload;
  if (!is_contiguous_memory) {
    // Determine the appropriate memory copy direction based on frame buffer's memory type
    cudaMemcpyKind copy_kind;
    if (frame_->get_memory_location() == MemoryLocation::Host) {
      copy_kind = cudaMemcpyDeviceToHost;
    } else {
      copy_kind = cudaMemcpyDeviceToDevice;
    }

    CUDA_TRY(cudaMemcpy(static_cast<uint8_t*>(frame_->get()) + current_byte_in_frame_,
                        current_payload_start_ptr_,
                        contiguous_mem_to_copy_,
                        copy_kind));
    current_payload_start_ptr_ = payload;
    current_byte_in_frame_ += contiguous_mem_to_copy_;
    contiguous_mem_to_copy_ = 0;
  }

  contiguous_mem_to_copy_ += ntohs(header->srdLength1);
  const auto& [is_corrupted, error] = validate_packet_integrity(header);
  if (is_corrupted) {
    HOLOSCAN_LOG_ERROR("Frame is corrupted: {}", error);
    if (!header->m) {
      waiting_for_end_of_frame_ = true;
    } else {
      HOLOSCAN_LOG_INFO("End of frame received restarting");
      reset_frame_state();
    }
    return;
  }

  int64_t bytes_left_after_copy =
      frame_->get_size() - current_byte_in_frame_ - contiguous_mem_to_copy_;
  bool frame_full = bytes_left_after_copy == 0;

  if (frame_full && header->m) {
    // Determine the appropriate memory copy direction based on frame buffer's memory type
    cudaMemcpyKind copy_kind;
    if (frame_->get_memory_location() == MemoryLocation::Host) {
      copy_kind = cudaMemcpyDeviceToHost;
    } else {
      copy_kind = cudaMemcpyDeviceToDevice;
    }

    CUDA_TRY(cudaMemcpy(static_cast<uint8_t*>(frame_->get()) + current_byte_in_frame_,
                        current_payload_start_ptr_,
                        contiguous_mem_to_copy_,
                        copy_kind));
    user_->on_new_frame(frame_);
    frame_ = user_->get_allocated_frame();
    reset_frame_state();
    return;
  }
}

void PacketsToFramesConsumer::reset_frame_state() {
  current_payload_start_ptr_ = nullptr;
  contiguous_mem_to_copy_ = 0;
  current_byte_in_frame_ = 0;
}

std::pair<bool, std::string> PacketsToFramesConsumer::validate_packet_integrity(
    const RTP_EX_SRDS* header) {
  int64_t bytes_left_after_copy =
      frame_->get_size() - current_byte_in_frame_ - contiguous_mem_to_copy_;
  bool frame_full = bytes_left_after_copy == 0;

  if (bytes_left_after_copy < 0) {
    return {true, "Frame received is not aligned to the frame size and will be dropped"};
  }

  if (frame_full && !header->m) { return {true, "Frame is full but marker was not not appear"}; }

  if (!frame_full && header->m) { return {true, "Marker appeared but frame is not full"}; }
  return {false, ""};
}
}  // namespace holoscan::ops
