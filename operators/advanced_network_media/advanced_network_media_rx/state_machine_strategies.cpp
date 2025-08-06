/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "state_machine_strategies.h"
#include "../common/adv_network_media_common.h"
#include <algorithm>
#include <optional>

namespace holoscan::ops {

// ========================================================================================
// StrategyFactory Implementation
// ========================================================================================

std::unique_ptr<StrategyDetector> StrategyFactory::create_detector() {
  return std::make_unique<StrategyDetector>();
}

std::unique_ptr<IStateMachineAwareStrategy> StrategyFactory::create_contiguous_strategy(
    nvidia::gxf::MemoryStorageType src_storage_type,
    nvidia::gxf::MemoryStorageType dst_storage_type) {
  return std::make_unique<StateMachineContiguousStrategy>(src_storage_type, dst_storage_type);
}

std::unique_ptr<IStateMachineAwareStrategy> StrategyFactory::create_strided_strategy(
    const StrideInfo& stride_info,
    nvidia::gxf::MemoryStorageType src_storage_type,
    nvidia::gxf::MemoryStorageType dst_storage_type) {
  return std::make_unique<StateMachineStridedStrategy>(stride_info, src_storage_type, dst_storage_type);
}

// ========================================================================================
// StrategyDetector Implementation  
// ========================================================================================

void StrategyDetector::configure_burst_parameters(size_t header_stride_size, 
                                                  size_t payload_stride_size, 
                                                  bool hds_enabled) {
  if (detection_complete_) {
    HOLOSCAN_LOG_DEBUG("Strategy already detected, ignoring burst parameter update");
    return;
  }
  
  // Check if configuration changed during detection
  if (packets_analyzed_ > 0) {
    bool config_changed = (expected_header_stride_ != header_stride_size ||
                          expected_payload_stride_ != payload_stride_size ||
                          hds_enabled_ != hds_enabled);
    if (config_changed) {
      HOLOSCAN_LOG_WARN("Burst configuration changed during detection, restarting analysis");
      reset();
    }
  }
  
  expected_header_stride_ = header_stride_size;
  expected_payload_stride_ = payload_stride_size;
  hds_enabled_ = hds_enabled;
  
  HOLOSCAN_LOG_DEBUG("Strategy detector configured: header_stride={}, payload_stride={}, hds_enabled={}",
                     header_stride_size, payload_stride_size, hds_enabled);
}

bool StrategyDetector::collect_packet(const RtpParams& rtp_params, uint8_t* payload, size_t payload_size) {
  if (detection_complete_) {
    return true;
  }
  
  // Validate input
  if (!payload || payload_size == 0) {
    HOLOSCAN_LOG_WARN("Invalid packet data for strategy detection, skipping");
    return false;
  }
  
  // Store packet information
  collected_payloads_.push_back(payload);
  collected_payload_sizes_.push_back(payload_size);
  collected_sequences_.push_back(rtp_params.sequence_number);
  packets_analyzed_++;
  
  PACKET_TRACE_LOG("Collected packet {} for detection: payload={}, size={}, seq={}",
                     packets_analyzed_, static_cast<void*>(payload), payload_size, rtp_params.sequence_number);
  
  return packets_analyzed_ >= DETECTION_PACKET_COUNT;
}

std::unique_ptr<IStateMachineAwareStrategy> StrategyDetector::detect_strategy(
    nvidia::gxf::MemoryStorageType src_storage_type,
    nvidia::gxf::MemoryStorageType dst_storage_type) {
  
  if (collected_payloads_.size() < 2) {
    HOLOSCAN_LOG_INFO("Insufficient packets for analysis ({} < 2), defaulting to CONTIGUOUS", 
                      collected_payloads_.size());
    detection_complete_ = true;
    return StrategyFactory::create_contiguous_strategy(src_storage_type, dst_storage_type);
  }
  
  // Validate sequence continuity
  if (!validate_sequence_continuity()) {
    HOLOSCAN_LOG_WARN("RTP sequence drops detected, restarting detection");
    reset();
    return nullptr;
  }
  
  // Check for buffer wraparound
  if (detect_buffer_wraparound()) {
    HOLOSCAN_LOG_WARN("Buffer wraparound detected, restarting detection");
    reset();
    return nullptr;
  }
  
  // Analyze pattern
  auto analysis_result = analyze_pattern();
  if (!analysis_result) {
    HOLOSCAN_LOG_WARN("Pattern analysis failed, restarting detection");
    reset();
    return nullptr;
  }
  
  auto [strategy_type, stride_info] = *analysis_result;
  detection_complete_ = true;
  
  HOLOSCAN_LOG_INFO("Strategy detection completed: {} (stride: {}, payload: {})",
                    strategy_type == CopyStrategy::CONTIGUOUS ? "CONTIGUOUS" : "STRIDED",
                    stride_info.stride_size, stride_info.payload_size);
  
  if (strategy_type == CopyStrategy::CONTIGUOUS) {
    return StrategyFactory::create_contiguous_strategy(src_storage_type, dst_storage_type);
  } else {
    return StrategyFactory::create_strided_strategy(stride_info, src_storage_type, dst_storage_type);
  }
}

void StrategyDetector::reset() {
  collected_payloads_.clear();
  collected_payload_sizes_.clear();
  collected_sequences_.clear();
  packets_analyzed_ = 0;
  detection_complete_ = false;
  
  HOLOSCAN_LOG_DEBUG("Strategy detector reset");
}

std::optional<std::pair<CopyStrategy, StrideInfo>> StrategyDetector::analyze_pattern() {
  if (collected_payloads_.size() < 2) {
    return std::nullopt;
  }
  
  // Verify payload size consistency
  size_t payload_size = collected_payload_sizes_[0];
  for (size_t i = 1; i < collected_payload_sizes_.size(); ++i) {
    if (collected_payload_sizes_[i] != payload_size) {
      HOLOSCAN_LOG_WARN("Inconsistent payload sizes: first={}, packet_{}={}",
                        payload_size, i, collected_payload_sizes_[i]);
      return std::nullopt;
    }
  }
  
  // Analyze memory layout
  bool is_exactly_contiguous = true;
  bool is_stride_consistent = true;
  size_t actual_stride = 0;
  
  for (size_t i = 1; i < collected_payloads_.size(); ++i) {
    uint8_t* prev_ptr = collected_payloads_[i - 1];
    uint8_t* curr_ptr = collected_payloads_[i];
    
    size_t pointer_diff = curr_ptr - prev_ptr;
    
    if (i == 1) {
      actual_stride = pointer_diff;
    }
    
    // Check exact contiguity
    uint8_t* expected_next_ptr = prev_ptr + payload_size;
    if (curr_ptr != expected_next_ptr) {
      is_exactly_contiguous = false;
      PACKET_TRACE_LOG("Non-contiguous detected: packet {}, expected={}, actual={}",
                         i, static_cast<void*>(expected_next_ptr), static_cast<void*>(curr_ptr));
    }
    
    // Check stride consistency
    if (pointer_diff != actual_stride) {
      is_stride_consistent = false;
      PACKET_TRACE_LOG("Inconsistent stride: packet {}, expected={}, actual={}",
                         i, actual_stride, pointer_diff);
      break;
    }
  }
  
  // Create stride info
  StrideInfo stride_info;
  stride_info.stride_size = actual_stride;
  stride_info.payload_size = payload_size;
  
  // Determine strategy
  CopyStrategy strategy;
  if (is_exactly_contiguous) {
    strategy = CopyStrategy::CONTIGUOUS;
    HOLOSCAN_LOG_DEBUG("Packets are exactly contiguous, using CONTIGUOUS strategy");
  } else if (is_stride_consistent) {
    strategy = CopyStrategy::STRIDED;
    HOLOSCAN_LOG_DEBUG("Consistent stride pattern detected, using STRIDED strategy (stride={}, payload={})",
                       actual_stride, payload_size);
  } else {
    strategy = CopyStrategy::CONTIGUOUS;
    HOLOSCAN_LOG_DEBUG("Inconsistent patterns, falling back to CONTIGUOUS strategy");
  }
  
  return std::make_pair(strategy, stride_info);
}

bool StrategyDetector::validate_sequence_continuity() const {
  if (collected_sequences_.size() < 2) {
    return true;
  }
  
  for (size_t i = 1; i < collected_sequences_.size(); ++i) {
    uint64_t prev_seq = collected_sequences_[i - 1];
    uint64_t curr_seq = collected_sequences_[i];
    uint64_t expected_seq = prev_seq + 1;
    
    if (curr_seq != expected_seq) {
      PACKET_TRACE_LOG("RTP sequence discontinuity: expected {}, got {} (prev was {})",
                         expected_seq, curr_seq, prev_seq);
      return false;
    }
  }
  
  return true;
}

bool StrategyDetector::detect_buffer_wraparound() const {
  if (collected_payloads_.size() < 2) {
    return false;
  }
  
  for (size_t i = 1; i < collected_payloads_.size(); ++i) {
    uint8_t* prev_ptr = collected_payloads_[i - 1];
    uint8_t* curr_ptr = collected_payloads_[i];
    
    if (curr_ptr < prev_ptr) {
      ptrdiff_t backward_diff = prev_ptr - curr_ptr;
      if (backward_diff > 1024 * 1024) { // 1MB threshold
        PACKET_TRACE_LOG("Potential buffer wraparound: {} -> {}",
                          static_cast<void*>(prev_ptr), static_cast<void*>(curr_ptr));
        return true;
      }
    }
  }
  
  return false;
}

// ========================================================================================
// StateMachineContiguousStrategy Implementation
// ========================================================================================

StateMachineContiguousStrategy::StateMachineContiguousStrategy(
    nvidia::gxf::MemoryStorageType src_storage_type,
    nvidia::gxf::MemoryStorageType dst_storage_type)
    : src_storage_type_(src_storage_type)
    , dst_storage_type_(dst_storage_type)
    , copy_kind_(CopyOperationHelper::get_copy_kind(src_storage_type, dst_storage_type)) {
}

StateEvent StateMachineContiguousStrategy::process_packet(FrameProcessingStateMachine& state_machine,
                                                         uint8_t* payload, 
                                                         size_t payload_size) {
  // Input validation for packet processing
  if (!payload || payload_size == 0) {
    HOLOSCAN_LOG_ERROR("ContiguousStrategy: Invalid packet data");
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  // Initialize accumulation if needed
  if (!accumulated_start_ptr_) {
    accumulated_start_ptr_ = payload;
    accumulated_size_ = 0;
    PACKET_TRACE_LOG("ContiguousStrategy: Starting new accumulation at {}", 
                       static_cast<void*>(payload));
  }
  
  // Check memory contiguity
  bool is_contiguous = (accumulated_start_ptr_ + accumulated_size_ == payload);
  
  if (!is_contiguous) {
    PACKET_TRACE_LOG("ContiguousStrategy: Contiguity break, executing copy for {} bytes",
                       accumulated_size_);
    
    // Execute pending copy before starting new accumulation
    StateEvent copy_result = execute_copy(state_machine);
    if (copy_result == StateEvent::CORRUPTION_DETECTED) {
      return copy_result;
    }
    
    // Start new accumulation
    accumulated_start_ptr_ = payload;
    accumulated_size_ = 0;
  }
  
  // Add current packet to accumulation
  accumulated_size_ += payload_size;
  
  // Safety check for frame bounds
  auto frame = state_machine.get_current_frame();
  if (frame && accumulated_size_ > frame->get_size()) {
    HOLOSCAN_LOG_ERROR("ContiguousStrategy: Accumulated size ({}) exceeds frame size ({})",
                       accumulated_size_, frame->get_size());
    reset();
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  return StateEvent::PACKET_ARRIVED;
}

bool StateMachineContiguousStrategy::has_pending_operations() const {
  return accumulated_size_ > 0 && accumulated_start_ptr_ != nullptr;
}

void StateMachineContiguousStrategy::reset() {
  accumulated_start_ptr_ = nullptr;
  accumulated_size_ = 0;
  PACKET_TRACE_LOG("ContiguousStrategy: Reset accumulation state");
}

StateEvent StateMachineContiguousStrategy::execute_pending_copy(FrameProcessingStateMachine& state_machine) {
  return execute_copy(state_machine);
}

StateEvent StateMachineContiguousStrategy::execute_copy(FrameProcessingStateMachine& state_machine) {
  if (!has_pending_operations()) {
    return StateEvent::COPY_EXECUTED;
  }
  
  // Validate copy bounds
  if (!validate_copy_bounds(state_machine)) {
    HOLOSCAN_LOG_ERROR("ContiguousStrategy: Copy bounds validation failed");
    reset();
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  auto frame = state_machine.get_current_frame();
  uint8_t* dst_ptr = static_cast<uint8_t*>(frame->get()) + state_machine.get_frame_position();
  
  PACKET_TRACE_LOG("ContiguousStrategy: Executing copy - pos={}, size={}, frame_size={}",
                     state_machine.get_frame_position(), accumulated_size_, frame->get_size());
  
  // Execute copy operation
  if (!CopyOperationHelper::safe_copy(dst_ptr, accumulated_start_ptr_, accumulated_size_, copy_kind_)) {
    HOLOSCAN_LOG_ERROR("ContiguousStrategy: Copy operation failed");
    reset();
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  // Update frame position
  state_machine.advance_frame_position(accumulated_size_);
  
  PACKET_TRACE_LOG("ContiguousStrategy: Copy completed - new_pos={}, copied={}",
                     state_machine.get_frame_position(), accumulated_size_);
  
  // Reset accumulation state
  reset();
  
  return StateEvent::COPY_EXECUTED;
}

bool StateMachineContiguousStrategy::validate_copy_bounds(FrameProcessingStateMachine& state_machine) const {
  auto frame = state_machine.get_current_frame();
  if (!frame) {
    return false;
  }
  
  size_t current_pos = state_machine.get_frame_position();
  size_t frame_size = frame->get_size();
  
  return (current_pos + accumulated_size_ <= frame_size);
}

// ========================================================================================
// StateMachineStridedStrategy Implementation
// ========================================================================================

StateMachineStridedStrategy::StateMachineStridedStrategy(
    const StrideInfo& stride_info,
    nvidia::gxf::MemoryStorageType src_storage_type,
    nvidia::gxf::MemoryStorageType dst_storage_type)
    : stride_info_(stride_info)
    , src_storage_type_(src_storage_type)
    , dst_storage_type_(dst_storage_type)
    , copy_kind_(CopyOperationHelper::get_copy_kind(src_storage_type, dst_storage_type)) {
}

StateEvent StateMachineStridedStrategy::process_packet(FrameProcessingStateMachine& state_machine,
                                                      uint8_t* payload, 
                                                      size_t payload_size) {
  // Input validation
  if (!payload || payload_size == 0) {
    HOLOSCAN_LOG_ERROR("StridedStrategy: Invalid packet data");
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  // Initialize accumulation if needed
  if (!first_packet_ptr_) {
    reset_accumulation(payload, payload_size);
    PACKET_TRACE_LOG("StridedStrategy: Starting accumulation with first packet at {}",
                       static_cast<void*>(payload));
    return StateEvent::PACKET_ARRIVED;
  }
  
  // Check if stride pattern is maintained
  if (!is_stride_maintained(payload, payload_size)) {
    PACKET_TRACE_LOG("StridedStrategy: Stride pattern broken, executing copy and restarting");
    
    // Execute accumulated copy if we have multiple packets
    StateEvent copy_result;
    if (accumulated_packet_count_ > 1) {
      copy_result = execute_strided_copy(state_machine);
    } else {
      copy_result = execute_individual_copy(state_machine, first_packet_ptr_, 
                                           stride_info_.payload_size);
    }
    
    if (copy_result == StateEvent::CORRUPTION_DETECTED) {
      return copy_result;
    }
    
    // Reset and start with current packet
    reset_accumulation(payload, payload_size);
    return StateEvent::PACKET_ARRIVED;
  }
  
  // Stride is maintained, continue accumulation
  last_packet_ptr_ = payload;
  accumulated_packet_count_++;
  accumulated_data_size_ += payload_size;
  
  PACKET_TRACE_LOG("StridedStrategy: Accumulated packet {}, total_size={}",
                     accumulated_packet_count_, accumulated_data_size_);
  
  return StateEvent::PACKET_ARRIVED;
}

bool StateMachineStridedStrategy::has_pending_operations() const {
  return accumulated_packet_count_ > 0 && first_packet_ptr_ != nullptr;
}

StateEvent StateMachineStridedStrategy::execute_pending_copy(FrameProcessingStateMachine& state_machine) {
  if (!has_pending_operations()) {
    return StateEvent::COPY_EXECUTED;
  }
  
  // Execute accumulated copy if we have multiple packets
  StateEvent copy_result;
  if (accumulated_packet_count_ > 1 && stride_validated_) {
    copy_result = execute_strided_copy(state_machine);
  } else {
    copy_result = execute_individual_copy(state_machine, first_packet_ptr_, stride_info_.payload_size);
  }
  
  if (copy_result == StateEvent::COPY_EXECUTED) {
    reset();
  }
  
  return copy_result;
}

void StateMachineStridedStrategy::reset() {
  first_packet_ptr_ = nullptr;
  last_packet_ptr_ = nullptr;
  accumulated_packet_count_ = 0;
  accumulated_data_size_ = 0;
  stride_validated_ = false;
  actual_stride_ = 0;
  PACKET_TRACE_LOG("StridedStrategy: Reset accumulation state");
}

bool StateMachineStridedStrategy::is_stride_maintained(uint8_t* payload, size_t payload_size) {
  if (!last_packet_ptr_) {
    // First stride check
    return true;
  }
  
  size_t actual_diff = payload - last_packet_ptr_;
  
  if (!stride_validated_) {
    // First stride validation
    actual_stride_ = actual_diff;
    stride_validated_ = true;
    
    if (actual_stride_ != stride_info_.stride_size) {
      PACKET_TRACE_LOG("StridedStrategy: Actual stride ({}) differs from expected ({})",
                         actual_stride_, stride_info_.stride_size);
    }
    
    return true;
  } else {
    // Subsequent stride validation
    if (actual_diff != actual_stride_) {
      PACKET_TRACE_LOG("StridedStrategy: Stride inconsistent: expected={}, actual={}",
                         actual_stride_, actual_diff);
      return false;
    }
    return true;
  }
}

StateEvent StateMachineStridedStrategy::execute_strided_copy(FrameProcessingStateMachine& state_machine) {
  if (accumulated_packet_count_ <= 1 || !first_packet_ptr_) {
    return StateEvent::COPY_EXECUTED;
  }
  
  // Validate copy bounds
  if (!validate_strided_copy_bounds(state_machine)) {
    HOLOSCAN_LOG_ERROR("StridedStrategy: Strided copy bounds validation failed");
    reset();
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  auto frame = state_machine.get_current_frame();
  uint8_t* dst_ptr = static_cast<uint8_t*>(frame->get()) + state_machine.get_frame_position();
  
  // Setup 2D copy parameters
  size_t width = stride_info_.payload_size;
  size_t height = accumulated_packet_count_;
  size_t src_pitch = actual_stride_;
  size_t dst_pitch = width; // Contiguous destination
  
  PACKET_TRACE_LOG("StridedStrategy: Executing 2D copy - width={}, height={}, src_pitch={}, dst_pitch={}",
                     width, height, src_pitch, dst_pitch);
  
  // Execute 2D copy
  if (!CopyOperationHelper::safe_copy_2d(dst_ptr, dst_pitch, first_packet_ptr_, src_pitch,
                                        width, height, copy_kind_)) {
    HOLOSCAN_LOG_ERROR("StridedStrategy: 2D copy operation failed");
    reset();
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  // Update frame position
  state_machine.advance_frame_position(accumulated_data_size_);
  
  PACKET_TRACE_LOG("StridedStrategy: Strided copy completed - new_pos={}, copied={}",
                     state_machine.get_frame_position(), accumulated_data_size_);
  
  // Reset accumulation
  reset();
  
  return StateEvent::COPY_EXECUTED;
}

StateEvent StateMachineStridedStrategy::execute_individual_copy(FrameProcessingStateMachine& state_machine,
                                                               uint8_t* payload, 
                                                               size_t payload_size) {
  auto frame = state_machine.get_current_frame();
  if (!frame) {
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  uint8_t* dst_ptr = static_cast<uint8_t*>(frame->get()) + state_machine.get_frame_position();
  
  // Bounds checking
  if (state_machine.get_frame_position() + payload_size > frame->get_size()) {
    HOLOSCAN_LOG_ERROR("StridedStrategy: Individual copy would exceed frame bounds");
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  PACKET_TRACE_LOG("StridedStrategy: Executing individual copy - size={}", payload_size);
  
  // Execute copy
  if (!CopyOperationHelper::safe_copy(dst_ptr, payload, payload_size, copy_kind_)) {
    HOLOSCAN_LOG_ERROR("StridedStrategy: Individual copy operation failed");
    return StateEvent::CORRUPTION_DETECTED;
  }
  
  // Update frame position
  state_machine.advance_frame_position(payload_size);
  
  return StateEvent::COPY_EXECUTED;
}

bool StateMachineStridedStrategy::validate_strided_copy_bounds(FrameProcessingStateMachine& state_machine) const {
  auto frame = state_machine.get_current_frame();
  if (!frame) {
    return false;
  }
  
  size_t total_copy_size = stride_info_.payload_size * accumulated_packet_count_;
  size_t current_pos = state_machine.get_frame_position();
  size_t frame_size = frame->get_size();
  
  return (current_pos + total_copy_size <= frame_size);
}

void StateMachineStridedStrategy::reset_accumulation(uint8_t* payload, size_t payload_size) {
  first_packet_ptr_ = payload;
  last_packet_ptr_ = payload;
  accumulated_packet_count_ = 1;
  accumulated_data_size_ = payload_size;
  stride_validated_ = false;
  actual_stride_ = 0;
}

// ========================================================================================
// CopyOperationHelper Implementation
// ========================================================================================

cudaMemcpyKind CopyOperationHelper::get_copy_kind(nvidia::gxf::MemoryStorageType src_storage_type,
                                                  nvidia::gxf::MemoryStorageType dst_storage_type) {
  if (src_storage_type == nvidia::gxf::MemoryStorageType::kHost) {
    return (dst_storage_type == nvidia::gxf::MemoryStorageType::kHost) 
        ? cudaMemcpyHostToHost 
        : cudaMemcpyHostToDevice;
  } else {
    return (dst_storage_type == nvidia::gxf::MemoryStorageType::kHost) 
        ? cudaMemcpyDeviceToHost 
        : cudaMemcpyDeviceToDevice;
  }
}

bool CopyOperationHelper::safe_copy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
  if (!dst || !src || size == 0) {
    HOLOSCAN_LOG_ERROR("CopyOperationHelper: Invalid copy parameters");
    return false;
  }
  
  try {
    CUDA_TRY(cudaMemcpy(dst, src, size, kind));
    return true;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("CopyOperationHelper: Copy failed - {}", e.what());
    return false;
  }
}

bool CopyOperationHelper::safe_copy_2d(void* dst, size_t dst_pitch, 
                                       const void* src, size_t src_pitch,
                                       size_t width, size_t height, 
                                       cudaMemcpyKind kind) {
  if (!dst || !src || width == 0 || height == 0) {
    HOLOSCAN_LOG_ERROR("CopyOperationHelper: Invalid 2D copy parameters");
    return false;
  }
  
  try {
    CUDA_TRY(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height, kind));
    return true;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("CopyOperationHelper: 2D copy failed - {}", e.what());
    return false;
  }
}

}  // namespace holoscan::ops 