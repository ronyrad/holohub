/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "frame_assembly_controller.h"
#include "memory_copy_strategies.h"
#include "../common/adv_network_media_common.h"

namespace holoscan::ops {
namespace detail {

// Import public interfaces for cleaner usage
using holoscan::ops::IFrameProvider;

// ========================================================================================
// FrameAssemblyController Implementation
// ========================================================================================

FrameAssemblyController::FrameAssemblyController(std::shared_ptr<IFrameProvider> frame_provider)
    : frame_provider_(frame_provider) {
  if (!frame_provider_) {
    throw std::invalid_argument("FrameProvider cannot be null");
  }

  // Initialize with a new frame
  allocate_new_frame();

  HOLOSCAN_LOG_DEBUG("FrameAssemblyController initialized");
}

StateTransitionResult FrameAssemblyController::process_event(StateEvent event,
                                                             const RtpParams* rtp_params,
                                                             uint8_t* payload) {
  // NOTE: rtp_params and payload are currently unused in state transition logic.
  // This assembly controller focuses purely on event-driven state transitions.
  // Parameters are kept for API consistency and future extensibility.
  (void)rtp_params;  // Suppress unused parameter warning
  (void)payload;     // Suppress unused parameter warning
  
  packets_processed_++;

  PACKET_TRACE_LOG("Processing event {} in state {}",
                   FrameAssemblyHelper::event_to_string(event),
                   FrameAssemblyHelper::state_to_string(context_.frame_state));

  StateTransitionResult result;

  // Route to appropriate state handler
  switch (context_.frame_state) {
    case FrameState::IDLE:
      result = handle_idle_state(event, rtp_params, payload);
      break;
    case FrameState::RECEIVING_PACKETS:
      result = handle_receiving_state(event, rtp_params, payload);
      break;
    case FrameState::COMPLETING_FRAME:
      result = handle_completing_state(event, rtp_params, payload);
      break;
    case FrameState::ERROR_RECOVERY:
      result = handle_error_recovery_state(event, rtp_params, payload);
      break;
    case FrameState::FRAME_READY:
      result = handle_frame_ready_state(event, rtp_params, payload);
      break;
    default:
      result = create_error_result("Unknown state");
      break;
  }

  // Update context if transition succeeded
  if (result.success && result.new_frame_state != context_.frame_state) {
    FrameState old_state = context_.frame_state;
    if (transition_to_state(result.new_frame_state)) {
      HOLOSCAN_LOG_DEBUG("State transition: {} -> {}",
                         FrameAssemblyHelper::state_to_string(old_state),
                         FrameAssemblyHelper::state_to_string(result.new_frame_state));
    } else {
      result = create_error_result("Invalid state transition");
    }
  }

  return result;
}

void FrameAssemblyController::reset() {
  context_.frame_state = FrameState::IDLE;
  context_.frame_position = 0;


  // Reset strategy if set
  if (strategy_) {
    strategy_->reset();
  }

  // Allocate new frame
  allocate_new_frame();

  HOLOSCAN_LOG_DEBUG("Assembly controller reset to initial state");
}

bool FrameAssemblyController::advance_frame_position(size_t bytes) {
  if (!validate_frame_bounds(bytes)) {
    HOLOSCAN_LOG_ERROR(
        "Frame position advancement would exceed bounds: current={}, bytes={}, frame_size={}",
        context_.frame_position,
        bytes,
        context_.current_frame ? context_.current_frame->get_size() : 0);
    return false;
  }

  context_.frame_position += bytes;

  PACKET_TRACE_LOG(
      "Frame position advanced by {} bytes to position {}", bytes, context_.frame_position);
  return true;
}

void FrameAssemblyController::set_strategy(std::shared_ptr<IMemoryCopyStrategy> strategy) {
  strategy_ = strategy;

  if (strategy_) {
    HOLOSCAN_LOG_DEBUG(
        "Strategy set: {}",
        strategy_->get_type() == CopyStrategy::CONTIGUOUS ? "CONTIGUOUS" : "STRIDED");
  } else {
    HOLOSCAN_LOG_DEBUG("Strategy cleared");
  }
}

bool FrameAssemblyController::allocate_new_frame() {
  context_.current_frame = frame_provider_->get_new_frame();
  context_.frame_position = 0;

  if (!context_.current_frame) {
    HOLOSCAN_LOG_ERROR("Frame allocation failed");
    return false;
  }

  PACKET_TRACE_LOG("New frame allocated: size={}", context_.current_frame->get_size());
  return true;
}

bool FrameAssemblyController::validate_frame_bounds(size_t required_bytes) const {
  if (!context_.current_frame) {
    return false;
  }

  return (context_.frame_position + required_bytes <= context_.current_frame->get_size());
}

bool FrameAssemblyController::transition_to_state(FrameState new_state) {
  if (!FrameAssemblyHelper::is_valid_transition(context_.frame_state, new_state)) {
    HOLOSCAN_LOG_ERROR("Invalid state transition: {} -> {}",
                       FrameAssemblyHelper::state_to_string(context_.frame_state),
                       FrameAssemblyHelper::state_to_string(new_state));
    return false;
  }

  context_.frame_state = new_state;
  return true;
}

StateTransitionResult FrameAssemblyController::handle_idle_state(StateEvent event,
                                                                 const RtpParams* rtp_params,
                                                                 uint8_t* payload) {
  switch (event) {
    case StateEvent::PACKET_ARRIVED:
    case StateEvent::STRATEGY_DETECTED:
      // Start receiving packets
      return create_success_result(FrameState::RECEIVING_PACKETS);

    case StateEvent::MARKER_DETECTED:
      // Single packet frame (edge case)
      return create_success_result(FrameState::COMPLETING_FRAME);

    case StateEvent::CORRUPTION_DETECTED:
      return create_success_result(FrameState::ERROR_RECOVERY);

    default:
      return create_error_result("Unexpected event in IDLE state");
  }
}

StateTransitionResult FrameAssemblyController::handle_receiving_state(StateEvent event,
                                                                      const RtpParams* rtp_params,
                                                                      uint8_t* payload) {
  switch (event) {
    case StateEvent::PACKET_ARRIVED:
      // Continue receiving packets
      return create_success_result(FrameState::RECEIVING_PACKETS);

    case StateEvent::MARKER_DETECTED: {
      // Frame completion triggered
      auto result = create_success_result(FrameState::COMPLETING_FRAME);
      result.should_complete_frame = true;
      return result;
    }

    case StateEvent::COPY_EXECUTED:
      // Copy operation completed, continue receiving
    
      return create_success_result(FrameState::RECEIVING_PACKETS);

    case StateEvent::CORRUPTION_DETECTED:
      return create_success_result(FrameState::ERROR_RECOVERY);

    default:
      return create_error_result("Unexpected event in RECEIVING_PACKETS state");
  }
}

StateTransitionResult FrameAssemblyController::handle_completing_state(StateEvent event,
                                                                       const RtpParams* rtp_params,
                                                                       uint8_t* payload) {
  switch (event) {
    case StateEvent::PACKET_ARRIVED:
      // Continue processing packets while completing frame
      PACKET_TRACE_LOG(
          "COMPLETING_FRAME: Processing additional packet, staying in completing state");
      return create_success_result(FrameState::COMPLETING_FRAME);

    case StateEvent::COPY_EXECUTED: {
      // Final copy completed, frame is ready
      auto result = create_success_result(FrameState::FRAME_READY);
      result.should_emit_frame = true;
      frames_completed_++;
      return result;
    }

    case StateEvent::FRAME_COMPLETED: {
      // Frame processing completed successfully
      auto result = create_success_result(FrameState::FRAME_READY);
      result.should_emit_frame = true;
      frames_completed_++;
      return result;
    }

    case StateEvent::MARKER_DETECTED:
      // Marker detected while completing - start new frame for next packet
      if (!allocate_new_frame()) {
        return create_error_result("Failed to allocate new frame for marker");
      }
      return create_success_result(FrameState::COMPLETING_FRAME);

    case StateEvent::CORRUPTION_DETECTED:
      return create_success_result(FrameState::ERROR_RECOVERY);

    default:
      return create_error_result("Unexpected event in COMPLETING_FRAME state");
  }
}

StateTransitionResult FrameAssemblyController::handle_error_recovery_state(
    StateEvent event, const RtpParams* rtp_params, uint8_t* payload) {
  switch (event) {
    case StateEvent::RECOVERY_MARKER:
    case StateEvent::MARKER_DETECTED:
      // Recovery marker received, start new frame
      error_recoveries_++;
      allocate_new_frame();
      return create_success_result(FrameState::IDLE);

    case StateEvent::PACKET_ARRIVED:
      // Stay in recovery state, waiting for marker
      return create_success_result(FrameState::ERROR_RECOVERY);

    case StateEvent::CORRUPTION_DETECTED:
      // Additional corruption detected, stay in recovery
      return create_success_result(FrameState::ERROR_RECOVERY);

    default:
      return create_error_result("Unexpected event in ERROR_RECOVERY state");
  }
}

StateTransitionResult FrameAssemblyController::handle_frame_ready_state(StateEvent event,
                                                                        const RtpParams* rtp_params,
                                                                        uint8_t* payload) {
  switch (event) {
    case StateEvent::FRAME_COMPLETED:
      // Frame emission completed, allocate new frame and return to idle
      if (!allocate_new_frame()) {
        return create_error_result("Failed to allocate new frame");
      }
      return create_success_result(FrameState::IDLE);

    case StateEvent::PACKET_ARRIVED:
      // New packet arrived while frame is ready - allocate new frame and start processing
      if (!allocate_new_frame()) {
        return create_error_result("Failed to allocate new frame");
      }
      return create_success_result(FrameState::RECEIVING_PACKETS);

    case StateEvent::MARKER_DETECTED:
      // New marker while frame is ready - handle as single packet frame
      if (!allocate_new_frame()) {
        return create_error_result("Failed to allocate new frame");
      }
      return create_success_result(FrameState::COMPLETING_FRAME);

    case StateEvent::CORRUPTION_DETECTED:
      // Corruption detected while frame is ready - go to error recovery
      return create_success_result(FrameState::ERROR_RECOVERY);

    default:
      return create_error_result("Unexpected event in FRAME_READY state");
  }
}

StateTransitionResult FrameAssemblyController::create_success_result(FrameState new_state) {
  StateTransitionResult result;
  result.success = true;
  result.new_frame_state = new_state;
  return result;
}

StateTransitionResult FrameAssemblyController::create_error_result(
    const std::string& error_message) {
  StateTransitionResult result;
  result.success = false;
  result.error_message = error_message;
  result.new_frame_state = FrameState::ERROR_RECOVERY;
  return result;
}

// ========================================================================================
// FrameAssemblyHelper Implementation
// ========================================================================================

std::string FrameAssemblyHelper::state_to_string(FrameState state) {
  switch (state) {
    case FrameState::IDLE:
      return "IDLE";
    case FrameState::RECEIVING_PACKETS:
      return "RECEIVING_PACKETS";
    case FrameState::COMPLETING_FRAME:
      return "COMPLETING_FRAME";
    case FrameState::ERROR_RECOVERY:
      return "ERROR_RECOVERY";
    case FrameState::FRAME_READY:
      return "FRAME_READY";
    default:
      return "UNKNOWN";
  }
}

std::string FrameAssemblyHelper::event_to_string(StateEvent event) {
  switch (event) {
    case StateEvent::PACKET_ARRIVED:
      return "PACKET_ARRIVED";
    case StateEvent::MARKER_DETECTED:
      return "MARKER_DETECTED";
    case StateEvent::COPY_EXECUTED:
      return "COPY_EXECUTED";
    case StateEvent::CORRUPTION_DETECTED:
      return "CORRUPTION_DETECTED";
    case StateEvent::RECOVERY_MARKER:
      return "RECOVERY_MARKER";
    case StateEvent::STRATEGY_DETECTED:
      return "STRATEGY_DETECTED";
    case StateEvent::FRAME_COMPLETED:
      return "FRAME_COMPLETED";
    default:
      return "UNKNOWN";
  }
}

bool FrameAssemblyHelper::is_valid_transition(FrameState from_state, FrameState to_state) {
  // Define valid state transitions
  switch (from_state) {
    case FrameState::IDLE:
      return (to_state == FrameState::RECEIVING_PACKETS ||
              to_state == FrameState::COMPLETING_FRAME || to_state == FrameState::ERROR_RECOVERY);

    case FrameState::RECEIVING_PACKETS:
      return (to_state == FrameState::RECEIVING_PACKETS ||
              to_state == FrameState::COMPLETING_FRAME || to_state == FrameState::ERROR_RECOVERY);

    case FrameState::COMPLETING_FRAME:
      return (to_state == FrameState::FRAME_READY || to_state == FrameState::ERROR_RECOVERY);

    case FrameState::ERROR_RECOVERY:
      return (to_state == FrameState::IDLE || to_state == FrameState::ERROR_RECOVERY);

    case FrameState::FRAME_READY:
      return (to_state == FrameState::IDLE || to_state == FrameState::RECEIVING_PACKETS ||
              to_state == FrameState::COMPLETING_FRAME || to_state == FrameState::ERROR_RECOVERY);

    default:
      return false;
  }
}

std::vector<StateEvent> FrameAssemblyHelper::get_valid_events(FrameState state) {
  switch (state) {
    case FrameState::IDLE:
      return {StateEvent::PACKET_ARRIVED,
              StateEvent::MARKER_DETECTED,
              StateEvent::STRATEGY_DETECTED,
              StateEvent::CORRUPTION_DETECTED};

    case FrameState::RECEIVING_PACKETS:
      return {StateEvent::PACKET_ARRIVED,
              StateEvent::MARKER_DETECTED,
              StateEvent::COPY_EXECUTED,
              StateEvent::CORRUPTION_DETECTED};

    case FrameState::COMPLETING_FRAME:
      return {
          StateEvent::COPY_EXECUTED, StateEvent::FRAME_COMPLETED, StateEvent::CORRUPTION_DETECTED};

    case FrameState::ERROR_RECOVERY:
      return {StateEvent::RECOVERY_MARKER,
              StateEvent::MARKER_DETECTED,
              StateEvent::PACKET_ARRIVED,
              StateEvent::CORRUPTION_DETECTED};

    case FrameState::FRAME_READY:
      return {StateEvent::FRAME_COMPLETED, StateEvent::PACKET_ARRIVED, StateEvent::MARKER_DETECTED};

    default:
      return {};
  }
}

}  // namespace detail
}  // namespace holoscan::ops