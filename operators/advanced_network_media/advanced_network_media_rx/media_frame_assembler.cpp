/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "media_frame_assembler.h"
#include "../common/adv_network_media_common.h"

namespace holoscan::ops {

// Import detail namespace classes for convenience
using detail::FrameAssemblyController;
using detail::StrategyFactory;
using detail::StrategyDetector;
using detail::IMemoryCopyStrategy;
using detail::StateEvent;
using detail::CopyStrategy;
using detail::FrameState;

// Helper functions to convert internal types to strings for statistics
std::string convert_strategy_to_string(CopyStrategy internal_strategy) {
  switch (internal_strategy) {
    case CopyStrategy::CONTIGUOUS:
      return "CONTIGUOUS";
    case CopyStrategy::STRIDED:
      return "STRIDED";
    default:
      return "UNKNOWN";
  }
}

std::string convert_state_to_string(FrameState internal_state) {
  switch (internal_state) {
    case FrameState::IDLE:
      return "IDLE";
    case FrameState::RECEIVING_PACKETS:
      return "RECEIVING_PACKETS";
    case FrameState::COMPLETING_FRAME:
      return "COMPLETING_FRAME";
    case FrameState::FRAME_READY:
      return "FRAME_READY";
    case FrameState::ERROR_RECOVERY:
      return "ERROR_RECOVERY";
    default:
      return "IDLE";
  }
}

// ========================================================================================
// MediaFrameAssembler Implementation
// ========================================================================================

MediaFrameAssembler::MediaFrameAssembler(std::shared_ptr<IFrameProvider> frame_provider,
                                         const ConverterConfiguration& config)
    : config_(config) {
  // Validate configuration
  if (!ConverterConfigurationHelper::validate_configuration(config_)) {
    throw std::invalid_argument("Invalid converter configuration");
  }

  // Create state machine
  state_machine_ = std::make_unique<FrameAssemblyController>(frame_provider);

  // Create strategy detector if needed
  if (config_.enable_strategy_detection && !config_.force_contiguous_strategy) {
    strategy_detector_ = StrategyFactory::create_detector();
    strategy_detection_active_ = true;
  } else if (config_.force_contiguous_strategy) {
    // Create contiguous strategy immediately
    current_strategy_ = StrategyFactory::create_contiguous_strategy(
        config_.source_memory_type, config_.destination_memory_type);
    setup_strategy(std::move(current_strategy_));
    strategy_detection_active_ = false;
  }

  HOLOSCAN_LOG_INFO("MediaFrameAssembler initialized: strategy_detection={}, force_contiguous={}",
                    config_.enable_strategy_detection,
                    config_.force_contiguous_strategy);
}

void MediaFrameAssembler::set_completion_handler(std::shared_ptr<IFrameCompletionHandler> handler) {
  completion_handler_ = handler;
}

void MediaFrameAssembler::configure_burst_parameters(size_t header_stride_size,
                                                     size_t payload_stride_size, bool hds_enabled) {
  config_.header_stride_size = header_stride_size;
  config_.payload_stride_size = payload_stride_size;
  config_.hds_enabled = hds_enabled;

  // Configure strategy detector if active
  if (strategy_detector_) {
    strategy_detector_->configure_burst_parameters(
        header_stride_size, payload_stride_size, hds_enabled);
  }

  HOLOSCAN_LOG_DEBUG("Burst parameters configured: header_stride={}, payload_stride={}, hds={}",
                     header_stride_size,
                     payload_stride_size,
                     hds_enabled);
}

void MediaFrameAssembler::configure_memory_types(nvidia::gxf::MemoryStorageType source_type,
                                                 nvidia::gxf::MemoryStorageType destination_type) {
  config_.source_memory_type = source_type;
  config_.destination_memory_type = destination_type;

  HOLOSCAN_LOG_DEBUG("Memory types configured: source={}, destination={}",
                     static_cast<int>(source_type),
                     static_cast<int>(destination_type));

  // If strategy is already set up, we may need to recreate it with new memory types
  if (current_strategy_ && !strategy_detection_active_) {
    CopyStrategy strategy_type = current_strategy_->get_type();

    if (strategy_type == CopyStrategy::CONTIGUOUS) {
      current_strategy_ =
          StrategyFactory::create_contiguous_strategy(source_type, destination_type);
    }
    // Note: For strided strategy, we would need the stride info, so we'd trigger redetection

    setup_strategy(std::move(current_strategy_));
  }
}

void MediaFrameAssembler::process_incoming_packet(const RtpParams& rtp_params, uint8_t* payload) {
  try {
    // Update statistics
    update_statistics(StateEvent::PACKET_ARRIVED);

    // Determine appropriate event for this packet
    StateEvent event = determine_event(rtp_params, payload);

    // Process event through state machine
    auto result = state_machine_->process_event(event, &rtp_params, payload);

    if (!result.success) {
      HOLOSCAN_LOG_ERROR("State machine processing failed: {}", result.error_message);
      handle_error_recovery(result.error_message);
      return;
    }

    // Execute actions based on state machine result
    execute_actions(result, rtp_params, payload);

    PACKET_TRACE_LOG("Packet processed successfully: seq={}, event={}, new_state={}",
                     rtp_params.sequence_number,
                     static_cast<int>(event),
                     static_cast<int>(result.new_frame_state));

  } catch (const std::exception& e) {
    std::string error_msg = std::string("Exception in packet processing: ") + e.what();
    HOLOSCAN_LOG_ERROR("{}", error_msg);
    handle_error_recovery(error_msg);
  }
}

void MediaFrameAssembler::force_strategy_redetection() {
  if (strategy_detector_) {
    strategy_detector_->reset();
    strategy_detection_active_ = true;
    current_strategy_.reset();
    state_machine_->set_strategy(nullptr);
    statistics_.strategy_redetections++;

    HOLOSCAN_LOG_INFO("Strategy redetection forced");
  }
}

void MediaFrameAssembler::reset() {
  state_machine_->reset();

  if (strategy_detector_) {
    strategy_detector_->reset();
    strategy_detection_active_ =
        config_.enable_strategy_detection && !config_.force_contiguous_strategy;
  }

  if (config_.force_contiguous_strategy) {
    current_strategy_ = StrategyFactory::create_contiguous_strategy(
        config_.source_memory_type, config_.destination_memory_type);
    setup_strategy(std::move(current_strategy_));
  } else {
    current_strategy_.reset();
  }

  // Reset statistics (keep cumulative counters)
  statistics_.current_strategy = "UNKNOWN";
  statistics_.current_frame_state = "IDLE";
  statistics_.last_error.clear();

  HOLOSCAN_LOG_INFO("Converter reset to initial state");
}

MediaFrameAssembler::Statistics MediaFrameAssembler::get_statistics() const {
  // Update current state information
  const auto& context = state_machine_->get_context();
  statistics_.current_frame_state = convert_state_to_string(context.frame_state);

  if (current_strategy_) {
    statistics_.current_strategy = convert_strategy_to_string(current_strategy_->get_type());
  }

  return statistics_;
}

bool MediaFrameAssembler::has_pending_operations() const {
  const auto& context = state_machine_->get_context();
  return context.has_pending_copy ||
         (current_strategy_ && current_strategy_->has_pending_operations());
}

std::shared_ptr<FrameBufferBase> MediaFrameAssembler::get_current_frame() const {
  return state_machine_->get_current_frame();
}

size_t MediaFrameAssembler::get_frame_position() const {
  return state_machine_->get_frame_position();
}

StateEvent MediaFrameAssembler::determine_event(const RtpParams& rtp_params, uint8_t* payload) {
  // Check for M-bit marker first
  if (rtp_params.m_bit) {
    const auto& context = state_machine_->get_context();
    if (context.frame_state == FrameState::ERROR_RECOVERY) {
      return StateEvent::RECOVERY_MARKER;
    } else {
      return StateEvent::MARKER_DETECTED;
    }
  }

  // Validate packet integrity
  if (!validate_packet_integrity(rtp_params)) {
    return StateEvent::CORRUPTION_DETECTED;
  }

  // Check if we're in strategy detection phase
  if (strategy_detection_active_ && strategy_detector_) {
    if (strategy_detector_->collect_packet(rtp_params, payload, rtp_params.payload_size)) {
      // Enough packets collected, attempt strategy detection
      auto detected_strategy = strategy_detector_->detect_strategy(config_.source_memory_type,
                                                                   config_.destination_memory_type);

      if (detected_strategy) {
        setup_strategy(std::move(detected_strategy));
        strategy_detection_active_ = false;
        return StateEvent::STRATEGY_DETECTED;
      } else {
        // Detection failed, will retry with more packets
        return StateEvent::PACKET_ARRIVED;
      }
    } else {
      // Still collecting packets for detection
      return StateEvent::PACKET_ARRIVED;
    }
  }

  return StateEvent::PACKET_ARRIVED;
}

void MediaFrameAssembler::execute_actions(const StateTransitionResult& result,
                                          const RtpParams& rtp_params, uint8_t* payload) {
  PACKET_TRACE_LOG("execute_actions: should_emit_frame={}, should_complete_frame={}, new_state={}",
                   result.should_emit_frame,
                   result.should_complete_frame,
                   static_cast<int>(result.new_frame_state));
  // Strategy processing
  if ((result.new_frame_state == FrameState::RECEIVING_PACKETS ||
       result.new_frame_state == FrameState::COMPLETING_FRAME) &&
      current_strategy_ && payload) {
    StateEvent strategy_result =
        current_strategy_->process_packet(*state_machine_, payload, rtp_params.payload_size);

    if (strategy_result == StateEvent::CORRUPTION_DETECTED) {
      handle_error_recovery("Strategy processing detected corruption");
      return;
    } else if (strategy_result == StateEvent::COPY_EXECUTED) {
      PACKET_TRACE_LOG("Strategy executed copy operation successfully");
    }
  }

  // Execute pending copies if requested
  if (result.should_execute_copy && current_strategy_) {
    if (current_strategy_->has_pending_operations()) {
      StateEvent copy_result = current_strategy_->execute_pending_copy(*state_machine_);
      if (copy_result == StateEvent::CORRUPTION_DETECTED) {
        handle_error_recovery("Copy execution failed");
        return;
      }
    }
  }

  // Handle frame completion
  if (result.should_complete_frame || result.new_frame_state == FrameState::COMPLETING_FRAME) {
    handle_frame_completion();
  }

  // Handle frame emission
  if (result.should_emit_frame) {
    auto frame = state_machine_->get_current_frame();
    if (frame && completion_handler_) {
      PACKET_TRACE_LOG("Emitting frame to completion handler");
      completion_handler_->on_frame_completed(frame);
      statistics_.frames_completed++;
    }

    // Signal frame emission complete to transition FRAME_READY -> IDLE
    PACKET_TRACE_LOG("Sending FRAME_COMPLETED event to transition FRAME_READY -> IDLE");
    auto emission_result = state_machine_->process_event(StateEvent::FRAME_COMPLETED);
    if (!emission_result.success) {
      HOLOSCAN_LOG_ERROR("Failed to complete frame emission: {}", emission_result.error_message);
    } else {
      PACKET_TRACE_LOG("Successfully transitioned to IDLE state");
    }
  }
}

bool MediaFrameAssembler::handle_strategy_detection(const RtpParams& rtp_params, uint8_t* payload) {
  if (!strategy_detection_active_ || !strategy_detector_) {
    return true;  // No detection needed or strategy already available
  }

  // Collect packet for analysis
  if (strategy_detector_->collect_packet(rtp_params, payload, rtp_params.payload_size)) {
    // Attempt strategy detection
    auto detected_strategy = strategy_detector_->detect_strategy(config_.source_memory_type,
                                                                 config_.destination_memory_type);

    if (detected_strategy) {
      setup_strategy(std::move(detected_strategy));
      strategy_detection_active_ = false;
      return true;
    } else {
      HOLOSCAN_LOG_DEBUG("Strategy detection failed, will retry");
      return false;
    }
  }

  PACKET_TRACE_LOG("Still collecting packets for strategy detection ({}/{})",
                   strategy_detector_->get_packets_analyzed(),
                   StrategyDetector::DETECTION_PACKET_COUNT);
  return false;
}

void MediaFrameAssembler::setup_strategy(std::unique_ptr<IMemoryCopyStrategy> strategy) {
  current_strategy_ = std::move(strategy);

  // Note: For the old interface compatibility, we would set the strategy in the state machine
  // but since IMemoryCopyStrategy is different from IPacketCopyStrategy, we manage it here

  if (current_strategy_) {
    HOLOSCAN_LOG_INFO(
        "Strategy setup completed: {}",
        current_strategy_->get_type() == CopyStrategy::CONTIGUOUS ? "CONTIGUOUS" : "STRIDED");
  }
}

bool MediaFrameAssembler::validate_packet_integrity(const RtpParams& rtp_params) {
  auto frame = state_machine_->get_current_frame();
  if (!frame) {
    return false;
  }

  int64_t bytes_left = frame->get_size() - state_machine_->get_frame_position();

  if (bytes_left < 0) {
    return false;  // Frame overflow
  }

  bool frame_full = (bytes_left == 0);
  if (frame_full && !rtp_params.m_bit) {
    return false;  // Frame full but no marker
  }

  return true;
}

void MediaFrameAssembler::handle_frame_completion() {
  // Execute any pending copy operations
  if (current_strategy_ && current_strategy_->has_pending_operations()) {
    StateEvent copy_result = current_strategy_->execute_pending_copy(*state_machine_);
    if (copy_result == StateEvent::CORRUPTION_DETECTED) {
      handle_error_recovery("Final copy operation failed");
      return;
    }
  }

  // Signal frame completion to state machine
  auto result = state_machine_->process_event(StateEvent::FRAME_COMPLETED);

  if (!result.success) {
    handle_error_recovery("Frame completion validation failed");
    return;
  }

  // Handle frame emission if requested by state machine
  if (result.should_emit_frame) {
    auto frame = state_machine_->get_current_frame();
    if (frame && completion_handler_) {
      PACKET_TRACE_LOG("Emitting frame to completion handler from handle_frame_completion");
      completion_handler_->on_frame_completed(frame);
      statistics_.frames_completed++;
    }

    // Signal frame emission complete to transition FRAME_READY -> IDLE
    PACKET_TRACE_LOG(
        "Sending FRAME_COMPLETED event to transition FRAME_READY -> IDLE from "
        "handle_frame_completion");
    auto emission_result = state_machine_->process_event(StateEvent::FRAME_COMPLETED);
    if (!emission_result.success) {
      HOLOSCAN_LOG_ERROR("Failed to complete frame emission in handle_frame_completion: {}",
                         emission_result.error_message);
    } else {
      PACKET_TRACE_LOG("Successfully transitioned to IDLE state from handle_frame_completion");
    }
  }
}

void MediaFrameAssembler::handle_error_recovery(const std::string& error_message) {
  statistics_.last_error = error_message;
  statistics_.errors_recovered++;

  if (completion_handler_) {
    completion_handler_->on_frame_error(error_message);
  }

  // Reset strategy if needed
  if (current_strategy_) {
    current_strategy_->reset();
  }

  HOLOSCAN_LOG_ERROR("Error recovery initiated: {}", error_message);
}

void MediaFrameAssembler::update_statistics(StateEvent event) {
  switch (event) {
    case StateEvent::PACKET_ARRIVED:
      statistics_.packets_processed++;
      break;
    case StateEvent::STRATEGY_DETECTED:
      statistics_.strategy_redetections++;
      break;
    default:
      break;
  }
}

// ========================================================================================
// DefaultFrameCompletionHandler Implementation
// ========================================================================================

DefaultFrameCompletionHandler::DefaultFrameCompletionHandler(
    std::function<void(std::shared_ptr<FrameBufferBase>)> frame_ready_callback,
    std::function<void(const std::string&)> error_callback)
    : frame_ready_callback_(frame_ready_callback), error_callback_(error_callback) {}

void DefaultFrameCompletionHandler::on_frame_completed(std::shared_ptr<FrameBufferBase> frame) {
  if (frame_ready_callback_) {
    frame_ready_callback_(frame);
  }
}

void DefaultFrameCompletionHandler::on_frame_error(const std::string& error_message) {
  if (error_callback_) {
    error_callback_(error_message);
  } else {
    HOLOSCAN_LOG_ERROR("Frame processing error: {}", error_message);
  }
}

// ========================================================================================
// ConverterConfigurationHelper Implementation
// ========================================================================================

ConverterConfiguration ConverterConfigurationHelper::create_from_burst_config(size_t header_stride,
                                                                              size_t payload_stride,
                                                                              bool hds_enabled,
                                                                              bool payload_on_cpu,
                                                                              bool frames_on_host) {
  ConverterConfiguration config;

  config.header_stride_size = header_stride;
  config.payload_stride_size = payload_stride;
  config.hds_enabled = hds_enabled;

  config.source_memory_type = payload_on_cpu ? nvidia::gxf::MemoryStorageType::kHost
                                             : nvidia::gxf::MemoryStorageType::kDevice;

  config.destination_memory_type = frames_on_host ? nvidia::gxf::MemoryStorageType::kHost
                                                  : nvidia::gxf::MemoryStorageType::kDevice;

  config.enable_strategy_detection = true;
  config.force_contiguous_strategy = false;

  return config;
}

ConverterConfiguration ConverterConfigurationHelper::create_test_config(bool force_contiguous) {
  ConverterConfiguration config;

  config.source_memory_type = nvidia::gxf::MemoryStorageType::kHost;
  config.destination_memory_type = nvidia::gxf::MemoryStorageType::kHost;
  config.hds_enabled = false;
  config.header_stride_size = 1500;
  config.payload_stride_size = 1500;
  config.force_contiguous_strategy = force_contiguous;
  config.enable_strategy_detection = !force_contiguous;

  return config;
}

bool ConverterConfigurationHelper::validate_configuration(const ConverterConfiguration& config) {
  // Basic validation
  if (config.enable_strategy_detection && config.force_contiguous_strategy) {
    HOLOSCAN_LOG_ERROR(
        "Configuration error: Cannot enable strategy detection and force contiguous strategy "
        "simultaneously");
    return false;
  }

  // Stride validation (if detection is enabled)
  if (config.enable_strategy_detection) {
    if (config.header_stride_size == 0 && config.payload_stride_size == 0) {
      HOLOSCAN_LOG_WARN(
          "Zero stride sizes with strategy detection enabled may affect detection accuracy");
    }
  }

  return true;
}

}  // namespace holoscan::ops