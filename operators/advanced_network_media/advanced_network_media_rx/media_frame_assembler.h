/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_RX_MEDIA_FRAME_ASSEMBLER_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_RX_MEDIA_FRAME_ASSEMBLER_H_

#include <memory>
#include <functional>
#include "frame_assembly_controller.h"
#include "memory_copy_strategies.h"
#include "advanced_network/common.h"
#include "../common/frame_buffer.h"

namespace holoscan::ops {

// Forward declarations for public interfaces
class IFrameProvider;

// Forward declarations for detail namespace types used internally
namespace detail {
enum class StateEvent;
struct StateTransitionResult;
class FrameAssemblyController;
class StrategyDetector;
class IMemoryCopyStrategy;
}  // namespace detail

// Import detail types for cleaner private method signatures
using detail::FrameAssemblyController;
using detail::IMemoryCopyStrategy;
using detail::StateEvent;
using detail::StateTransitionResult;
using detail::StrategyDetector;

/**
 * @brief Configuration for strategy detection and memory settings
 */
struct ConverterConfiguration {
  // Memory configuration
  nvidia::gxf::MemoryStorageType source_memory_type = nvidia::gxf::MemoryStorageType::kDevice;
  nvidia::gxf::MemoryStorageType destination_memory_type = nvidia::gxf::MemoryStorageType::kDevice;

  // Burst configuration
  size_t header_stride_size = 0;
  size_t payload_stride_size = 0;
  bool hds_enabled = false;

  // Detection configuration
  bool force_contiguous_strategy = false;
  bool enable_strategy_detection = true;
};

/**
 * @brief Callback interface for frame completion events
 */
class IFrameCompletionHandler {
 public:
  virtual ~IFrameCompletionHandler() = default;

  /**
   * @brief Called when a frame is completed and ready for emission
   * @param frame Completed frame buffer
   */
  virtual void on_frame_completed(std::shared_ptr<FrameBufferBase> frame) = 0;

  /**
   * @brief Called when frame processing encounters an error
   * @param error_message Error description
   */
  virtual void on_frame_error(const std::string& error_message) = 0;
};

/**
 * @brief State machine based packets to frames converter
 *
 * This class provides a clean, state machine driven approach to converting
 * network packets into video frames with automatic strategy detection and
 * robust error handling.
 */
class MediaFrameAssembler {
 public:
  /**
   * @brief Constructor
   * @param frame_provider Provider for frame allocation
   * @param config Converter configuration
   */
  MediaFrameAssembler(std::shared_ptr<IFrameProvider> frame_provider,
                      const ConverterConfiguration& config = {});

  /**
   * @brief Set frame completion handler
   * @param handler Callback handler for frame events
   */
  void set_completion_handler(std::shared_ptr<IFrameCompletionHandler> handler);

  /**
   * @brief Configure burst parameters for strategy detection
   * @param header_stride_size Header stride from burst info
   * @param payload_stride_size Payload stride from burst info
   * @param hds_enabled Whether header data split is enabled
   */
  void configure_burst_parameters(size_t header_stride_size, size_t payload_stride_size,
                                  bool hds_enabled);

  /**
   * @brief Update memory configuration
   * @param source_type Source memory storage type
   * @param destination_type Destination memory storage type
   */
  void configure_memory_types(nvidia::gxf::MemoryStorageType source_type,
                              nvidia::gxf::MemoryStorageType destination_type);

  /**
   * @brief Process incoming RTP packet - MAIN ENTRY POINT
   * @param rtp_params Parsed RTP parameters
   * @param payload Packet payload data
   */
  void process_incoming_packet(const RtpParams& rtp_params, uint8_t* payload);

  /**
   * @brief Force strategy redetection (for testing or config changes)
   */
  void force_strategy_redetection();

  /**
   * @brief Reset converter to initial state
   */
  void reset();

  /**
   * @brief Get current converter statistics
   * @return Statistics structure
   */
  struct Statistics {
    size_t packets_processed = 0;
    size_t frames_completed = 0;
    size_t errors_recovered = 0;
    size_t strategy_redetections = 0;
    std::string current_strategy = "UNKNOWN";
    std::string current_frame_state = "IDLE";
    std::string last_error;
  };

  Statistics get_statistics() const;

  /**
   * @brief Check if converter has pending operations
   * @return True if copy operations are pending
   */
  bool has_pending_operations() const;

  /**
   * @brief Get current frame for external operations (debugging)
   * @return Current frame buffer or nullptr
   */
  std::shared_ptr<FrameBufferBase> get_current_frame() const;

  /**
   * @brief Get current frame position (debugging)
   * @return Current byte position in frame
   */
  size_t get_frame_position() const;

 private:
  /**
   * @brief Determine state machine event from packet parameters
   * @param rtp_params RTP packet parameters
   * @param payload Packet payload
   * @return Appropriate state event
   */
  StateEvent determine_event(const RtpParams& rtp_params, uint8_t* payload);

  /**
   * @brief Execute actions based on state machine transition result
   * @param result State transition result
   * @param rtp_params RTP packet parameters
   * @param payload Packet payload
   */
  void execute_actions(const StateTransitionResult& result, const RtpParams& rtp_params,
                       uint8_t* payload);

  /**
   * @brief Handle strategy detection and setup
   * @param rtp_params RTP packet parameters
   * @param payload Packet payload
   * @return True if strategy is ready for processing
   */
  bool handle_strategy_detection(const RtpParams& rtp_params, uint8_t* payload);

  /**
   * @brief Set up strategy once detection is complete
   * @param strategy Detected strategy
   */
  void setup_strategy(std::unique_ptr<IMemoryCopyStrategy> strategy);

  /**
   * @brief Validate packet integrity
   * @param rtp_params RTP packet parameters
   * @return True if packet is valid
   */
  bool validate_packet_integrity(const RtpParams& rtp_params);

  /**
   * @brief Handle frame completion processing
   */
  void handle_frame_completion();

  /**
   * @brief Handle error recovery
   * @param error_message Error description
   */
  void handle_error_recovery(const std::string& error_message);

  /**
   * @brief Update statistics
   * @param event State event that occurred
   */
  void update_statistics(StateEvent event);

 private:
  // Core components
  std::unique_ptr<FrameAssemblyController> state_machine_;
  std::unique_ptr<StrategyDetector> strategy_detector_;
  std::unique_ptr<IMemoryCopyStrategy> current_strategy_;

  // Configuration
  ConverterConfiguration config_;

  // Callback handlers
  std::shared_ptr<IFrameCompletionHandler> completion_handler_;

  // Statistics
  mutable Statistics statistics_;

  // State tracking
  bool strategy_detection_active_ = false;
};

/**
 * @brief Default frame completion handler that can be used with the converter
 */
class DefaultFrameCompletionHandler : public IFrameCompletionHandler {
 public:
  /**
   * @brief Constructor
   * @param frame_ready_callback Callback for completed frames
   * @param error_callback Callback for errors
   */
  DefaultFrameCompletionHandler(
      std::function<void(std::shared_ptr<FrameBufferBase>)> frame_ready_callback,
      std::function<void(const std::string&)> error_callback = nullptr);

  // IFrameCompletionHandler interface
  void on_frame_completed(std::shared_ptr<FrameBufferBase> frame) override;
  void on_frame_error(const std::string& error_message) override;

 private:
  std::function<void(std::shared_ptr<FrameBufferBase>)> frame_ready_callback_;
  std::function<void(const std::string&)> error_callback_;
};

/**
 * @brief Utility functions for converter configuration
 */
class ConverterConfigurationHelper {
 public:
  /**
   * @brief Create configuration from burst parameters
   * @param header_stride Header stride size
   * @param payload_stride Payload stride size
   * @param hds_enabled HDS setting
   * @param payload_on_cpu Whether payload is in CPU memory
   * @param frames_on_host Whether frames should be in host memory
   * @return Converter configuration
   */
  static ConverterConfiguration create_from_burst_config(size_t header_stride,
                                                         size_t payload_stride, bool hds_enabled,
                                                         bool payload_on_cpu, bool frames_on_host);

  /**
   * @brief Create configuration for testing with forced strategy
   * @param force_contiguous Whether to force contiguous strategy
   * @return Test configuration
   */
  static ConverterConfiguration create_test_config(bool force_contiguous = true);

  /**
   * @brief Validate configuration parameters
   * @param config Configuration to validate
   * @return True if configuration is valid
   */
  static bool validate_configuration(const ConverterConfiguration& config);
};

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_RX_MEDIA_FRAME_ASSEMBLER_H_