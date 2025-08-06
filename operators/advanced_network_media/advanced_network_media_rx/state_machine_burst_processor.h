/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_RX_STATE_MACHINE_BURST_PROCESSOR_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_RX_STATE_MACHINE_BURST_PROCESSOR_H_

#include <memory>
#include "state_machine_packets_to_frames_converter.h"
#include "advanced_network/common.h"
#include "advanced_network/managers/rivermax/rivermax_ano_data_types.h"

namespace holoscan::ops {

using namespace holoscan::advanced_network;

/**
 * @brief State machine burst processor that integrates with the state machine converter
 * 
 * This class handles burst-level operations and forwards individual packets
 * to the StateMachinePacketsToFramesConverter for state machine processing.
 */
class StateMachineBurstProcessor {
public:
  /**
   * @brief Constructor
   * @param converter The refactored converter with state machine
   */
  explicit StateMachineBurstProcessor(std::shared_ptr<StateMachinePacketsToFramesConverter> converter);

  /**
   * @brief Process a burst of packets
   * @param burst The burst containing packets to process
   * @param hds_enabled Whether header-data split is enabled
   */
  void process_burst(BurstParams* burst, bool hds_enabled);

private:
  /**
   * @brief Configure converter with burst parameters
   * @param burst The burst containing configuration info
   */
  void configure_converter_from_burst(BurstParams* burst);
  
  /**
   * @brief Process all packets in the burst
   * @param burst The burst containing packets
   * @param hds_enabled Whether HDS is enabled
   */
  void process_packets_in_burst(BurstParams* burst, bool hds_enabled);
  
  /**
   * @brief Extract RTP header and payload from packet
   * @param burst The burst containing packets
   * @param packet_index Index of packet in burst
   * @param hds_enabled Whether HDS is enabled
   * @param rtp_params Output RTP parameters
   * @return Pointer to packet payload
   */
  uint8_t* extract_packet_data(BurstParams* burst, 
                              size_t packet_index, 
                              bool hds_enabled,
                              RtpParams& rtp_params);

private:
  // Constants for packet array indexing
  static constexpr int CPU_PKTS = 0;
  static constexpr int GPU_PKTS = 1;

  std::shared_ptr<StateMachinePacketsToFramesConverter> converter_;
  bool configuration_initialized_ = false;
};

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_RX_STATE_MACHINE_BURST_PROCESSOR_H_