/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "network_burst_processor.h"
#include "../common/adv_network_media_common.h"
#include "advanced_network/common.h"

namespace holoscan::ops {

NetworkBurstProcessor::NetworkBurstProcessor(
    std::shared_ptr<MediaFrameAssembler> assembler)
    : assembler_(assembler) {
  if (!assembler_) {
    throw std::invalid_argument("MediaFrameAssembler cannot be null");
  }
}

void NetworkBurstProcessor::process_burst(BurstParams* burst, bool hds_enabled) {
  if (!burst || burst->hdr.hdr.num_pkts == 0) {
    return;
  }

  // Configure assembler with burst parameters on first burst
  configure_assembler_from_burst(burst);

  // Process all packets in the burst through frame assembler
  process_packets_in_burst(burst, hds_enabled);
}

void NetworkBurstProcessor::configure_assembler_from_burst(BurstParams* burst) {
  // Access burst extended info from custom_burst_data
  const auto* burst_info =
      reinterpret_cast<const AnoBurstExtendedInfo*>(&(burst->hdr.custom_burst_data));

  if (!configuration_initialized_) {
    // Configure assembler with burst parameters
    assembler_->configure_burst_parameters(
        burst_info->header_stride_size, burst_info->payload_stride_size, burst_info->hds_on);

    // Configure memory types based on burst info
    nvidia::gxf::MemoryStorageType src_type = burst_info->payload_on_cpu
                                                  ? nvidia::gxf::MemoryStorageType::kHost
                                                  : nvidia::gxf::MemoryStorageType::kDevice;

    // Destination type is determined by frame allocation in the operator
    nvidia::gxf::MemoryStorageType dst_type = nvidia::gxf::MemoryStorageType::kDevice;

    assembler_->configure_memory_types(src_type, dst_type);

    configuration_initialized_ = true;

    HOLOSCAN_LOG_INFO(
        "Refactored burst processor configured: header_stride={}, payload_stride={}, "
        "hds_on={}, payload_on_cpu={}",
        burst_info->header_stride_size,
        burst_info->payload_stride_size,
        burst_info->hds_on,
        burst_info->payload_on_cpu);
  }
}

void NetworkBurstProcessor::process_packets_in_burst(BurstParams* burst, bool hds_enabled) {
  // Process each packet through the frame assembler
  for (size_t i = 0; i < burst->hdr.hdr.num_pkts; ++i) {
    RtpParams rtp_params;
    uint8_t* payload = extract_packet_data(burst, i, hds_enabled, rtp_params);

    if (payload) {
      PACKET_TRACE_LOG("About to process packet {}/{}: seq={}, m_bit={}, size={}, payload_ptr={}",
                       i + 1,
                       burst->hdr.hdr.num_pkts,
                       rtp_params.sequence_number,
                       rtp_params.m_bit,
                       rtp_params.payload_size,
                       static_cast<void*>(payload));

      assembler_->process_incoming_packet(rtp_params, payload);

      PACKET_TRACE_LOG("Processed packet {}/{}: seq={}, m_bit={}, size={}",
                       i + 1,
                       burst->hdr.hdr.num_pkts,
                       rtp_params.sequence_number,
                       rtp_params.m_bit,
                       rtp_params.payload_size);
    } else {
      HOLOSCAN_LOG_WARN("Failed to extract payload from packet {}", i);
    }
  }
}

uint8_t* NetworkBurstProcessor::extract_packet_data(BurstParams* burst, size_t packet_index,
                                                    bool hds_enabled, RtpParams& rtp_params) {
  if (packet_index >= burst->hdr.hdr.num_pkts) {
    HOLOSCAN_LOG_ERROR(
        "Packet index {} out of range (max: {})", packet_index, burst->hdr.hdr.num_pkts);
    return nullptr;
  }

  if (hds_enabled) {
    // Header-Data Split mode: headers on CPU, payloads on GPU
    uint8_t* header_ptr = reinterpret_cast<uint8_t*>(burst->pkts[CPU_PKTS][packet_index]);
    uint8_t* payload_ptr = reinterpret_cast<uint8_t*>(burst->pkts[GPU_PKTS][packet_index]);

    if (!header_ptr || !payload_ptr) {
      HOLOSCAN_LOG_ERROR("Null pointer in HDS packet {}: header={}, payload={}",
                         packet_index,
                         static_cast<void*>(header_ptr),
                         static_cast<void*>(payload_ptr));
      return nullptr;
    }

    // Parse RTP header from CPU memory
    if (!parse_rtp_header(header_ptr, rtp_params)) {
      HOLOSCAN_LOG_ERROR("Failed to parse RTP header for packet {}", packet_index);
      return nullptr;
    }

    PACKET_TRACE_LOG("HDS packet {}: header_ptr={}, payload_ptr={}, seq={}",
                     packet_index,
                     static_cast<void*>(header_ptr),
                     static_cast<void*>(payload_ptr),
                     rtp_params.sequence_number);

    return payload_ptr;

  } else {
    // Standard mode: complete packet on CPU
    uint8_t* packet_ptr = reinterpret_cast<uint8_t*>(burst->pkts[CPU_PKTS][packet_index]);

    if (!packet_ptr) {
      HOLOSCAN_LOG_ERROR("Null packet pointer for packet {}", packet_index);
      return nullptr;
    }

    // Parse RTP header from beginning of packet
    if (!parse_rtp_header(packet_ptr, rtp_params)) {
      HOLOSCAN_LOG_ERROR("Failed to parse RTP header for packet {}", packet_index);
      return nullptr;
    }

    // Payload starts after RTP header
    uint8_t* payload_ptr = packet_ptr + RTP_SINGLE_SRD_HEADER_SIZE;

    PACKET_TRACE_LOG("Standard packet {}: packet_ptr={}, payload_ptr={}, seq={}",
                     packet_index,
                     static_cast<void*>(packet_ptr),
                     static_cast<void*>(payload_ptr),
                     rtp_params.sequence_number);

    return payload_ptr;
  }
}

}  // namespace holoscan::ops