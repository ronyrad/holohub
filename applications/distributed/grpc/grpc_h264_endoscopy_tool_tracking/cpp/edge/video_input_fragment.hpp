/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_EDGE_VIDEO_INPUT_FRAGMENT_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_EDGE_VIDEO_INPUT_FRAGMENT_HPP

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/gxf_codelet/gxf_codelet.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <grpc_client.hpp>

#include "../gxf_imports.hpp"
#include "video_input_fragment.hpp"
#include "viz_fragment.hpp"

namespace holohub::grpc_h264_endoscopy_tool_tracking {
using namespace holoscan;
using namespace holoscan::ops;

/**
 * @class VideoInputFragment
 * @brief A fragment class for handling video input and processing.
 *
 * This class is responsible for reading video bitstreams, sending requests to a gRPC client for
 * inference, and processing. The video bitstreams also decodes video frames for the Holoviz to
 * display the frames.
 */
class VideoInputFragment : public holoscan::Fragment {
 public:
  explicit VideoInputFragment(const std::string& datapath, const uint32_t width,
                              const uint32_t height)
      : datapath_(datapath), width_(width), height_(height) {}

  ~VideoInputFragment() { entity_client_service_->stop_entity_stream(); }

  void compose() override {
    condition_ = make_condition<AsynchronousCondition>("response_available_condition");
    request_queue_ =
        make_resource<ConditionVariableQueue<std::shared_ptr<EntityRequest>>>("request_queue");
    response_queue_ =
        make_resource<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>>(
            "response_queue", condition_);

    auto bitstream_reader = make_operator<VideoReadBitstreamOp>(
        "bitstream_reader",
        from_config("bitstream_reader"),
        Arg("input_file_path", datapath_ + "/surgical_video.264"),
        make_condition<CountCondition>(750),
        make_condition<PeriodicCondition>("periodic-condition",
                                          Arg("recess_period") = std::string("25hz")),
        Arg("pool") = make_resource<RMMAllocator>(
            "pool", Arg("device_memory_max_size") = std::string("256MB")));

    // The GrpcClientRequestOp is responsible for sending data to the gRPC server.
    auto outgoing_requests = make_operator<GrpcClientRequestOp>(
        "outgoing_requests",
        Arg("request_queue") = request_queue_,
        Arg("allocator") = make_resource<RMMAllocator>(
            "pool", Arg("device_memory_max_size") = std::string("256MB")));

    auto response_condition = make_condition<AsynchronousCondition>("response_condition");
    auto video_decoder_context = make_resource<VideoDecoderContext>(
        "decoder-context", Arg("async_scheduling_term") = response_condition);

    auto request_condition = make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request =
        make_operator<VideoDecoderRequestOp>("video_decoder_request",
                                             from_config("video_decoder_request"),
                                             Arg("async_scheduling_term") = request_condition,
                                             Arg("videodecoder_context") = video_decoder_context);

    auto video_decoder_response = make_operator<VideoDecoderResponseOp>(
        "video_decoder_response",
        from_config("video_decoder_response"),
        Arg("pool") = make_resource<RMMAllocator>(
            "pool", Arg("device_memory_max_size") = std::string("256MB")),
        Arg("videodecoder_context") = video_decoder_context);

    auto decoder_output_format_converter = make_operator<FormatConverterOp>(
        "decoder_output_format_converter",
        from_config("decoder_output_format_converter"),
        Arg("pool") = make_resource<RMMAllocator>(
            "pool", Arg("device_memory_max_size") = std::string("256MB")));

    // The GrpcClientResponseOp is responsible for handling incoming responses from the gRPC server.
    auto incoming_responses =
        make_operator<GrpcClientResponseOp>("incoming_responses",
                                            Arg("condition") = condition_,
                                            Arg("response_queue") = response_queue_);

    // Send the frames to the gRPC server for processing.
    add_flow(bitstream_reader, outgoing_requests, {{"output_transmitter", "input"}});

    add_flow(bitstream_reader, video_decoder_request, {{"output_transmitter", "input_frame"}});
    add_flow(video_decoder_response,
             decoder_output_format_converter,
             {{"output_transmitter", "source_video"}});

    // Here we add the operator to process the response queue with data received from the gRPC
    // server. The operator will convert the data to a GXF Entity and send it to the Holoviz.
    add_operator(incoming_responses);

    entity_client_service_ = std::make_shared<EntityClientService>(
        from_config("grpc_client.server_address").as<std::string>(),
        from_config("grpc_client.rpc_timeout").as<uint32_t>(),
        from_config("grpc_client.interrupt").as<bool>(),
        request_queue_,
        response_queue_,
        outgoing_requests);
    entity_client_service_->start_entity_stream();
  }

 private:
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue_;
  std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>> response_queue_;
  std::shared_ptr<AsynchronousCondition> condition_;
  std::shared_ptr<EntityClientService> entity_client_service_;
  std::string datapath_;
  uint32_t width_;
  uint32_t height_;
};

}  // namespace holohub::grpc_h264_endoscopy_tool_tracking
#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_EDGE_VIDEO_INPUT_FRAGMENT_HPP */
