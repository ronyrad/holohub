#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from holoscan.schedulers import MultiThreadScheduler

from holohub.advanced_network_common import _advanced_network_common as adv_network_common
from holohub.advanced_network_media_rx import _advanced_network_media_rx as adv_network_media_rx
from holohub.advanced_network_media_tx import _advanced_network_media_tx as adv_network_media_tx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def check_rx_tx_enabled(app, require_rx=True, require_tx=False):
    """
    Check if RX and TX are enabled in the advanced network configuration.

    Args:
        app: The Holoscan Application instance
        require_rx: Whether RX must be enabled (default: True)
        require_tx: Whether TX must be enabled (default: False)

    Returns:
        tuple: (rx_enabled, tx_enabled)

    Raises:
        SystemExit: If required functionality is not enabled
    """
    try:
        # Manual parsing of the advanced_network config
        adv_net_config_dict = app.kwargs("advanced_network")

        rx_enabled = False
        tx_enabled = False

        # Check if there are interfaces with RX/TX configurations
        if "cfg" in adv_net_config_dict and "interfaces" in adv_net_config_dict["cfg"]:
            for interface in adv_net_config_dict["cfg"]["interfaces"]:
                if "rx" in interface:
                    rx_enabled = True
                if "tx" in interface:
                    tx_enabled = True

        logger.info(f"RX enabled: {rx_enabled}, TX enabled: {tx_enabled}")

        if require_rx and not rx_enabled:
            logger.error("RX is not enabled. Please enable RX in the config file.")
            sys.exit(1)

        if require_tx and not tx_enabled:
            logger.error("TX is not enabled. Please enable TX in the config file.")
            sys.exit(1)

        return rx_enabled, tx_enabled

    except Exception as e:
        logger.warning(f"Could not check RX/TX status from advanced_network config: {e}")
        # Fallback: check if we have the required operator configs
        try:
            if require_rx:
                app.from_config("advanced_network_media_rx")
                logger.info("RX is enabled (found advanced_network_media_rx config)")
            if require_tx:
                app.from_config("advanced_network_media_tx")
                logger.info("TX is enabled (found advanced_network_media_tx config)")
            return require_rx, require_tx
        except Exception as e2:
            if require_rx:
                logger.error("RX is not enabled. Please enable RX in the config file.")
                logger.error(f"Could not find advanced_network_media_rx configuration: {e2}")
                sys.exit(1)
            if require_tx:
                logger.error("TX is not enabled. Please enable TX in the config file.")
                logger.error(f"Could not find advanced_network_media_tx configuration: {e2}")
                sys.exit(1)
            return False, False


class FormatInferenceInputOp(Operator):
    """Operator to format input image for inference"""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Get input tensor - should work reliably now with tensor format
        tensor = cp.asarray(in_message.get("preprocessed"))

        # Transpose
        tensor = cp.moveaxis(tensor, 2, 0)[cp.newaxis]
        # Copy as a contiguous array to avoid issue with strides
        tensor = cp.ascontiguousarray(tensor)

        # Create output message
        op_output.emit(dict(preprocessed=tensor), "out")


class PostprocessorOp(Operator):
    """Operator to post-process inference output:
    * Non-max suppression
    * Make boxes compatible with Holoviz

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Output tensor names
        self.outputs = [
            "boxes",
            "noses",
            "left_eyes",
            "right_eyes",
            "left_ears",
            "right_ears",
            "left_shoulders",
            "right_shoulders",
            "left_elbows",
            "right_elbows",
            "left_wrists",
            "right_wrists",
            "left_hips",
            "right_hips",
            "left_knees",
            "right_knees",
            "left_ankles",
            "right_ankles",
            "segments",
        ]

        # Indices for each keypoint as defined by YOLOv8 pose model
        self.NOSE = slice(5, 7)
        self.LEFT_EYE = slice(8, 10)
        self.RIGHT_EYE = slice(11, 13)
        self.LEFT_EAR = slice(14, 16)
        self.RIGHT_EAR = slice(17, 19)
        self.LEFT_SHOULDER = slice(20, 22)
        self.RIGHT_SHOULDER = slice(23, 25)
        self.LEFT_ELBOW = slice(26, 28)
        self.RIGHT_ELBOW = slice(29, 31)
        self.LEFT_WRIST = slice(32, 34)
        self.RIGHT_WRIST = slice(35, 37)
        self.LEFT_HIP = slice(38, 40)
        self.RIGHT_HIP = slice(41, 43)
        self.LEFT_KNEE = slice(44, 46)
        self.RIGHT_KNEE = slice(47, 49)
        self.LEFT_ANKLE = slice(50, 52)
        self.RIGHT_ANKLE = slice(53, 55)

    def setup(self, spec: OperatorSpec):
        """
        input: "in"    - Input tensors coming from output of inference model
        output: "out"  - The post-processed output after applying thresholding and non-max suppression.
                         Outputs are the boxes, keypoints, and segments.  See self.outputs for the list of outputs.
        params:
            iou_threshold:    Intersection over Union (IoU) threshold for non-max suppression (default: 0.5)
            score_threshold:  Score threshold for filtering out low scores (default: 0.5)
            image_dim:        Image dimensions for normalizing the boxes (default: None)

        Returns:
            None
        """
        spec.input("in")
        spec.output("out")
        spec.param("iou_threshold", 0.5)
        spec.param("score_threshold", 0.5)
        spec.param("image_dim", None)

    def get_keypoints(self, detection):
        # Keypoints to be returned including our own "neck" keypoint
        keypoints = {
            "nose": detection[self.NOSE],
            "left_eye": detection[self.LEFT_EYE],
            "right_eye": detection[self.RIGHT_EYE],
            "left_ear": detection[self.LEFT_EAR],
            "right_ear": detection[self.RIGHT_EAR],
            "neck": (detection[self.LEFT_SHOULDER] + detection[self.RIGHT_SHOULDER]) / 2,
            "left_shoulder": detection[self.LEFT_SHOULDER],
            "right_shoulder": detection[self.RIGHT_SHOULDER],
            "left_elbow": detection[self.LEFT_ELBOW],
            "right_elbow": detection[self.RIGHT_ELBOW],
            "left_wrist": detection[self.LEFT_WRIST],
            "right_wrist": detection[self.RIGHT_WRIST],
            "left_hip": detection[self.LEFT_HIP],
            "right_hip": detection[self.RIGHT_HIP],
            "left_knee": detection[self.LEFT_KNEE],
            "right_knee": detection[self.RIGHT_KNEE],
            "left_ankle": detection[self.LEFT_ANKLE],
            "right_ankle": detection[self.RIGHT_ANKLE],
        }

        return keypoints

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Convert input to cupy array
        results = cp.asarray(in_message.get("inference_output"))[0]

        # Filter out low scores
        results = results[:, results[4, :] > self.score_threshold]
        scores = results[4, :]

        # If no detections, return zeros for all outputs
        if results.shape[1] == 0:
            out_message = Entity(context)
            zeros = hs.as_tensor(np.zeros([1, 2, 2]).astype(np.float32))

            for output in self.outputs:
                out_message.add(zeros, output)
            op_output.emit(out_message, "out")
            return

        results = results.transpose([1, 0])

        segments = []
        for i, detection in enumerate(results):
            # fmt: off
            kp = self.get_keypoints(detection)
            # Every two points defines a segment
            segments.append([kp["nose"], kp["left_eye"],      # nose <-> left eye
                             kp["nose"], kp["right_eye"],     # nose <-> right eye
                             kp["left_eye"], kp["left_ear"],  # ...
                             kp["right_eye"], kp["right_ear"],
                             kp["left_shoulder"], kp["right_shoulder"],
                             kp["left_shoulder"], kp["left_elbow"],
                             kp["left_elbow"], kp["left_wrist"],
                             kp["right_shoulder"], kp["right_elbow"],
                             kp["right_elbow"], kp["right_wrist"],
                             kp["left_shoulder"], kp["left_hip"],
                             kp["left_hip"], kp["left_knee"],
                             kp["left_knee"], kp["left_ankle"],
                             kp["right_shoulder"], kp["right_hip"],
                             kp["right_hip"], kp["right_knee"],
                             kp["right_knee"], kp["right_ankle"],
                             kp["left_hip"], kp["right_hip"],
                             kp["left_ear"], kp["neck"],
                             kp["right_ear"], kp["neck"],
                             ])
            # fmt: on

        cx, cy, w, h = results[:, 0], results[:, 1], results[:, 2], results[:, 3]
        x1, x2 = cx - w / 2, cx + w / 2
        y1, y2 = cy - h / 2, cy + h / 2

        data = {
            "boxes": cp.asarray(np.stack([x1, y1, x2, y2], axis=-1)).transpose([1, 0]),
            "noses": results[:, self.NOSE],
            "left_eyes": results[:, self.LEFT_EYE],
            "right_eyes": results[:, self.RIGHT_EYE],
            "left_ears": results[:, self.LEFT_EAR],
            "right_ears": results[:, self.RIGHT_EAR],
            "left_shoulders": results[:, self.LEFT_SHOULDER],
            "right_shoulders": results[:, self.RIGHT_SHOULDER],
            "left_elbows": results[:, self.LEFT_ELBOW],
            "right_elbows": results[:, self.RIGHT_ELBOW],
            "left_wrists": results[:, self.LEFT_WRIST],
            "right_wrists": results[:, self.RIGHT_WRIST],
            "left_hips": results[:, self.LEFT_HIP],
            "right_hips": results[:, self.RIGHT_HIP],
            "left_knees": results[:, self.LEFT_KNEE],
            "right_knees": results[:, self.RIGHT_KNEE],
            "left_ankles": results[:, self.LEFT_ANKLE],
            "right_ankles": results[:, self.RIGHT_ANKLE],
            "segments": cp.asarray(segments),
        }
        scores = cp.asarray(scores)

        out = self.nms(data, scores)

        # Rearrange boxes to be compatible with Holoviz
        out["boxes"] = cp.reshape(out["boxes"][None], (1, -1, 2))

        # Create output message
        out_message = Entity(context)
        for output in self.outputs:
            out_message.add(hs.as_tensor(out[output] / self.image_dim), output)
        op_output.emit(out_message, "out")

    def nms(self, inputs, scores):
        """Non-max suppression (NMS)
        Performs non-maximum suppression on input boxes according to their intersection-over-union (IoU).
        Filter out detections where the IoU is >= self.iou_threshold.

        See https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/ for an introduction to non-max suppression.

        Parameters
        ----------
        inputs : dictionary containing boxes, keypoints, and segments
        scores : array (n,)

        Returns
        ----------
        outputs : dictionary containing remaining boxes, keypoints, and segments after non-max supprerssion

        """

        boxes = inputs["boxes"]
        segments = inputs["segments"]

        if len(boxes) == 0:
            return cp.asarray([]), cp.asarray([])

        # Get coordinates
        x0, y0, x1, y1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :]

        # Area of bounding boxes
        area = (x1 - x0 + 1) * (y1 - y0 + 1)

        # Get indices of sorted scores
        indices = cp.argsort(scores)

        # Output boxes and scores
        boxes_out, segments_out, scores_out = [], [], []

        selected_indices = []

        # Iterate over bounding boxes
        while len(indices) > 0:
            # Get index with highest score from remaining indices
            index = indices[-1]
            selected_indices.append(index)
            # Pick bounding box with highest score
            boxes_out.append(boxes[:, index])
            segments_out.extend(segments[index])
            scores_out.append(scores[index])

            # Get coordinates
            x00 = cp.maximum(x0[index], x0[indices[:-1]])
            x11 = cp.minimum(x1[index], x1[indices[:-1]])
            y00 = cp.maximum(y0[index], y0[indices[:-1]])
            y11 = cp.minimum(y1[index], y1[indices[:-1]])

            # Compute IOU
            width = cp.maximum(0, x11 - x00 + 1)
            height = cp.maximum(0, y11 - y00 + 1)
            overlap = width * height
            union = area[index] + area[indices[:-1]] - overlap
            iou = overlap / union

            # Threshold and prune
            left = cp.where(iou < self.iou_threshold)
            indices = indices[left]

        selected_indices = cp.asarray(selected_indices)

        outputs = {
            "boxes": cp.asarray(boxes_out),
            "segments": cp.asarray(segments_out),
            "noses": inputs["noses"][selected_indices],
            "left_eyes": inputs["left_eyes"][selected_indices],
            "right_eyes": inputs["right_eyes"][selected_indices],
            "left_ears": inputs["left_ears"][selected_indices],
            "right_ears": inputs["right_ears"][selected_indices],
            "left_shoulders": inputs["left_shoulders"][selected_indices],
            "right_shoulders": inputs["right_shoulders"][selected_indices],
            "left_elbows": inputs["left_elbows"][selected_indices],
            "right_elbows": inputs["right_elbows"][selected_indices],
            "left_wrists": inputs["left_wrists"][selected_indices],
            "right_wrists": inputs["right_wrists"][selected_indices],
            "left_hips": inputs["left_hips"][selected_indices],
            "right_hips": inputs["right_hips"][selected_indices],
            "left_knees": inputs["left_knees"][selected_indices],
            "right_knees": inputs["right_knees"][selected_indices],
            "left_ankles": inputs["left_ankles"][selected_indices],
            "right_ankles": inputs["right_ankles"][selected_indices],
        }

        return outputs


class AdvNetworkingInferenceApp(Application):
    def __init__(self, data_path="none"):
        """Initialize the advanced networking inference demo application"""

        super().__init__()

        # set name
        self.name = "Advanced Networking Inference Demo App"

        if data_path == "none":
            data_path = os.path.join(
                os.environ.get("HOLOHUB_DATA_PATH", "./data"), "adv_networking_inference_demo"
            )

        self.sample_data_path = data_path

    def compose(self):
        # Initialize advanced network
        try:
            adv_net_config = self.from_config("advanced_network")
            if adv_network_common.adv_net_init(adv_net_config) != adv_network_common.Status.SUCCESS:
                logger.error("Failed to configure the Advanced Network manager")
                sys.exit(1)
            logger.info("Configured the Advanced Network manager")
        except Exception as e:
            logger.error(f"Failed to get advanced network config or initialize: {e}")
            sys.exit(1)

        # Get manager type
        try:
            mgr_type = adv_network_common.get_manager_type()
            logger.info(
                f"Using Advanced Network manager {adv_network_common.manager_type_to_string(mgr_type)}"
            )
        except Exception as e:
            logger.warning(f"Could not get manager type: {e}")

        output_type = "visualization"  # default

        try:
            output_config = self.kwargs("output_config")
            output_type = output_config.get("output_type", "visualization").lower()
            logger.info(f"Output type: {output_type}")
        except Exception:
            logger.info("No output_config found, using default output type: visualization")

        # Validate output type
        valid_output_types = ["visualization", "tx", "none"]
        if output_type not in valid_output_types:
            logger.error(f"ERROR: Invalid output_type '{output_type}'!")
            logger.error(f"Valid options are: {', '.join(valid_output_types)}")
            logger.error("  - 'visualization': Windowed display at configured resolution")
            logger.error(
                "  - 'tx': Headless network transmission at optimal resolution (1920x1080)"
            )
            logger.error("  - 'none': No output (for testing/debugging)")
            sys.exit(1)

        # Validate that output type is not 'none' in production
        if output_type == "none":
            logger.warning(
                "WARNING: Output type is 'none' - no visualization or transmission will occur!"
            )
            logger.warning("This mode is intended for testing/debugging only.")

        # Derive boolean flags for backward compatibility with existing logic
        enable_tx_output = output_type == "tx"
        enable_visualization = output_type == "visualization"
        logger.info(
            f"Derived flags - TX: {enable_tx_output}, Visualization: {enable_visualization}"
        )

        # Check RX/TX enabled status
        check_rx_tx_enabled(self, require_rx=True, require_tx=enable_tx_output)
        logger.info("RX is enabled, proceeding with application setup")

        allocator = UnboundedAllocator(self, name="allocator")

        # Create shared CUDA stream pool for all format converters and CUDA operations
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream_pool",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=2,  # Reserve streams for: preprocessor, TX format conversion
            max_size=6,  # Reduced from 8 since we removed input format converter
        )

        # ===== OPERATOR CONSTRUCTION IN PIPELINE ORDER =====

        # 1. RX operator (leftmost in pipeline)
        logger.info("Creating RX operator...")
        try:
            adv_net_media_rx = adv_network_media_rx.AdvNetworkMediaRxOp(
                fragment=self,
                name="advanced_network_media_rx",
                **self.kwargs("advanced_network_media_rx"),
            )
        except Exception as e:
            logger.error(f"Failed to create AdvNetworkMediaRxOp: {e}")
            sys.exit(1)

        # 2. Preprocessor - prepares data for inference (no input format conversion needed)
        logger.info("Creating preprocessor...")
        preprocessor_args = self.kwargs("preprocessor").copy()

        # With fixed RX operator, use RGB888 as input format
        in_dtype = "rgb888"

        # RX operator outputs tensor with default name, preprocessor expects default input
        preprocessor_args["in_tensor_name"] = ""
        preprocessor_args["out_tensor_name"] = "preprocessed"

        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=allocator,
            cuda_stream_pool=cuda_stream_pool,
            in_dtype=in_dtype,
            **preprocessor_args,
        )

        # 3. Format inference input operator - transposes tensor for inference
        logger.info("Creating format inference input operator...")
        format_input = FormatInferenceInputOp(
            self,
            name="transpose",
            allocator=allocator,
        )

        # 4. Inference operator - runs YOLO pose detection
        logger.info("Creating inference operator...")
        inference_args = self.kwargs("inference")
        inference_args["model_path_map"] = {
            "yolo_pose": os.path.join(self.sample_data_path, "yolo11l-pose.onnx")
        }
        inference = InferenceOp(
            self,
            name="inference",
            allocator=allocator,
            **inference_args,
        )

        # 5. Postprocessor - processes inference results
        logger.info("Creating postprocessor...")
        postprocessor_args = self.kwargs("postprocessor")
        # PostprocessorOp expects image_dim parameter (single value for square input)
        postprocessor_args["image_dim"] = preprocessor_args[
            "resize_width"
        ]  # Use width (should be same as height for square input)
        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=allocator,
            **postprocessor_args,
        )

        # 6. HolovizOp - visualization and rendering
        logger.info("Creating HolovizOp...")
        holoviz = None
        holoviz_args = None

        if output_type == "tx":
            # TX mode: headless, high resolution
            try:
                holoviz_args = self.kwargs("holoviz_tx_only").copy()
                logger.info("Using holoviz_tx_only configuration (headless mode)")
            except Exception:
                holoviz_args = self.kwargs("holoviz").copy()
                holoviz_args["headless"] = True
                holoviz_args["enable_render_buffer_output"] = True
                logger.info("Using default holoviz configuration with TX modifications")

            # Use TX resolution for optimal quality
            tx_config = self.kwargs("advanced_network_media_tx")
            holoviz_args["width"] = tx_config.get("frame_width", 1920)
            holoviz_args["height"] = tx_config.get("frame_height", 1080)
            logger.info(
                f"HolovizOp: TX mode at {holoviz_args['width']}x{holoviz_args['height']} (headless)"
            )

        elif output_type == "visualization":
            # Visualization mode: windowed, configured resolution
            holoviz_args = self.kwargs("holoviz").copy()
            logger.info(
                f"HolovizOp: Visualization mode at {holoviz_args.get('width', 'default')}x{holoviz_args.get('height', 'default')} (windowed)"
            )

        elif output_type == "none":
            # No output mode: skip HolovizOp creation
            logger.info("Skipping HolovizOp creation (output_type is 'none')")

        if holoviz_args:
            holoviz = HolovizOp(self, allocator=allocator, name="holoviz", **holoviz_args)
            logger.info("Created HolovizOp with appropriate configuration")

        # 7. TX format converter and TX operator (rightmost in pipeline)
        tx_format_converter = None
        tx_operator = None
        if output_type == "tx":
            logger.info("Creating TX operators...")
            try:
                # TX format converter - converts RGBA to RGB888 for transmission
                tx_config = self.kwargs("advanced_network_media_tx")
                tx_width = tx_config.get("frame_width", 1920)
                tx_height = tx_config.get("frame_height", 1080)

                tx_format_converter = FormatConverterOp(
                    fragment=self,
                    name="tx_format_converter",
                    pool=allocator,
                    cuda_stream_pool=cuda_stream_pool,
                    in_dtype="rgba8888",
                    out_dtype="rgb888",
                    # Don't specify out_tensor_name to use default (empty string)
                    # No resize needed - HolovizOp already renders at TX resolution
                )
                logger.info(
                    f"Created TX format converter for {tx_width}x{tx_height} (RGBA -> RGB888)"
                )

                # TX operator - transmits processed video over network
                tx_operator = adv_network_media_tx.AdvNetworkMediaTxOp(
                    fragment=self,
                    name="advanced_network_media_tx",
                    **self.kwargs("advanced_network_media_tx"),
                )
                logger.info("Created AdvNetworkMediaTxOp for output transmission")

            except Exception as e:
                logger.error(f"Failed to create TX operators: {e}")
                sys.exit(1)

        # ===== PIPELINE CONNECTIONS =====
        logger.info("Setting up pipeline connections...")

        # Network RX -> HolovizOp (for background image display)
        if holoviz:
            self.add_flow(adv_net_media_rx, holoviz, {("out_video_buffer", "receivers")})
            logger.info("Connected RX operator to HolovizOp for background image")

        # Network RX -> Preprocessor (for inference pipeline)
        self.add_flow(adv_net_media_rx, preprocessor, {("out_video_buffer", "")})
        logger.info("Connected RX operator to preprocessor for inference pipeline")

        # Common inference pipeline connections
        self.add_flow(
            preprocessor, format_input, {("tensor", "in")}
        )  # preprocessor outputs via "tensor" port
        self.add_flow(
            format_input, inference, {("out", "receivers")}
        )  # format_input outputs via "out" port
        self.add_flow(inference, postprocessor, {("transmitter", "in")})

        # Connect postprocessor to HolovizOp for pose overlay (if HolovizOp exists)
        if holoviz:
            self.add_flow(postprocessor, holoviz, {("out", "receivers")})
            logger.info("Connected inference pipeline to HolovizOp for pose overlay")
        else:
            logger.info("No HolovizOp - inference results will not be visualized")

        # TX output pipeline connections (if TX mode is enabled)
        if output_type == "tx" and tx_operator and tx_format_converter and holoviz:
            self.add_flow(holoviz, tx_format_converter, {("render_buffer_output", "")})
            self.add_flow(tx_format_converter, tx_operator, {("", "input")})
            logger.info(
                "Connected TX pipeline: HolovizOp -> TX Format Converter -> TX Operator (RGBA -> RGB888)"
            )

        # Log final pipeline configuration
        logger.info(
            f"Pipeline configured for output_type: '{output_type}' with direct RX connections"
        )

        # Set up scheduler
        try:
            scheduler_config = self.kwargs("scheduler")
            scheduler = MultiThreadScheduler(
                fragment=self, name="multithread-scheduler", **scheduler_config
            )
            self.scheduler(scheduler)
        except Exception as e:
            logger.error(f"Failed to set up scheduler: {e}")
            sys.exit(1)

        logger.info("Application composition completed successfully")


def main():
    # Parse args
    parser = ArgumentParser(description="Advanced Networking Inference Demo Application.")
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(
            os.path.dirname(__file__), "./adv_networking_inference_demo.yaml"
        )
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            logger.error(
                "Please provide a config file with --config or create adv_networking_inference_demo.yaml"
            )
            sys.exit(1)
    else:
        config_file = args.config

    # Convert to absolute path if relative
    config_path = Path(config_file)
    if not config_path.is_absolute():
        # Get the directory of the script and make path relative to it
        script_dir = Path(__file__).parent.resolve()
        config_path = script_dir / config_path

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Using config file: {config_path}")

    try:
        app = AdvNetworkingInferenceApp(args.data)
        app.config(str(config_path))
        app.enable_metadata(False)

        logger.info("Starting application...")
        app.run()

        logger.info("Application finished")

    except Exception as e:
        logger.error(f"Application failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Shutdown advanced network (matching C++ behavior)
        try:
            adv_network_common.shutdown()
            logger.info("Advanced Network shutdown completed")
        except Exception as e:
            logger.warning(f"Error during advanced network shutdown: {e}")


if __name__ == "__main__":
    main()
