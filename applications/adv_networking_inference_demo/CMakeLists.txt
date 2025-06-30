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

project(adv_networking_inference_demo NONE)

find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Download the yolov11 pose model if it doesn't exist
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/adv_networking_inference_demo/yolo11l-pose.onnx"
    COMMAND "PATH=${PATH}:/workspace/holohub/.local/bin" /usr/local/bin/yolo export model=yolo11l-pose.pt format=onnx
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/adv_networking_inference_demo"
    COMMAND mv yolo11l-pose.onnx "${HOLOHUB_DATA_DIR}/adv_networking_inference_demo/"
    )

add_custom_target(adv_networking_inference_demo_data ALL
                  DEPENDS
                  "${HOLOHUB_DATA_DIR}/adv_networking_inference_demo/yolo11l-pose.onnx")


# Install application and dependencies into the install/ directory for packaging
install(
  FILES adv_networking_inference_demo.py
  DESTINATION bin/adv_networking_inference_demo
  COMPONENT adv_networking_inference_demo-py
)

install(
  FILES adv_networking_inference_demo.yaml
  DESTINATION bin/adv_networking_inference_demo
  COMPONENT adv_networking_inference_demo-py
)

install(
  FILES README.md
  DESTINATION bin/adv_networking_inference_demo
  COMPONENT adv_networking_inference_demo-py
)

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing (reduce processing time for tests)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/adv_networking_inference_demo.yaml" CONFIG_FILE)
  # Note: This inference demo doesn't have a count parameter like body_pose_estimation
  # so we'll use the original config for testing
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/adv_networking_inference_demo_testing.yaml ${CONFIG_FILE})

  add_custom_target(adv_networking_inference_demo_test ALL
    DEPENDS "adv_networking_inference_demo.py"
    DEPENDS adv_networking_inference_demo_data
  )

  # Add test - this will require actual network hardware, so it's more of an integration test
  add_test(NAME adv_networking_inference_demo_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/adv_networking_inference_demo.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/adv_networking_inference_demo_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/adv_networking_inference_demo"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST adv_networking_inference_demo_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib")

  # Note: The test may fail if no network hardware is available, which is expected
  set_tests_properties(adv_networking_inference_demo_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Application finished;Advanced Network shutdown completed"
                FAIL_REGULAR_EXPRESSION "Fatal|FATAL|Critical|CRITICAL")

endif()
