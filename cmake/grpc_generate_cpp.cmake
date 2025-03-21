# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# A helper function to generate C++ code from given .proto files.
cmake_minimum_required(VERSION 3.8)
find_package(Threads REQUIRED)

# Output:
# SRCS: list of source files to be generated by protoc compiler
# HDRS: list of header files to be generated by protoc compiler
# INCLUDE_DIRS: list of directories where the generated files will be stored
# Input:
# PROTO_FILE: list of .proto files that need to be compiled into C++ code
function(grpc_generate_cpp SRCS HDRS INCLUDE_DIRS)
    # Expect:
    # - PROTOC_EXECUTABLE: path to protoc
    # - GRPC_CPP_EXECUTABLE: path to grpc_cpp_plugin
    if(NOT ARGN)
        message(SEND_ERROR "Error: grpc_generate_cpp() called without any .proto files")
        return()
    endif()

    foreach(PROTO_FILE ${ARGN})
        message(STATUS "Build proto file ${PROTO_FILE} in C++")
        # Get the full path to the proto file
        get_filename_component(_abs_proto_file "${PROTO_FILE}" ABSOLUTE)
        # Get the name of the proto file without extension
        get_filename_component(_proto_name_we ${PROTO_FILE} NAME_WE)
        # Get the parent directory of the proto file
        get_filename_component(_proto_parent_dir ${_abs_proto_file} DIRECTORY)
        # Append 'generated' to the parent directory
        set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/cpp")
        file(MAKE_DIRECTORY ${_generated_dir})

        set(_protobuf_include_path -I ${_proto_parent_dir})

        set(_proto_srcs "${_generated_dir}/${_proto_name_we}.pb.cc")
        set(_proto_hdrs "${_generated_dir}/${_proto_name_we}.pb.h")
        set(_grpc_srcs "${_generated_dir}/${_proto_name_we}.grpc.pb.cc")
        set(_grpc_hdrs "${_generated_dir}/${_proto_name_we}.grpc.pb.h")

        file(REMOVE "${_proto_srcs}" "${_proto_hdrs}" "${_grpc_srcs}" "${_grpc_hdrs}")
        add_custom_command(
            OUTPUT "${_proto_srcs}" "${_proto_hdrs}" "${_grpc_srcs}" "${_grpc_hdrs}"
            COMMAND ${PROTOC_EXECUTABLE}
            ARGS --grpc_out=${_generated_dir}
            --cpp_out=${_generated_dir}
            --plugin=protoc-gen-grpc=${GRPC_CPP_EXECUTABLE}
            ${_protobuf_include_path} ${_abs_proto_file}
            DEPENDS ${_abs_proto_file}
            COMMENT "Running gRPC C++ protocol buffer compiler on ${PROTO_FILE}"
            VERBATIM
        )

        list(APPEND ${SRCS} "${_proto_srcs}")
        list(APPEND ${HDRS} "${_proto_hdrs}")
        list(APPEND ${SRCS} "${_grpc_srcs}")
        list(APPEND ${HDRS} "${_grpc_hdrs}")
        list(APPEND ${INCLUDE_DIRS} "${_generated_dir}")
    endforeach()

    set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
    set(${SRCS} "${${SRCS}}" PARENT_SCOPE)
    set(${HDRS} "${${HDRS}}" PARENT_SCOPE)
    set(${INCLUDE_DIRS} "${${INCLUDE_DIRS}}" PARENT_SCOPE)
endfunction()

include(FetchContent)
set(ABSL_ENABLE_INSTALL OFF)
set(gRPC_INSTALL OFF)
set(protobuf_INSTALL OFF)
set(CARES_INSTALL OFF)
FetchContent_Declare(
  grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  # when using gRPC, you will actually set this to an existing tag, such as
  # v1.25.0, v1.26.0 etc..
  # For the purpose of testing, we override the tag used to the commit
  # that's currently under test.
  GIT_TAG        v1.54.2)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(grpc)

set(PROTOBUF_LIBPROTOBUF libprotobuf)
set(GRPCPP_REFLECTION grpc++_reflection)
set(PROTOC_EXECUTABLE $<TARGET_FILE:protoc>)
set(GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(GRPC_CPP_EXECUTABLE grpc_cpp_plugin)
else()
  set(GRPC_CPP_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()
# Expose variables with PARENT_SCOPE so that
# root project can use it for including headers and using executables
set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE} PARENT_SCOPE)
set(GRPC_CPP_EXECUTABLE ${GRPC_CPP_EXECUTABLE} PARENT_SCOPE)
set(PROTOBUF_LIBPROTOBUF ${PROTOBUF_LIBPROTOBUF} PARENT_SCOPE)
set(GRPCPP_REFLECTION ${GRPCPP_REFLECTION} PARENT_SCOPE)
set(GRPC_GRPCPP ${GRPC_GRPCPP} PARENT_SCOPE)
