/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "adv_network_tx.h"
#include "advanced_network/manager.h"
#include <memory>
#include <assert.h>

using namespace holoscan::advanced_network;

namespace holoscan::ops {

struct AdvNetworkOpTx::AdvNetworkOpTxImpl {
  NetworkConfig cfg;
  Manager* mgr;
};

void AdvNetworkOpTx::setup(OperatorSpec& spec) {
  spec.input<BurstParams*>("burst_in");

  spec.param(cfg_,
             "cfg",
             "Configuration",
             "Configuration for the advanced network operator",
             NetworkConfig());
}

void AdvNetworkOpTx::stop() {
  HOLOSCAN_LOG_INFO("AdvNetworkOpTx::stop()");
  impl->mgr->shutdown();
}

void AdvNetworkOpTx::initialize() {
  HOLOSCAN_LOG_INFO("AdvNetworkOpTx::initialize()");
  register_converter<holoscan::advanced_network::NetworkConfig>();

  holoscan::Operator::initialize();
  if (Init() < 0) { throw std::runtime_error("ANO initialization failed"); }
}

int AdvNetworkOpTx::Init() {
  impl = new AdvNetworkOpTxImpl();
  impl->cfg = cfg_.get();

  ManagerFactory::set_manager_type(impl->cfg.common_.manager_type);

  impl->mgr = &(ManagerFactory::get_active_manager());

  assert(impl->mgr != nullptr && "ANO Manager is not initialized");

  if (!impl->mgr->set_config_and_initialize(impl->cfg)) { return -1; }

  return 0;
}

void AdvNetworkOpTx::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                             [[maybe_unused]] ExecutionContext&) {
  BurstParams* d_params;
  auto rx = op_input.receive<BurstParams*>("burst_in");
  if (!rx.has_value() || rx.value() == nullptr) {
    HOLOSCAN_LOG_ERROR("No burst received from input");
    return;
  }

  const auto tx_res = impl->mgr->send_tx_burst(rx.value());
  if (tx_res != Status::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to send TX burst to ANO: {}", static_cast<int>(tx_res));
    return;
  }
}
};  // namespace holoscan::ops
