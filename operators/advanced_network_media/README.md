### Advanced Network Media Operators

This directory contains operators for high-performance media streaming over advanced network infrastructure. The operators provide efficient transmission and reception of media frames (such as video) using NVIDIA's Rivermax SDK and other high-performance networking technologies.

> [!NOTE]
> These operators build upon the [Advanced Network library](../advanced_network/README.md) to provide specialized functionality for media streaming applications. They are designed for professional broadcast and media streaming use cases that require strict timing and high throughput.

#### Operators

The Advanced Network Media library provides two main operators:

##### `holoscan::ops::AdvNetworkMediaRxOp`

Operator for receiving media frames over advanced network infrastructure. This operator receives video frames over Rivermax-enabled network infrastructure and outputs them as GXF VideoBuffer or Tensor entities.

**Inputs**
- None (receives data directly from network interface via Advanced Network Manager library)

**Outputs**
- **`output`**: Video frames as GXF entities (VideoBuffer or Tensor)
  - type: `gxf::Entity`

**Parameters**
- **`interface_name`**: Name of the network interface to use for receiving
  - type: `std::string`
- **`queue_id`**: Queue ID for the network interface (default: 0)
  - type: `uint16_t`
- **`frame_width`**: Width of incoming video frames in pixels
  - type: `uint32_t`
- **`frame_height`**: Height of incoming video frames in pixels
  - type: `uint32_t`
- **`bit_depth`**: Bit depth of the video format
  - type: `uint32_t`
- **`video_format`**: Video format specification (e.g., "RGB888", "YUV422")
  - type: `std::string`
- **`hds`**: Indicates if header-data split mode is enabled in the input data
  - type: `bool`
- **`output_format`**: Output format for the received frames ("video_buffer" for VideoBuffer, "tensor" for Tensor)
  - type: `std::string`
- **`memory_location`**: Memory location for frame buffers ("device", "host", etc.)
  - type: `std::string`

##### `holoscan::ops::AdvNetworkMediaTxOp`

Operator for transmitting media frames over advanced network infrastructure. This operator processes video frames from GXF entities (either VideoBuffer or Tensor) and transmits them over Rivermax-enabled network infrastructure.

**Inputs**
- **`input`**: Video frames as GXF entities (VideoBuffer or Tensor)
  - type: `gxf::Entity`

**Outputs**
- None (transmits data directly to network interface)

**Parameters**
- **`interface_name`**: Name of the network interface to use for transmission
  - type: `std::string`
- **`queue_id`**: Queue ID for the network interface (default: 0)
  - type: `uint16_t`
- **`video_format`**: Video format specification (e.g., "RGB888", "YUV422")
  - type: `std::string`
- **`bit_depth`**: Bit depth of the video format
  - type: `uint32_t`
- **`frame_width`**: Width of video frames to transmit in pixels
  - type: `uint32_t`
- **`frame_height`**: Height of video frames to transmit in pixels
  - type: `uint32_t`

#### Requirements

- All requirements from the [Advanced Network library](../advanced_network/README.md)
- NVIDIA Rivermax SDK
- Compatible video formats and frame rates
- Proper network configuration for media streaming

#### Features

- **High-performance media streaming**: Optimized for professional broadcast applications
- **SMPTE 2110 compliance**: Supports industry-standard media over IP protocols
- **Low latency**: Direct hardware access minimizes processing delays
- **GPU acceleration**: Supports GPUDirect for zero-copy operations
- **Flexible formats**: Support for various video formats and bit depths
- **Header-data split**: Optimized memory handling for improved performance

#### Example Usage

For complete examples of how to use these operators, see:

- [Advanced Networking Media Player](../../applications/adv_networking_media_player/README.md) - Demonstrates receiving and displaying media streams
- [Advanced Networking Media Sender](../../applications/adv_networking_media_sender/README.md) - Demonstrates transmitting media streams


Please refer to the [Advanced Network library documentation](../advanced_network/README.md) for detailed configuration instructions.

#### System Requirements

> [!IMPORTANT]  
> Review the [High Performance Networking tutorial](../../tutorials/high_performance_networking/README.md) for guided instructions to configure your system and test the Advanced Network Media operators.

- Linux
- NVIDIA NIC with ConnectX-6 or later chip
- NVIDIA Rivermax SDK
- System tuning as described in the High Performance Networking tutorial
- Sufficient memory and bandwidth for media streaming workloads

