# Advanced Networking Media Player

The Advanced Networking Media Player is a high-performance application for receiving and displaying media streams over advanced network infrastructure using NVIDIA's Rivermax SDK. This application demonstrates real-time media streaming capabilities with ultra-low latency and high throughput.

## Overview

This application showcases professional-grade media streaming over IP networks, utilizing NVIDIA's advanced networking technologies. It receives media streams using the SMPTE 2110 standard and can either display them in real-time or save them to disk for further processing.

### Key Features

- **High-Performance Streaming**: Receive media streams with minimal latency using Rivermax SDK
- **SMPTE 2110 Compliance**: Industry-standard media over IP protocol support
- **Flexible Output**: Choose between real-time visualization or file output
- **GPU Acceleration**: Leverage GPUDirect for zero-copy operations
- **Multiple Format Support**: RGB888, YUV420, NV12, and other common video formats
- **Header-Data Split**: Optimized memory handling for improved performance

### Pipeline Architecture

```
Network Interface → AdvNetworkMediaRxOp → HolovizOp (Display)
                                      └─→ FramesWriter (File)
```

## Requirements

### Hardware Requirements
- Linux system (x86_64 or aarch64)
- NVIDIA NIC with ConnectX-6 or later chip
- NVIDIA GPU (for visualization and GPU acceleration)
- Sufficient network bandwidth for target media streams

### Software Requirements
- NVIDIA Rivermax SDK
- NVIDIA GPU drivers
- MOFED drivers (5.8-1.0.1.1 or later)
- DOCA 2.7 or later (if using DOCA backend)
- System tuning as described in the [High Performance Networking tutorial](../../tutorials/high_performance_networking/README.md)

## Build Instructions

### Build Docker Image

Build the Docker image with Rivermax support:

```bash
./dev_container build --docker_file operators/advanced_network/Dockerfile --img holohub:rivermax --build-args "--target rivermax"
```

### Launch Container

Launch the Rivermax-enabled container:

```bash
./operators/advanced_network/run_rivermax.sh
```

### Build Application

Inside the container, build the application:

```bash
# For C++ version
./run build adv_networking_media_player --configure-args "-DANO_MGR=rivermax"
```

## Configuration

The application uses a YAML configuration file that defines:

- **Network settings**: Interface configuration, IP addresses, ports
- **Memory regions**: CPU and GPU buffer configurations
- **Video parameters**: Format, resolution, bit depth, frame rate
- **Output options**: Visualization or file output settings

### Example Configuration Sections

#### Network Interface Configuration
```yaml
advanced_network:
  cfg:
    version: 1
    manager: "rivermax"
    master_core: 6              # Master CPU core
    debug: 1
    log_level: "error"

    memory_regions:
    - name: "Data_RX_CPU"
      kind: "host"
      affinity: 0
      access:
        - local
      num_bufs: 43200
      buf_size: 20
    - name: "Data_RX_GPU"
      kind: "device"
      affinity: 0
      access:
        - local
      num_bufs: 43200
      buf_size: 1440
    interfaces:
    - name: data1
      address:  cc:00.1
      rx:
        queues:
        - name: "Data"
          id: 0
          cpu_core:  "12"
          batch_size: 4320
          output_port: "bench_rx_out_1"
          memory_regions: 
          - "Data_RX_CPU"
          - "Data_RX_GPU"
          rivermax_rx_settings:
            settings_type: "ipo_receiver"
            memory_registration: true
            #allocator_type: "huge_page_2mb"
            verbose: true
            max_path_diff_us: 10000
            ext_seq_num: true
            sleep_between_operations_us: 0
            local_ip_addresses:
              - 2.1.0.12
            source_ip_addresses:
              - 2.1.0.12
            destination_ip_addresses:
              - 224.1.1.2
            destination_ports:
              - 50001
            stats_report_interval_ms: 3000
            send_packet_ext_info: true
```

#### Video Format Configuration
```yaml
advanced_network_media_rx:
  interface_name: cc:00.1
  video_format: RGB888
  frame_width: 1920
  frame_height: 1080
  bit_depth: 8
  hds: true
  output_format: tensor #can be video_buffer
  memory_location: device
```

#### Output Configuration
```yaml
media_player_config:
  write_to_file: false    # Set to true for file output
  visualize: true         # Set to true for real-time display
  input_format: "rgb888"
```

## Running the Application

### Prerequisites

Before running, ensure your environment is properly configured:

```bash
# Update PYTHONPATH for Python applications
export PYTHONPATH=${PYTHONPATH}:/opt/nvidia/holoscan/python/lib:$PWD/build/adv_networking_media_player/python/lib:$PWD

# Ensure proper system configuration (run as root if needed)
# See High Performance Networking tutorial for system tuning
```

### C++ Application

```bash
./build/adv_networking_media_player/applications/adv_networking_media_player/cpp/adv_networking_media_player adv_networking_media_player.yaml
```

### Python Application

```bash
python applications/adv_networking_media_player/python/adv_networking_media_player.py ../adv_networking_media_player.yaml
```

## Output Options

### Real-time Visualization

When `visualize: true` is set in the configuration:
- Received media streams are displayed in real-time using HolovizOp
- Supports format conversion for optimal display
- Minimal latency for live monitoring applications

### File Output

When `write_to_file: true` is set in the configuration:
- Media frames are saved to disk for later analysis
- Configure output path and number of frames in the `frames_writer` section
- Supports both host and device memory sources

## Supported Video Formats

- **RGB888**: 24-bit RGB color
- **YUV420**: 4:2:0 chroma subsampling
- **NV12**: Semi-planar YUV 4:2:0

## Troubleshooting

### Common Issues

1. **Permission Errors**: Run with appropriate privileges for network interface access
2. **Network Configuration**: Verify IP addresses, ports, and interface names
3. **Memory Issues**: Adjust buffer sizes based on available system memory
4. **Performance**: Check system tuning and CPU isolation settings

### Debug Options

Enable debug logging by setting `log_level: "debug"` in the advanced_network configuration section.

### Git Configuration (if needed)

If you encounter Git-related issues during build:

```bash
git config --global --add safe.directory '*'
```

## Performance Tuning

- **CPU Isolation**: Isolate CPU cores for network processing
- **Memory Configuration**: Tune buffer sizes based on frame rate and resolution
- **GPU Memory**: Ensure sufficient GPU memory for frame buffers
- **Network Settings**: Configure appropriate batch sizes and queue parameters

## Related Documentation

- [Advanced Network Operators](../../operators/advanced_network/README.md)
- [Advanced Network Media Operators](../../operators/advanced_network_media/README.md)
- [High Performance Networking Tutorial](../../tutorials/high_performance_networking/README.md)
- [Advanced Networking Media Sender](../adv_networking_media_sender/README.md)
