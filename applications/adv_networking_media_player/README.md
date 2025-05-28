# README

This project relies on the Rivermax Manager and utilizes the `dev_container` script to build the Docker image and the `run` script to build the application. Some additional configuration may be required for Rivermax to function correctly, please look at the Rivermax setup script `operators/advanced_network/run_rivermax.sh`.

## Build and Run Instructions

### Build Docker Image

To build the Docker image, use the following command:
```bash
sudo ./dev_container build --docker_file operators/advanced_network/DockerfileRivermax --img holohub:rivermax
```

### Run Docker Image

To run the Docker image, use:
```bash
./operators/advanced_network/run_rivermax.sh
```

### Build the Application

#### Build Without Inference
To build the application without inference support:
```bash
./run build network_player --type release --configure-args "-GNinja -DANO_MGR=rivermax"
```

#### Build With Inference
To build the application with inference enabled:
```bash
./run build network_player --type release --configure-args "-GNinja -DANO_MGR=rivermax -DNETWORK_PLAYER_INFERENCE=ON"
```

## Running the Application

Before running any Python applications, make sure to update the `PYTHONPATH` as follows:
```bash
export PYTHONPATH=${PYTHONPATH}:/opt/nvidia/holoscan/python/lib:$PWD/build/network_player/python/lib:$PWD
```

### Run Basic App on Port 50001

- **Python**:
    ```bash
    python applications/network_player/python/main.py applications/network_player/network_player_rmax.yaml
    ```

- **C++**:
    ```bash
    build/network_player/applications/network_player/network_player applications/network_player/network_player_rmax.yaml
    ```

### Run Multi Streams on Ports 50001 - 50004

- **Python**:
    ```bash
    python applications/network_player/python/main.py applications/network_player/network_player_rmax_multi_streams.yaml
    ```

- **C++**:
    ```bash
    build/network_player/applications/network_player/network_player applications/network_player/network_player_rmax_multi_streams.yaml
    ```

### Run With Inference (Port 50001)

To run the application with inference (only available when built with inference support):
```bash
python applications/network_player/python/inference.py applications/network_player/python/network_player_rmax_inference.yaml
```

*Note: The model conversion might take some time on the first run. Do not send any data until the model is fully loaded.*

### Git Troubleshooting

If you encounter issues with Git during the build process, you can resolve them by running:

```bash
git config --global --add safe.directory '*'
```
