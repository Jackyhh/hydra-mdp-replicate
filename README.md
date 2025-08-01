# Hydra-MDP Replication

A personal replication and ongoing extension of NVIDIA’s [Hydra-MDP](https://github.com/NVlabs/hydra-mdp) framework for end-to-end motion planning in autonomous driving.

## 🚀 Features

- **Core Model**: Full reimplementation of the Hydra-MDP architecture (`hydra_mdp.py`), including backbone and transformer modules.  
- **Training Pipeline**: Configurable training entrypoint in `main.py` and `trainer.py`, with support for multi-GPU setups.  
- **Perception Module**: Standalone perception network (`perception_network.py`) and trajectory decoding (`trajectory_decoder.py`).  
- **Data Interface**: Utility functions in `data_utils.py` (currently under development).  

## 📌 Project Status

> **Work in Progress**  
> - Core model and training/inference pipelines are functional.  
> - Data I/O (`data_utils.py`) is partially implemented; dataset loading and preprocessing will be finalized soon.  
> - Contributions toward dataset adapters, clearer documentation, and demo notebooks are welcome.

## 📦 Installation

```bash
git clone https://github.com/Jackyhh/hydra-mdp-replicate.git
cd hydra-mdp-replicate
pip install -r requirements.txt

## ✉️  Contact

For questions or contributions, feel free to open an issue or reach out at <guojiaqi9316@gmail.com>.
This project is independently maintained and is not officially affiliated with NVIDIA.

