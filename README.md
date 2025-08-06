# ATGen: Active Learning for Natural Language Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for applying active learning techniques to natural language generation tasks. This repository contains implementations of various active learning strategies specifically designed for text generation models, helping to reduce annotation costs while maximizing model performance.

## 🌟 Features

- **Multiple Active Learning Strategies**: Implementation of strategies like HUDS, HADAS, FAC-LOC, IDDS, and more
- **Flexible Model Support**: Compatible with various language models (Qwen, Llama, etc.)
- **Comprehensive Evaluation**: Supports multiple evaluation metrics including ROUGE, BLEU, BERTScore, AlignScore, etc.
- **Interactive Visualization**: Streamlit dashboard for exploring results and comparing strategies
- **Hydra Configuration**: Easily configurable experiments through Hydra's YAML-based configuration system
- **PEFT Integration**: Efficient fine-tuning using Parameter-Efficient Fine-Tuning methods

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (for model training)
- Dependencies listed in `requirements.txt`

## 🔧 Installation

### From PyPI (Stable Release)

```bash
pip install atgen
```

### From GitHub Main Branch (Latest Development)

```bash
pip install git+https://github.com/Aktsvigun/atgen.git
```

### Editable/Development Installation

For development (e.g. a new AL / subset selection strategy) or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/Aktsvigun/atgen.git
cd atgen

# Install in editable mode
pip install -e .
```

This will install the package in editable mode and allow you to make changes to the code and see them immediately reflected without reinstalling the package.

## 🚀 Usage

### Running Active Learning Experiments

Experiments can be launched using the `run-al` command:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_NAME=base run-al
```

Parameters:
- `CUDA_VISIBLE_DEVICES`: Specify which GPU to use
- `HYDRA_CONFIG_NAME`: Configuration file (e.g., `base`, `custom`, `test`)

Additional parameters can be overridden via the command line following Hydra's syntax:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_NAME=base run-al al.strategy=huds model.checkpoint=Qwen/Qwen2.5-7B
```

### Interactive Dashboard

Launch the Streamlit application to explore and visualize your experiments:

```bash
streamlit run Welcome.py
```

Navigate to `http://localhost:8501` in your web browser to access the dashboard.

## 📁 Project Structure

- `configs/`: Configuration files for experiments
  - `al/`: Active learning strategy configurations
  - `data/`: Dataset configurations
  - `labeller/`: Labeller configurations
- `src/atgen/`: Main package
  - `strategies/`: Implementation of active learning strategies
  - `metrics/`: Code for evaluation metrics
  - `utils/`: Utility functions
  - `run_scripts/`: Scripts for running experiments
  - `labellers/`: Labelling mechanisms
  - `visualize/`: Visualization tools
- `pages/`: Streamlit application pages
- `outputs/`: Experimental results storage
- `cache/`: Cached computations to speed up repeated runs

## 📚 Supported Active Learning Strategies

- `huds`: Hypothetical Document Scoring
- `hadas`: Harmonic Diversity Scoring
- `random`: Random sampling baseline
- `fac-loc`: Facility Location strategy
- `idds`: Improved Diverse Density Scoring
- And more...

## 📊 Supported Datasets

The toolkit comes pre-configured for several datasets including summarization, question answering, and other generative tasks. Custom datasets can be added by creating new configuration files.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🔗 Citation

If you use this toolkit in your research, please cite:

```
@inproceedings{tsvigun-etal-2025-atgen,
    title = "{ATG}en: A Framework for Active Text Generation",
    author = "Tsvigun, Akim  and
      Vasilev, Daniil  and
      Tsvigun, Ivan  and
      Lysenko, Ivan  and
      Bektleuov, Talgat  and
      Medvedev, Aleksandr  and
      Vinogradova, Uliana  and
      Severin, Nikita  and
      Mozikov, Mikhail  and
      Savchenko, Andrey  and
      Makarov, Ilya  and
      Rostislav, Grigorev  and
      Kuleev, Ramil  and
      Zhdanov, Fedor  and
      Shelmanov, Artem",
    editor = "Mishra, Pushkar  and
      Muresan, Smaranda  and
      Yu, Tao",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-demo.63/",
    doi = "10.18653/v1/2025.acl-demo.63",
    pages = "653--665",
    ISBN = "979-8-89176-253-4",
}
```
