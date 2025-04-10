# ComfyUI-UNO

Make UNO avialbe in ComfyUI.

🔥🔥 [UNO](https://github.com/bytedance/UNO): A Universal Customization Method for Both Single and Multi-Subject Conditioning. The arXiv [paper](https://arxiv.org/abs/2504.02160) of UNO is released. Less-to-More Generalization: Unlocking More Controllability by In-Context Generation.


## Installation

1. Make sure you have ComfyUI installed

2. Clone this repository into your ComfyUI's custom_nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/Yuan-ManX/ComfyUI-UNO.git
```

3. Install dependencies:
```
cd ComfyUI-UNO
pip install -r requirements.txt
```

## Models

Download [model](https://huggingface.co/bytedance-research/UNO)

Download checkpoints in one of the three ways:

1. Directly run the inference scripts, the checkpoints will be downloaded automatically by the `hf_hub_download` function in the code to your `$HF_HOME`(the default value is `~/.cache/huggingface`).

2. use `huggingface-cli download <repo name>` to download `black-forest-labs/FLUX.1-dev`, `xlabs-ai/xflux_text_encoders`, `openai/clip-vit-large-patch14`, `bytedance-research/UNO`, then run the inference scripts.

3.  use `huggingface-cli download <repo name> --local-dir <LOCAL_DIR>` to download all the checkpoints menthioned in 2. to the directories your want. Then set the environment variable `AE`, `FLUX`, `T5`, `CLIP`, `LORA` to the corresponding paths. Finally, run the inference scripts.
