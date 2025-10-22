# LongCat-Audio-Semantic

This project provides tools to extract continuous semantic features from intermediate layers of the encoder from the [LongCat-Audio-Codec](https://github.com/meituan-longcat/LongCat-Audio-Codec) model.

Instead of using the final quantized tokens from the codec, this repository allows for the extraction of rich, continuous vector representations from a specified transformer layer within the semantic encoder.

## Purpose

The primary goal is to obtain a continuous representation of audio semantics, which can be used for various downstream tasks, such as:
*   Audio content analysis and retrieval.
*   As input features for other neural networks.
*   Research on speech representation learning.

## Components

*   `encoder.py`: Contains the core logic for loading the pre-trained `LongCat-Audio-Codec` semantic encoder and extracting features from a specified layer. It defines the `EmbeddingExtractor` class.
*   `feature_extractor.py`: A module responsible for audio preprocessing. It computes Fbank features from a raw waveform and applies Cepstral Mean and Variance Normalization (CMVN).
*   `cmvn.npy`: Stores the statistics (mean and variance) required for CMVN. This file is necessary for the `feature_extractor`.
*   `LongCat-Audio-Codec/`: A git submodule pointing to the original repository, which provides the model architecture and pre-trained weights.

## Prerequisites

1.  **Install Dependencies:**
    It is recommended to first follow the installation instructions in the original `LongCat-Audio-Codec/README.md` to set up the environment and install required packages like `torch`.

2.  **Download Model Files:**
    You must download the pre-trained encoder model and the CMVN statistics file from the [LongCat-Audio-Codec Hugging Face repository](https://huggingface.co/meituan-longcat/LongCat-Audio-Codec).
    *   `LongCatAudioCodec_encoder.pt`
    *   `LongCatAudioCodec_encoder_cmvn.npy` (rename to `cmvn.npy` in this project's root directory).

    Place `LongCatAudioCodec_encoder.pt` in a location of your choice and `cmvn.npy` in the root of this project.

## Usage

The following example demonstrates how to use the `EmbeddingExtractor` to extract features from an audio file.

Create a Python script (e.g., `extract.py`):

```python
import torch
import torchaudio
from encoder import build_VGGtf_encoder # Import the builder function

# --- Configuration ---
CKPT_ENCODER_PATH = "LongCat-Audio-Codec/ckpts/LongCatAudioCodec_encoder.pt"
CMVN_PATH = "cmvn.npy"
TARGET_LAYER = 13 # Example: Extract from the 13th layer (out of 26)
AUDIO_FILE = "LongCat-Audio-Codec/demos/org/common.wav"

# --- 1. Define a corrected EmbeddingExtractor ---
# The original class in encoder.py has a scope issue.
# This corrected version properly assigns the feature_extractor.
class EmbeddingExtractor(torch.nn.Module):
    def __init__(self, ckpt_encoder_path, cmvn_path):
        super().__init__()
        # build_VGGtf_encoder returns both the model and the feature extractor
        self.model, self.feature_extractor = build_VGGtf_encoder(ckpt_encoder_path, cmvn_path)

    def forward(self, wavs: torch.Tensor, wavs_lens: torch.Tensor, n_layer: int):
        wavs = wavs.float()
        # The feature_extractor is now correctly a member of the class
        feats, feat_lens = self.feature_extractor(wavs, wavs_lens)
        # get_mid_emb extracts features from the specified layer
        feats, feat_lens = self.model.get_mid_emb(feats, feat_lens, n_layer)
        # Transpose to (Batch, Time, Dim)
        embed = feats.transpose(0, 1).contiguous()
        return embed, feat_lens

# --- 2. Load Model ---
print("Loading the embedding extractor...")
extractor = EmbeddingExtractor(CKPT_ENCODER_PATH, CMVN_PATH)
extractor.eval()
print("Model loaded.")

# --- 3. Load Audio ---
print(f"Loading audio: {AUDIO_FILE}")
waveform, sample_rate = torchaudio.load(AUDIO_FILE)

# Ensure audio is single-channel and at the correct sample rate (16kHz)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

wavs = waveform
wavs_lens = torch.tensor([wavs.shape[1]], dtype=torch.long)

# --- 4. Extract Features ---
print(f"Extracting features from layer {TARGET_LAYER}...")
with torch.no_grad():
    # The forward pass handles feature computation and model inference
    embeddings, out_lens = extractor(wavs, wavs_lens, n_layer=TARGET_LAYER)

print(f"Successfully extracted features.")
print(f"Output shape: {embeddings.shape}") # (Batch, Time, Dimension)
print(f"Output length: {out_lens.item()}")

# --- 5. Save Features ---
output_path = "output_features.pt"
torch.save(embeddings, output_path)
print(f"Features saved to {output_path}")
```

### Notes on `encoder.py`

The `EmbeddingExtractor` class as originally written in `encoder.py` does not correctly store the `feature_extractor` returned by `build_VGGtf_encoder`. The usage example above includes a corrected class definition that should be used for proper functionality. The workflow is as follows:

1.  The `EmbeddingExtractor` is initialized with paths to the model checkpoint and CMVN file.
2.  It calls `build_VGGtf_encoder` to construct the VGG-Transformer model and the associated feature extractor.
3.  The `forward` method takes a raw waveform, computes Fbank features, and then passes them to the `model.get_mid_emb` method, which runs the encoder up to the specified `n_layer` to extract the continuous embeddings.
