# LongCat-Audio-Semantic

This project extracts continuous semantic features from the intermediate layers of the [LongCat-Audio-Codec](https://github.com/meituan-longcat/LongCat-Audio-Codec) encoder.

These continuous vector representations of audio can be used for downstream tasks like content analysis, audio retrieval, or as inputs for other neural networks.

## Prerequisites

1.  **Install Dependencies:** Follow the installation instructions in the original `LongCat-Audio-Codec/README.md` to set up your environment.

2.  **Download Model Files:** Download the following files from the [LongCat-Audio-Codec Hugging Face repository](https://huggingface.co/meituan-longcat/LongCat-Audio-Codec):
    *   `LongCatAudioCodec_encoder.pt`
    *   `LongCatAudioCodec_encoder_cmvn.npy` (rename this to `cmvn.npy` in this project's root directory).


