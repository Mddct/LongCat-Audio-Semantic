import torch
import torchaudio
from encoder import EmbeddingExtractor

# --- Configuration ---
CKPT_ENCODER_PATH = "LongCat-Audio-Codec/ckpts/LongCatAudioCodec_encoder.pt"
CMVN_PATH = "cmvn.npy"
TARGET_LAYER = 13 # Example: Extract from the 13th layer (out of 26)
AUDIO_FILE = "LongCat-Audio-Codec/demos/org/common.wav"

# --- 1. Load Model ---
print("Loading the embedding extractor...")
extractor = EmbeddingExtractor(CKPT_ENCODER_PATH, CMVN_PATH)
extractor.eval()
print("Model loaded.")

# --- 2. Load Audio ---
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

# --- 3. Extract Features ---
print(f"Extracting features from layer {TARGET_LAYER}...")
with torch.no_grad():
    # The forward pass handles feature computation and model inference
    embeddings, out_lens = extractor(wavs, wavs_lens, n_layer=TARGET_LAYER)

print(f"Successfully extracted features.")
print(f"Output shape: {embeddings.shape}") # (Batch, Time, Dimension)
print(f"Output length: {out_lens.item()}")

# --- 4. Save Features ---
output_path = "output_features.pt"
torch.save(embeddings, output_path)
print(f"Features saved to {output_path}")