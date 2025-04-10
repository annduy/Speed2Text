from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperTokenizer
import torch
import librosa

# Load processor and model
model_path = ".\\EraX-WoW-Turbo-V1.1"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# Load audio file and resample to 16 kHz
audio_path = "sample.wav"  # Replace with your filename
audio, sr = librosa.load(audio_path, sr=16000)

# Prepare inputs
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

# Inference (no gradient needed)
# Generate transcription
with torch.no_grad():
    generated_ids = model.generate(**inputs)

# Decode output
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Print the transcription
print("📝 Transcription:", transcription)
