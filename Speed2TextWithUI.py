import tkinter as tk
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import librosa
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Load model & processor
model_path = "./EraX-WoW-Turbo-V1.1"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
model.eval()

# Globals for recording
fs = 16000  # Sample rate
recording = []
is_recording = False

# --- UI Setup ---
window = tk.Tk()
window.title("🗣️ Whisper Speech-to-Text")
window.geometry("500x400")

# Instructions
instruction_label = tk.Label(window, text="Press and hold the red button to record", font=("Arial", 12))
instruction_label.pack(pady=10)

# Transcription output (text box)
output_text = tk.Text(window, wrap="word", height=10, width=60, font=("Arial", 12))
output_text.pack(pady=10)

# --- Recording Functions ---
def start_recording(event=None):
    global recording, is_recording
    is_recording = True
    recording = []

    def callback(indata, frames, time, status):
        if is_recording:
            recording.append(indata.copy())

    threading.Thread(target=lambda: sd.InputStream(samplerate=fs, channels=1, callback=callback).start()).start()

def stop_recording(event=None):
    global is_recording
    is_recording = False

    if not recording:
        return

    # Combine audio chunks
    audio_np = np.concatenate(recording, axis=0).flatten()

    # Clear text and show loading
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "⏳ Transcribing...\n")

    threading.Thread(target=lambda: transcribe(audio_np)).start()

def transcribe(audio_np):
    inputs = processor(audio_np, sampling_rate=fs, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, transcription)

# --- Record Button ---
record_button = tk.Canvas(window, width=100, height=100, bg="white", highlightthickness=0)
circle = record_button.create_oval(10, 10, 90, 90, fill="red")
record_button.pack(pady=10)

record_button.bind("<ButtonPress-1>", start_recording)
record_button.bind("<ButtonRelease-1>", stop_recording)

# --- Start UI ---
window.mainloop()
