import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import gc

# Load model and processor (only once)
model_path = "./EraX-WoW-Turbo-V1.1"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
model.eval()

# Global vars
fs = 16000  # Whisper sample rate
duration = 10  # Max recording duration in seconds
recording = None
recording_thread = None
is_recording = False

# --- UI Setup ---
window = tk.Tk()
window.title("🗣️ Whisper Speech-to-Text")
window.geometry("500x400")

instruction_label = tk.Label(window, text="Press and hold the red button to record", font=("Arial", 12))
instruction_label.pack(pady=10)

output_text = tk.Text(window, wrap="word", height=10, width=60, font=("Arial", 12))
output_text.pack(pady=10)

# --- Record & Transcribe Functions ---
def start_recording(event=None):
    global is_recording, recording
    is_recording = True

    # Clean up any prior audio data
    sd.stop()
    recording = None
    gc.collect()  # optional: clear memory of old recording

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "🎙️ Recording... (Release to transcribe)\n")

    def record():
        global recording
        # Fresh new buffer every time
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

    threading.Thread(target=record).start()


def stop_recording(event=None):
    global is_recording
    if not is_recording:
        return
    is_recording = False

    # Optional delay to ensure buffer closes cleanly
    threading.Timer(0.1, transcribe).start()

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "⏳ Transcribing...\n")

def transcribe():
    global recording
    audio = recording.flatten()

    # Resample if needed (just in case)
    if len(audio) == 0:
        output_text.insert(tk.END, "❌ No audio recorded.")
        return

    inputs = processor(audio, sampling_rate=fs, return_tensors="pt")

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

# --- Keep UI Running ---
window.mainloop()
