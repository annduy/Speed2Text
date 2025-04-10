import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torchaudio
import time
import gc

# Load model and processor
model_path = "./EraX-WoW-Turbo-V1.1"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# Settings
fs = 16000
duration = 10  # max duration in seconds
is_recording = False
audio_buffer = None


def transcribe():
    global audio_buffer
    if audio_buffer is None:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "⚠️ No audio captured.\n")
        return

    audio = np.squeeze(audio_buffer)
    inputs = processor(audio, sampling_rate=fs, return_tensors="pt")

    # 🟡 Add forced language
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            forced_decoder_ids=forced_decoder_ids
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"📝 {text}\n")
    


def start_recording(event=None):
    global is_recording, audio_buffer
    is_recording = True
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "🎙️ Recording... (Release to transcribe)\n")

    def record_audio():
        global audio_buffer
        audio_buffer = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Make sure recording is finished before proceeding

    threading.Thread(target=record_audio).start()


def stop_recording(event=None):
    global is_recording
    if not is_recording:
        return
    is_recording = False
    

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "⏳ Transcribing...\n")
    def delayed_transcribe():
        time.sleep(0.1)  # slight delay to ensure audio_buffer is ready
        transcribe()

    threading.Thread(target=delayed_transcribe).start()

# Build GUI
root = tk.Tk()
root.title("🎤 Speech-to-Text Transcriber")
root.geometry("500x300")

record_btn = tk.Button(root, text="● Hold to Talk", font=("Arial", 16), bg="red", fg="white", width=20)
record_btn.pack(pady=30)
record_btn.bind("<ButtonPress-1>", start_recording)
record_btn.bind("<ButtonRelease-1>", stop_recording)

output_text = tk.Text(root, height=5, font=("Arial", 14))
output_text.pack(pady=20)

root.mainloop()
