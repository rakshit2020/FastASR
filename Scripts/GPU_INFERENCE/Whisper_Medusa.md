## Whisper Medusa

Whisper is an advanced encoder-decoder model for speech transcription and translation, processing audio through encoding and decoding stages. Given its large size and slow inference speed, various optimization strategies like Faster-Whisper and Speculative Decoding have been proposed to enhance performance. Our Medusa model builds on Whisper by predicting multiple tokens per iteration, which significantly improves speed with small degradation in WER. We train and evaluate our model on the LibriSpeech dataset, demonstrating speed improvements.


## Inference Script
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```
Then install the package:
```bash
git clone https://github.com/aiola-lab/whisper-medusa.git
cd whisper-medusa
pip install -e .
```
```python
import torch
import torchaudio
from whisper_medusa import WhisperMedusaModel
from transformers import WhisperProcessor

class ASRInferenceMedusa:
    def __init__(self, model_name="aiola/whisper-medusa-v1", device=None, sampling_rate=16000):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate

        self.model = WhisperMedusaModel.from_pretrained(model_name).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def load_and_process_audio(self, audio_path):
        input_speech, sr = torchaudio.load(audio_path)
        if input_speech.shape[0] > 1:  # If stereo, average the channels
            input_speech = input_speech.mean(dim=0, keepdim=True)

        if sr != self.sampling_rate:
            input_speech = torchaudio.transforms.Resample(sr, self.sampling_rate)(input_speech)

        input_features = self.processor(input_speech.squeeze(), return_tensors="pt", sampling_rate=self.sampling_rate).input_features
        return input_features.to(self.device)

    def run_inference(self, audio_path, language="en"):
        input_features = self.load_and_process_audio(audio_path)
        model_output = self.model.generate(input_features, language=language)
        predict_ids = model_output[0]
        pred = self.processor.decode(predict_ids, skip_special_tokens=True)
        print(pred)
        return pred

# Example usage
if __name__ == "__main__":
    audio_path = "path/to/audio.wav"
    asr_inference = ASRInferenceMedusa(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    asr_inference.run_inference(audio_path)

```