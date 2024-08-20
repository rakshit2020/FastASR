## whisper-cpp-python



whisper-cpp-python is a Python module inspired by llama-cpp-python that provides a Python interface to the whisper.cpp model. This module automatically parses the C++ header file of the project during building time, generating the corresponding Python bindings.

## Inference Script

```bash
pip install whisper-cpp-python
```
Download the models from [clicking here](https://huggingface.co/ggerganov/whisper.cpp/tree/main)

```python
from whisper_cpp_python import Whisper
whisper = Whisper(model_path= "./ggml-tiny.en.bin")
import time
t1 = time.time()
output = whisper.translate('/path/to/ur/audio')
t2 = time.time()
print("Inference time - ",round((t2-t1),2))
print(output['text'])
```