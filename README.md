# Spark-GGUF
Some mildly interesting [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) scripts using python bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Installation
Clone the repo:
```sh
git clone --recursive https://github.com/zuellni/spark-gguf
cd spark-gguf
```

Create a venv:
```sh
python -m venv venv
venv\scripts\activate # windows
source venv/bin/activate # linux
```

Install torch:
```sh
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Install llama-cpp-python:
```sh
pip install llama-cpp-python -C cmake.args="-DGGML_CUDA=1;-DGGML_CUDA_F16=1;-DGGML_CUDA_FA_ALL_QUANTS=1"
```
See the instructions in the [original repo](https://github.com/abetlen/llama-cpp-python) if this fails.

Install other requirements:
```sh
pip install -r requirements.txt
pip install faster-whisper # optional, only if you'd like to use faster whisper
```

## Voice Changer
```sh
python voice_changer.py -a <speaker file> -i <input device id> -o <output device id>
```
If you want to use the output as mic input, you will need to install something like [VB-Cable](https://vb-audio.com/Cable) on Windows, or an equivalent on Linux. Set your real mic as `--input` for the script (run with `--help` to show ids) and `Cable Input` as `--output`. Set `Cable Output` as your default system mic and enable `listen to this device` in sound settings to monitor the output.