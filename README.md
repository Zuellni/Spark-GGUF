# Spark-TTS Scripts
Some mildly interesting [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) scripts using python bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp).

## FAQ
**Q**: How fast?  
**A**: Around 1 second of delay from mic to playback on a 3060 with the `q8_0` quant.

**Q**: Can I run it on RPI4B/Nokia 3310?  
**A**: No, but you can walk it. Very slowly.

**Q**: This is not SOTA as of `<current time>`. Why even bother?  
**A**: At least it's not another implementation of Sesame's rug pull.

**Q**: [FastAPI](https://github.com/fastapi) server for general use?  
**A**: Yes, I'll ask DeepSeek to code it when I have the time.

**Q**: Why not [ExLlamaV2](https://github.com/turboderp-org/exllamav2)?  
**A**: I had issues quantizing the model and it was only generating nonsense.

## Installation
Clone the repo:
```sh
git clone --recursive https://github.com/zuellni/spark-tts-scripts
cd spark-tts-scripts
```

Create a venv:
```sh
python -m venv spark
spark\scripts\activate # windows
. spark/bin/activate # linux
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

## Usage
```sh
python voice_changer.py -i <input device id> -o <output device id> -a <your speaker file>
```
If you want to route the output to a microphone, you will need to install something like [VB-Cable](https://vb-audio.com/Cable) on Windows, or an equivalent on Linux. Set your real microphone as `input` for the script (run without args to show ids) and `Cable Input` as `output`. Set `Cable Output` as your default system microphone and enable `listen to this device` in settings to monitor the output.