import soundfile as sf

from models import Codec, Spark
from utils import Timer

with Timer("Loaded codec"):
    codec = Codec()

with Timer("Loaded spark"):
    spark = Spark(file="model.fp32.gguf")

speaker = "path_to_speaker.wav"
text = (
    "She sells seashells by the seashore."
    "The shells she sells are seashells, I'm sure."
    "So if she sells seashells on the seashore, Then I'm sure she sells seashore shells."
)

with Timer("Encoded audio"):
    tokens, inputs = codec.encode(speaker)

with Timer("Generated audio"):
    outputs = spark(text, inputs)

with Timer("Decoded audio"):
    audio = codec.decode(tokens, outputs)

sf.write("test.wav", audio, codec.sample_rate)
spark.unload()
