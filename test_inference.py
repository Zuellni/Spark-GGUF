import soundfile as sf

from classes import Bicodec, Spark
from utils import Timer

input = "../../voices/marie.wav"
output = "spark.wav"
text = (
    "She sells seashells by the seashore. "
    "The shells she sells are seashells, I'm sure. "
    "So if she sells seashells on the seashore, then I'm sure she sells seashore shells."
)

with Timer("Loaded bicodec"):
    bicodec = Bicodec("../models/bicodec", "../models/wav2vec2")

with Timer("Loaded spark"):
    spark = Spark("../models/spark/model.q8_0.gguf")

with Timer("Encoded audio"):
    tokens, codes = bicodec.encode(input)

with Timer("Generated audio"):
    codes = spark.generate(text, codes)

with Timer("Decoded audio"):
    audio = bicodec.decode(tokens, codes)

sf.write(output, audio, bicodec.sample_rate)
spark.unload()
