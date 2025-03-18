import soundfile as sf
from semantic_text_splitter import TextSplitter

from models import Codec, Spark
from utils import Logger, Timer

with Timer("Loaded codec"):
    codec = Codec(dtype="float32")

with Timer("Loaded spark"):
    spark = Spark(dtype="f32")

speaker = "path_to_speaker.wav"
text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it. 'And what is the use of a book,' thought Alice 'without pictures or conversation?' So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her."

splitter = TextSplitter(300)
chunks = []

with Timer("Encoded audio"):
    tokens, tokens_str = codec.encode(speaker)

with Timer("Generated audio"):
    for chunk in splitter.chunks(text):
        Logger.info(chunk)
        chunk = spark(chunk, tokens_str)
        chunks.append(chunk)

with Timer("Decoded audio"):
    audio = codec.decode(tokens, "".join(chunks))

sf.write("test.wav", audio, codec.sample_rate)
spark.unload()
