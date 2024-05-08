import typing
import os
import os.path
import librosa
import pydub
import numpy as np
from tempfile import TemporaryFile

from .config import default_cache_dir
from .song import Song


class Mixer:
    """Mixer."""

    def __init__(
        self,
        song1: str,
        song2: str,
        cached: bool = False,
        cache_dir: str = default_cache_dir,
        sample_rate: int = 22050,
    ):
        """Create a mixer.

        Args:
            song1: Path of song to play first
            song2: Path of song to play second
            cached: Should songs be cached?
            cache_dir: Cache directory
            sample_rate: new sample rate for audio files
        """
        self.sr = sample_rate
        self.cached = cached
        self.cache_dir = cache_dir
        if cached and not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.song1 = Song(song1, 0, sample_rate, cached, cache_dir)
        self.song2 = Song(song2, 1, sample_rate, cached, cache_dir)

        self.mixed: typing.Optional[typing.Any] = None
        self.mixability = -1

    def load_songs(self):
        self.y1 = self.song1.load()
        self.y2 = self.song2.load()

    def export(self, filename: str = "mixed.mp3"):
        assert filename.endswith(".mp3")
        if self.mixed is None:
            raise RuntimeError("Must mix before exporting")
        self.mixed.export(out_f=filename, format="mp3")

    def analyse(self):
        self.y1 = self.song1.analyse()
        self.y2 = self.song2.analyse()

        # Compute crossfade sizes
        na = self.song1.beats.shape[0] - 1

        scores = [
            sum(self.song1.beats[na - i + 1] * self.song2.beats[i] for i in range(1, t + 1)) / t
            for t in range(2, na // 4)
        ]
        beats = np.argmax(scores) + 2

        in_duration = 1000 * librosa.get_duration(y=self.song1.y, sr=self.sr)
        self.fade_out_start = (
            1000 * librosa.frames_to_time(self.song1.beats, sr=self.sr)[-beats // 2]
        )
        self.fade_out_length = in_duration - self.fade_out_start
        self.fade_in_length = (
            1000 * librosa.frames_to_time(self.song2.beats, sr=self.sr)[beats // 2]
        )

        if self.fade_in_length >= self.fade_out_length:
            self.speed = self.fade_in_length / self.fade_out_length
        else:
            self.speed = self.fade_out_length / self.fade_in_length

        if self.speed > 1.2:
            return

        # Load files
        s1 = pydub.AudioSegment.from_file(self.song1.path, format="mp3")
        s2 = pydub.AudioSegment.from_file(self.song2.path, format="mp3")

        if self.fade_in_length >= self.fade_out_length:
            s1_fade = s1[self.fade_out_start :]
        else:
            s1_fade = s1[self.fade_out_start :].speedup(self.speed)
        if self.fade_in_length <= self.fade_out_length:
            s2_fade = s2[: self.fade_in_length]
        else:
            s2_fade = s2[: self.fade_in_length].speedup(self.speed)

        s1_fade = np.array(s1_fade.get_array_of_samples(), dtype=np.float32)
        s2_fade = np.array(s2_fade.get_array_of_samples(), dtype=np.float32)

        chroma1 = librosa.feature.chroma_cqt(y=s1_fade, sr=self.sr)
        chroma2 = librosa.feature.chroma_cqt(y=s2_fade, sr=self.sr)

        nsamples = min(chroma1.shape[1], chroma2.shape[1])
        chroma1 = chroma1[:, :nsamples]
        chroma2 = chroma2[:, :nsamples]

        norm = np.linalg.norm
        self.mixability = norm(chroma1 * chroma2) / norm(chroma1) / norm(chroma2) / self.speed

    def mix(self):
        if self.speed > 1.2:
            raise ValueError("Too much speed up")

        # Load files
        s1 = pydub.AudioSegment.from_file(self.song1.path, format="mp3")
        s2 = pydub.AudioSegment.from_file(self.song2.path, format="mp3")

        s1_pre = s1[: self.fade_out_start]
        s1_fade = s1[self.fade_out_start :]
        s2_fade = s2[: self.fade_in_length]
        s2_post = s2[self.fade_in_length :]

        out = TemporaryFile()

        if self.fade_in_length >= self.fade_out_length:
            out.write(s1_pre._data)
            xf = s1_fade.fade(to_gain=-120, start=0, end=float("inf"))
        else:
            out.write(s1_pre[: -200 * 9]._data)
            for i in range(9, 0, -1):
                if i == 1:
                    out.write(s1_pre[-200:].speedup(self.speed ** (1 - i / 10))._data)
                else:
                    out.write(
                        s1_pre[-200 * i : -200 * (i - 1)].speedup(self.speed ** (1 - i / 10))._data
                    )
            xf = s1_fade.speedup(self.speed).fade(to_gain=-120, start=0, end=float("inf"))

        if self.fade_in_length <= self.fade_out_length:
            xf *= s2_fade.fade(from_gain=-120, start=0, end=float("inf"))
            out.write(xf._data)
            out.write(s2_post._data)
        else:
            xf *= s2_fade.speedup(self.speed).fade(from_gain=-120, start=0, end=float("inf"))
            out.write(xf._data)
            for i in range(1, 10):
                out.write(
                    s2_post[200 * (i - 1) : 200 * i].speedup(self.speed ** (1 - i / 10))._data
                )
            out.write(s2_post[200 * 9 :]._data)

        out.seek(0)

        self.mixed = s1_pre._spawn(data=out)
