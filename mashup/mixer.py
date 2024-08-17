import typing
import os
import os.path
import librosa
import pydub
import numpy as np

from .config import default_cache_dir
from .song import Song

mid_gain = -3
fade_size = 2000


def apply_fade_in(clip):
    if clip.duration_seconds < 4 * fade_size / 1000:
        return clip.fade(from_gain=-120, start=0, end=float("inf"))
    out = clip[:fade_size].fade(from_gain=-120, to_gain=mid_gain, start=0, end=float("inf"))
    out += clip[fade_size : -3 * fade_size].apply_gain(mid_gain)
    out += clip[-3 * fade_size :].fade(from_gain=mid_gain, start=0, end=float("inf"))
    return out


def apply_fade_out(clip):
    if clip.duration_seconds < 4 * fade_size / 1000:
        return clip.fade(to_gain=-120, start=0, end=float("inf"))
    out = clip[:fade_size].fade(to_gain=mid_gain, start=0, end=float("inf"))
    out += clip[fade_size : -3 * fade_size].apply_gain(mid_gain)
    out += clip[-3 * fade_size :].fade(from_gain=mid_gain, to_gain=-120, start=0, end=float("inf"))
    return out


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
        self.fade_start: typing.Optional[float] = None
        self.fade_end: typing.Optional[float] = None
        self.song2_fade_end: typing.Optional[float] = None
        self.mixability = -1.0

    def load_songs(self):
        self.song1.load()
        self.song2.load()

    def export(self, filename: str = "mixed.mp3"):
        assert filename.endswith(".mp3")
        if self.mixed is None:
            raise RuntimeError("Must mix before exporting")
        self.mixed.export(out_f=filename, format="mp3")

    def frames_to_secs(self, frames):
        return librosa.frames_to_time(frames, sr=self.sr)

    def analyse(self, compute_mixability: bool = True):
        self.song1.analyse()
        self.song2.analyse()

        assert self.song1.beats is not None
        assert self.song2.beats is not None

        # Compute crossfade sizes
        nb = min(len(self.song1.beats), len(self.song2.beats)) // 3
        b1 = self.song1.beats[-nb:]
        b2 = self.song2.beats[:nb]
        while self.frames_to_secs(min(b1[-1] - b1[0], b2[-1] - b2[0])) > 30:
            b1 = b1[1:]
            b2 = b2[:-1]

        self.fade_out = [
            self.frames_to_secs(b1[0]) * 1000,
            self.frames_to_secs(b1[-1]) * 1000,
        ]
        self.fade_in = [self.frames_to_secs(b2[0]) * 1000, self.frames_to_secs(b2[-1]) * 1000]

        out_len = self.fade_out[1] - self.fade_out[0]
        in_len = self.fade_in[1] - self.fade_in[0]

        self.speed = max(out_len, in_len) / min(out_len, in_len)

        if not compute_mixability or self.speed >= 1.2:
            return

        s1 = pydub.AudioSegment.from_file(self.song1.path, format="mp3")
        s2 = pydub.AudioSegment.from_file(self.song2.path, format="mp3")

        s1_fade = s1[self.fade_out[0] : self.fade_out[1]]
        s2_fade = s2[self.fade_in[0] : self.fade_in[1]]
        if self.speed > 1.01:
            if out_len > in_len:
                s1_fade = s1_fade.speedup(self.speed)
            else:
                s2_fade = s2_fade.speedup(self.speed)

        s1_fade = np.array(s1_fade.get_array_of_samples(), dtype=np.float32)
        s2_fade = np.array(s2_fade.get_array_of_samples(), dtype=np.float32)

        chroma1 = librosa.feature.chroma_cqt(y=s1_fade, sr=self.sr)
        chroma2 = librosa.feature.chroma_cqt(y=s2_fade, sr=self.sr)

        nsamples = min(chroma1.shape[1], chroma2.shape[1])
        chroma1 = chroma1[:, :nsamples]
        chroma2 = chroma2[:, :nsamples]

        b1 = self.frames_to_secs(b1)
        b2 = self.frames_to_secs(b2)
        b1 = [i - b1[0] for i in b1]
        b2 = [i - b2[0] for i in b2]
        if self.speed > 1.01:
            if out_len > in_len:
                b1 = [i / self.speed for i in b1]
            else:
                b2 = [i / self.speed for i in b2]

        beatdiff = sum(abs(i - j) for i, j in zip(b1, b2)) / len(b1)

        norm = np.linalg.norm
        self.mixability = (
            norm(chroma1 * chroma2) / norm(chroma1) / norm(chroma2) / self.speed / (1 + beatdiff)
        )

    def mix(self, shortened: bool = False):
        if self.speed > 1.2:
            raise ValueError("Too much speed up")

        # Load files
        s1 = pydub.AudioSegment.from_file(self.song1.path, format="mp3")
        s2 = pydub.AudioSegment.from_file(self.song2.path, format="mp3")

        s1_pre = s1[: self.fade_out[0]]
        s1_fade = s1[self.fade_out[0] :]
        s2_fade = s2[self.fade_in[0] : self.fade_in[1]]
        s2_post = s2[self.fade_in[1] :]

        out_len = self.fade_out[1] - self.fade_out[0]
        in_len = self.fade_in[1] - self.fade_in[0]

        if in_len >= out_len or self.speed < 1.01:
            if shortened:
                out = s1_pre[-1000:]
            else:
                out = s1_pre
            self.fade_start = out.duration_seconds
        else:
            if shortened:
                out = s1_pre[-2500 : -200 * 9]
            else:
                out = s1_pre[: -200 * 9]
            self.fade_start = out.duration_seconds
            for i in range(9, 0, -1):
                if i == 1:
                    out += s1_pre[-200:].speedup(self.speed ** (1 - i / 10))
                else:
                    out += s1_pre[-200 * i : -200 * (i - 1)].speedup(self.speed ** (1 - i / 10))

        fade_start = out.duration_seconds

        if in_len > out_len and self.speed >= 1.01:
            s2_fade = s2_fade.speedup(self.speed)
        out += apply_fade_in(s2_fade)

        if in_len <= out_len or self.speed < 1.01:
            self.fade_end = out.duration_seconds
            self.song2_fade_end = s2.duration_seconds - s2_post.duration_seconds

            if shortened:
                out += s2_post[:1000]
            else:
                out += s2_post
        else:
            for i in range(1, 10):
                out += s2_post[200 * (i - 1) : 200 * i].speedup(self.speed ** (1 - i / 10))

            self.fade_end = out.duration_seconds
            self.song2_fade_end = s2.duration_seconds - s2_post[200 * 9 :].duration_seconds

            if shortened:
                out += s2_post[200 * 9 : 2500]
            else:
                out += s2_post[200 * 9 :]

        if in_len < out_len and self.speed >= 1.01:
            s1_fade = s1_fade.speedup(self.speed)
        out = out.overlay(apply_fade_out(s1_fade), position=1000 * fade_start)

        self.mixed = out
