import pickle
import librosa
import os.path

from mash.config import default_cache_dir


class Song:
    def __init__(
        self,
        path: str,
        position: int,
        sample_rate: int = 22050,
        cached: bool = False,
        cache_dir: str = default_cache_dir,
    ):
        assert path.endswith(".mp3")
        self.path = path
        self.name = path.split("/")[-1][:-4]
        self.position = position
        self.cached = cached
        self.cache_dir = cache_dir
        if cached and not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.sr = sample_rate
        self.y = None
        self.tempo = None
        self.beats = None

    def load(self):
        if self.cached and os.path.exists(os.path.join(self.cache_dir, f"{self.name}.pkl")):
            with open(os.path.join(self.cache_dir, f"{self.name}.pkl"), 'rb') as f:
                self.y = pickle.load(f)

        self.y, sr = librosa.load(self.path, sr=self.sr)
        if self.cached:
            with open(os.path.join(self.cache_dir, f"{self.name}.pkl"), 'wb') as f:
                pickle.dump(self.y, f)

    def analyse(self):
        if self.y is None:
            raise RuntimeError("Song must be loaded first")
        self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
