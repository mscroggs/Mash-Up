import os
import os.path
import typing
import librosa
import pydub
import numpy as np
from tempfile import TemporaryFile
import pickle, json, os

from mash.song import Song


class Mixer:
    """Mixer."""

    def __init__(
        self,
        song1: str,
        song2: str,
        cached: bool = False,
        cache_dir: typing.Optional[str] = None,
        sample_rate: int = 22050,
    ):
        """Create a mixer.

        Args:
            song1: Path of song to play first
            song2: Path of song to play second
            cached: Should *TODO: something* be cached
            cache_dir: Cache directory
            sample_rate: new sample rate for audio files
        """
        self.best_score = None
        self.song1 = Song(song1)
        self.song2 = Song(song2)
        self.songs = [Song(song1), Song(song2)]
        self.sr = sample_rate

        self.cached = cached
        if cached:
            if cache_dir is None:
                self.cache_dir = os.path.join(
                    os.path.join(os.path.expanduser("~"), ".cache"),
                    "mix")
            else:
                self.cache_dir = cache_dir
            if not os.path.isdir(self.cache_dir):
                os.mkdir(self.cache_dir)

        self.Yin = []
        self.Yout = []
        self.pathIn = []
        self.pathOut = []
        self.beats = {'in': [], 'out': []}
        self.tempo = {'in': 0, 'out': 0}

        self._load()
        self._extract()
        self._segment()
        self._speedUp()

    def mix(self):
        self.out = self._mix()

    def export(self, filename: str = "mixed.mp3"):
        assert filename.endswith(".mp3")
        print("Exporting...")
        self.out.export(out_f=filename, format="mp3")
        print("[SUCCESS] Export as `final.mp3`")

    def _load(self):
        for i, song in enumerate(self.songs):
            if self.cached and os.path.exists(os.path.join(self.cache_dir, f"{song.name}.pkl")):
                print("\nLoading", song.name, "from cache")
                with open(os.path.join(self.cache_dir, f"{song.name}.pkl"), 'rb') as f:
                    if i == 0:
                        print("Yin=", song.name)
                        self.Yin = pickle.load(f)
                        self.pathIn = song.path
                    else:
                        print("Yout=", song.name)
                        self.Yout.append(pickle.load(f))
                        self.pathOut.append(song.path)
            else:
                print("\nLoading", song.name)
                y, sr = librosa.load(song.path, sr=self.sr)
                if i == 0:
                    self.Yin = y
                    self.pathIn = song.path
                else:
                    self.Yout.append(y)
                    self.pathOut.append(song.path)
                print("[SUCCESS] Loaded", song.name)

                if self.cached:
                    try:
                        with open(os.path.join(self.cache_dir, f"{song.name}.pkl"), 'wb') as f:
                            pickle.dump(y, f)
                            print("[SUCCESS] Cached", song.name)
                    except Exception as e:
                        print("[FAILED] Caching", song.name)
                        print(e)

    def _extract(self):
        # TODO: Add cosine distance similarity to choose the best mixout
        self.Yout = self.Yout[0] # NOTE: considering 1mixin & 1mixout
        self.pathOut = self.pathOut[0]

        self.tempo['in'], self.beats['in'] = librosa.beat.beat_track(y=self.Yin, sr=self.sr)
        self.tempo['out'], self.beats['out'] = librosa.beat.beat_track(y=self.Yout, sr=self.sr)

        print("TempoIn=", self.tempo['in'])
        print("TempoOut=", self.tempo['out'])

        self.otac()
        self._crossFadeRegion()

    def otac(self): # Optimal Tempo Adjustment Coefficient Computation
        C = [-2, -1, 0, 1, 2]

        if self.tempo['in'] == self.tempo['out']:
            self.tempo['tgt'] = self.tempo['in']
            return

        Tin_ = [(2**c)*self.tempo['in'] for c in C]
        TinIndex_ = np.argmin(np.absolute(Tin_ - self.tempo['out']))
        Copt = C[TinIndex_]
        Bopt = (2**Copt)*self.tempo['in']

        Tlow = min(Bopt, self.tempo['out'])
        Thigh = max(Bopt, self.tempo['out'])

        a, b = 0.765, 1
        Ttgt = (a-b)*Tlow + np.sqrt( ((a-b)**2)*(Tlow**2) + 4*a*b*Thigh*Tlow )
        Ttgt = Ttgt/(2*a)

        print("FoptIn=", Ttgt/Bopt)
        print("FoptOut=", Ttgt/self.tempo['out'])
        print("Ttgt=", Ttgt)

        self.tempo['tgt'] = Ttgt
        return Ttgt

    def _crossFadeRegion(self): # Computes the cross fade region for the mixed song
        Na = self.beats['in'].shape[0]-1

        scores = [self._score(i, Na) for i in range(2, int(Na/4))]
        noBeats = np.argmax(scores)+2

        inDuration = librosa.get_duration(y=self.Yin, sr=self.sr)
        fadeInStart = librosa.frames_to_time(self.beats['in'], sr=self.sr)[-int(noBeats/2)]
        fadeIn = inDuration - fadeInStart

        fadeOut = librosa.frames_to_time(self.beats['out'], sr=self.sr)[int(noBeats/2)]

        self.best_score = np.max(scores)
        print("Best Power Corelation Scores=", self.best_score)
        print("Number of beats in cross fade region=", noBeats)
        print("fadeInStart=", fadeInStart)
        print("fadeOutEnd=", fadeOut)
        print("Cross Fade Time=", fadeIn+fadeOut)

        self.crossFade = [fadeInStart*1000, fadeOut*1000] # In milliseconds


    def _score(self, T, Na):
        cr = 0
        for i in range(1, T+1):
            cr += self.beats['in'][Na-i+1]*self.beats['out'][i]
        return cr/T

    def _segment(self):
        print("Started Segmentation")

        sIn = pydub.AudioSegment.from_file(self.pathIn, format="mp3")
        sOut = pydub.AudioSegment.from_file(self.pathOut, format="mp3")

        print("[SUCCESS] Segmented audio files")

        self.segments = {
            'in': [ sIn[:self.crossFade[0]], sIn[self.crossFade[0]:] ],
            'out': [ sOut[:self.crossFade[1]], sOut[self.crossFade[1]:] ],
        }
        del sIn, sOut

    def _speedUp(self):
        s1 = self.segments['in'][1]
        s2 = self.segments['out'][0]

        speed1 = self.tempo['tgt']/self.tempo['in']
        speed2 = self.tempo['tgt']/self.tempo['out']

        print("Playback Speed of in end segment=",speed1,'X')
        print("Playback Speed of out start segment=",speed2,'X')

        s1 = s1.speedup(playback_speed=speed1)
        s2 = s1.speedup(playback_speed=speed2)

    def _mix(self):
        xf = self.segments['in'][1].fade(to_gain=-120, start=0, end=float('inf'))
        xf *= self.segments['out'][0].fade(from_gain=-120, start=0, end=float('inf'))

        out = TemporaryFile()

        out.write(self.segments['in'][0]._data)
        out.write(xf._data)
        out.write(self.segments['out'][1]._data)

        out.seek(0)

        print("[SUCCESS] Mixed 4 audio segment to 1")
        return self.segments['in'][0]._spawn(data=out)
