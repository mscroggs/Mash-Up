class Song:
    def __init__(self, path: str):
        assert path.endswith(".mp3")
        self.path = path
        self.name = path.split("/")[-1][:-4]
