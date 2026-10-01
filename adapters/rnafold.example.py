from mitaptamer.integrations import RNAfoldBackend


class LocalRNAfold(RNAfoldBackend):
    def __init__(self, **settings):
        self.settings = settings

    def fold(self, sequences):
        raise NotImplementedError("Connect RNAfold here, or configure mitaptamer.integrations:RNAfoldCLI")
