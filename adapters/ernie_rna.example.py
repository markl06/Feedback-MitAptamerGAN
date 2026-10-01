from mitaptamer.integrations import ErnieRNAEncoder


class LocalErnieRNA(ErnieRNAEncoder):
    def __init__(self, checkpoint, embedding_dim=768, device="cpu", pooling="mean"):
        self.embedding_dim = embedding_dim
        self.checkpoint = checkpoint
        self.device = device
        self.pooling = pooling
        raise NotImplementedError("Connect your local ERNIE-RNA implementation here")

    def encode(self, sequences):
        raise NotImplementedError("Return actual ERNIE-RNA features; do not return placeholder zeros")
