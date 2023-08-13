import torch
class Config():
    def __init__(self):
        self.n_vocab=6
        self.embed_size=64
        self.hidden_size=64
        self.num_layers=1
        self.dropout=0.7
        self.num_classes=5
        self.pad_size=32
        self.batch_size=16
        self.device=torch.device( 'cpu')
        self.lr=0.001
        self.epochs=1000
