class AudioEmbeddingToParams(nn.Module):
    def __init__(self):
        super().__init__()
        #    self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(DIM, DIM)
        self.lin2 = nn.Linear(DIM, DIM)
        self.lin3 = nn.Linear(DIM, NPARAMS)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, DIM)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        # Want 0/1 range
        x = self.sigmoid(x)
        return x
