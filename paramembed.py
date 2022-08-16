class ParamEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        #    self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(NPARAMS, DIM)
        self.lin2 = nn.Linear(DIM, DIM)
        self.lin3 = nn.Linear(DIM, DIM)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x
