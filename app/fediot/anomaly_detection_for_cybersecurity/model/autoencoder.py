import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, output_dim=115):
        super(AutoEncoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(output_dim, round(output_dim * 0.75)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.75), round(output_dim * 0.50)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.50), round(output_dim * 0.33)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.33), round(output_dim * 0.25)),
        )
        self.dec = nn.Sequential(
            nn.Linear(round(output_dim * 0.25), round(output_dim * 0.33)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.33), round(output_dim * 0.50)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.50), round(output_dim * 0.75)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.75), output_dim),
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode
