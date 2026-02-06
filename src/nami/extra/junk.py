import torch
import torch.nn as nn

    
class FiLMConditionalField(nn.Module):
    def __init__(self, dim=2, cond_dim=1, hidden=128):
        super().__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
             # output scale and shift
            nn.Linear(hidden, 2 * hidden),
        )
        
        self.main_net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t, c):
        lead_shape = x.shape[:-1]
        
        if t.dim() == 0:
            t_exp = t.expand(*lead_shape).unsqueeze(-1)
        else:
            t_exp = t.unsqueeze(-1).expand(*lead_shape, 1)
        
        # main network
        h = self.main_net(torch.cat([x, t_exp], dim=-1))
        
        # film modulation
        film_params = self.cond_net(c)
        scale, shift = torch.chunk(film_params, 2, dim=-1)
        h = scale * h + shift
        
        return h