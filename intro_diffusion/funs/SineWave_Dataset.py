from torch.utils.data import Dataset
import torch

class SineWave_DataSet(Dataset):
    def __init__(self, freq_max = 10, n_dim = 200):
        self.sample_generator = self.data_generating
        self.n_dim = n_dim
        self.freq_max = freq_max
        self.n_pts = n_dim

    def data_generating(self):
        with torch.no_grad():
            freq = torch.rand(1) * self.freq_max            
            coord = torch.linspace(0, 2 * torch.pi,  self.n_dim)
            y = torch.sin(coord * freq)

        return freq, y
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        
        return self.sample_generator()
        