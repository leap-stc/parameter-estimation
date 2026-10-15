from torch.utils.data import Dataset
import torch


class SineWave_DataSet(Dataset):
    def __init__(self, freq_max = 10, n_dim = 200, random_bias = True):
        self.sample_generator = self.data_generating
        self.n_dim = n_dim
        self.freq_max = freq_max
        self.n_pts = n_dim
        self.random_bias = random_bias

    def data_generating(self):
        with torch.no_grad():
            freq = torch.rand(1) * self.freq_max            
            coord = torch.linspace(0, 2 * torch.pi,  self.n_dim)
            y = torch.sin(coord * freq)

            if self.random_bias:
                y = y + torch.randn(1)
        
        return freq, y
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        
        return self.sample_generator()






class SineWave_Random_DataSet(Dataset):
    def __init__(self, freq_max = 10, n_dim = 200, random_phase = False, random_bias = True):
        self.sample_generator = self.data_generating
        self.n_dim = n_dim
        self.freq_max = freq_max
        self.n_pts = n_dim
        self.random_phase = random_phase
        self.random_bias = random_bias
    def data_generating(self):
        with torch.no_grad():
            freq = torch.rand(1) * self.freq_max 
            phase = torch.rand(1) * 2 * torch.pi
            bias = torch.randn(1)
            coord = torch.linspace(0, 2 * torch.pi,  self.n_dim)
            if self.random_phase:
                y = torch.sin(coord * freq + phase)
            else:
                y = torch.sin(coord * freq)

            if self.random_bias:
                y = y + bias
            

        return freq, y, phase, bias
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        
        return self.sample_generator()



