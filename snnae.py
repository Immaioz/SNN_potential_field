import torch, torch.nn as nn
import snntorch as snn

# === Modello SNN ===
class SAE(nn.Module):
    def __init__(self,num_inputs,num_hidden, num_outputs, num_steps=25,beta=0.95):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_steps = num_steps
        self.beta = beta
        
        self.fc1 = nn.Linear(self.num_inputs,self.num_hidden)
        self.lif1 = snn.Leaky(beta=self.beta)
        self.fc2 = nn.Linear(self.num_hidden, self.num_outputs)
        self.lif2 = snn.Leaky(beta=self.beta)

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    
    def compute_reconstruction_error(self,sample):
        self.eval()
        with torch.no_grad():
            spk,mem = self(sample)
            mem = torch.mean(mem,axis=0)
            mse = torch.nn.MSELoss()
            return mse(sample,mem).item()
    



import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # puoi usare ReLU o altra attivazione
        x = F.relu(self.fc2(x))  # opzionale, dipende dal task
        return x

    # Reconstruction error MSE per un singolo batch/sample
    def compute_reconstruction_error(model, sample):
        model.eval()
        with torch.no_grad():
            recon = model(sample)
            mse = torch.nn.MSELoss()
            return mse(sample, recon).item()

    # Errore di ricostruzione per ogni esempio nel batch
    def compute_reconstructions(model, sample):
        model.eval()
        with torch.no_grad():
            recon = model(sample)
            error = torch.square(sample - recon)
            error = torch.mean(error, dim=1)  # media su features
            return error  # tensor di shape (batch_size,)
