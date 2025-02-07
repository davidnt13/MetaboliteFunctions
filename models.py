# Additional ML models


import numpy as np
import pandas as pd
import torch
import chemprop
from lightning import pytorch as pl
import math
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping

# ========== #
# PyTorch NN #
# ========== #

class SimplePyTorchDataset(torch.utils.data.Dataset):
    """
    Dataset object for use with PyTorch models.
    """
    def __init__(self, X, y, ids=[]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.id = ids
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNN(torch.nn.Module):
    """
    Simple fully connected MLP in PyTorch.
    """

    def __init__(self, input_size=100, y_scaler=None):
        super().__init__()
        # torch.manual_seed(0)
    
        # (The model definition could also be moved to the forward method)
        self.model = \
            torch.nn.Sequential(torch.nn.Linear(input_size, 
                                                int(input_size/2)), 
                                torch.nn.ReLU(),
                                torch.nn.Linear(int(input_size/2), 
                                                int(input_size/4)),
                                torch.nn.ReLU(),
                                torch.nn.Linear(int(input_size/4), 1), 
                               )
        self.y_scaler = y_scaler
        self.training_loss = []
        self.validation_loss = []
    
    def forward(self, X):
        return self.model(X)
    
    def fit(self, 
            X, y, 
            X_val=None, y_val=None, 
            n_epochs=40, 
            batch_size=100, 
            verbose=True,
            saveLoss = "torch_loss.csv"):
        loss_fn = torch.nn.MSELoss()  # mean square error
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
    
        dataset = SimplePyTorchDataset(X, y)

        batches_per_epoch = math.ceil(len(dataset)/batch_size)
      
        train_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True)
        
        patience = 10
        best_loss = float('inf')
        counter = 0

        for epoch in range(n_epochs):
            epoch_loss = 0

            for X_batch, y_batch in train_loader:

                y_pred = self.forward(X_batch)
                loss = loss_fn(y_pred, y_batch)

                # Remove previous epoch gradients:
                optimizer.zero_grad()    # Backward propagation
                loss.backward()    # Optimize
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= batches_per_epoch
            self.training_loss.append(epoch_loss)

            if (X_val is not None) and (y_val is not None):
                y_val_scaled = \
                    torch.tensor(self.y_scaler.transform(
                                     np.array(y_val).reshape(-1, 1)),
                                 dtype=torch.float32)
                y_val_pred = self.forward(X_val)
                val_loss = loss_fn(y_val_pred, 
                               y_val_scaled)
                self.validation_loss.append(loss.item())
            
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    counter = 0  # Reset patience counter if validation loss improves
                else:
                    counter += 1  # Increment counter if no improvement

                # If patience limit is reached, stop training
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch} due to no improvement.")
                    break

            if verbose:
                print(f'Epoch: {epoch:4d}, Loss: {epoch_loss:.3f}')
        
        if (saveLoss != ''):
            lossResults = pd.DataFrame(data = zip(self.training_loss, self.validation_loss), columns = ["Training Loss", "Validation Loss"])
            lossResults.to_csv(saveLoss)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        y_pred = self.forward(X).detach().numpy()
        # Unscale y data:
        if self.y_scaler is not None:
            y_pred = self.y_scaler.inverse_transform(y_pred)
        return y_pred.squeeze()

    def plot_training_loss(self):
        plt.plot(range(len(self.training_loss)), self.training_loss)
        plt.plot(range(len(self.validation_loss)), self.validation_loss)
        plt.show()


# ======== #
# Chemprop #
# ======== #

class ChempropModel():
    """
    Wrapper to use chemprop model in place of sklearn models in scripts.
    """

    def __init__(self,
                 y_scaler=None,
                 max_epochs=40,
                 **kwargs):

        # Set up model:
        mp = chemprop.nn.BondMessagePassing()
        agg = chemprop.nn.MeanAggregation()
        ffn = chemprop.nn.RegressionFFN()
        if y_scaler is not None:
            output_transform = \
                chemprop.nn.UnscaleTransform.from_standard_scaler(y_scaler)
        else:
            output_transform = None
        ffn = chemprop.nn.RegressionFFN(input_dim=mp.output_dim+desc_size, output_transform=output_transform)
        self.mpnn = chemprop.models.MPNN(mp, agg, ffn, 
                                         #batch_norm, metric_list)
                                         )
        
        early_stopping = EarlyStopping(
                monitor = 'val_loss',
                patience = 10,
                mode = 'min',
                verbose = True
                )

        # Set up pytorch trainer:
        self.trainer = pl.Trainer(logger=False,
                                  # Save model checkpoints in the 
                                  # "checkpoints" folder:
                                  enable_checkpointing=True,
                                  enable_progress_bar=True,
                                  accelerator="auto",
                                  devices=1,
                                  max_epochs=max_epochs,
                                  #callbacks = [early_stopping]
                                 )

    def fit(self,
            train_loader,
            #val_loader=None,
            *args,
            **kwargs):
        self.trainer.fit(self.mpnn, train_loader) #, val_loader)

    def predict(self,
                test_loader):
        return np.concatenate(self.trainer.predict(self.mpnn, test_loader), 
                              axis=0).squeeze()
