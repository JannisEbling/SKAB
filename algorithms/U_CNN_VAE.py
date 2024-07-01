import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


class Conv1D_VAE(nn.Module):
    def __init__(self, input_dim, timesteps, intermediate_dim, latent_dim):
        super(Conv1D_VAE, self).__init__()
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, intermediate_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(intermediate_dim, intermediate_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )

        self.fc1 = nn.Linear(intermediate_dim * timesteps // 4, latent_dim)
        self.fc2 = nn.Linear(intermediate_dim * timesteps // 4, latent_dim)
        self.fc3 = nn.Linear(latent_dim, intermediate_dim * timesteps // 4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                intermediate_dim, intermediate_dim, kernel_size=2, stride=2
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(intermediate_dim, input_dim, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc3(z)
        h = h.view(h.size(0), self.intermediate_dim, self.timesteps // 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.mse_loss(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def train(model, train_loader, epochs=20, learning_rate=1e-3, early_stopping=True):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.permute(0, 2, 1)
            data = Variable(data)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(
            "Epoch: {} Average loss: {:.4f}".format(
                epoch + 1, train_loss / len(train_loader.dataset)
            )
        )

        if (
            early_stopping
            and epoch > 5
            and train_loss / len(train_loader.dataset) > prev_loss
        ):
            print("Early stopping")
            break
        prev_loss = train_loss / len(train_loader.dataset)


def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.permute(0, 2, 1)
            data = Variable(data)
            recon_batch, _, _ = model(data)
            predictions.append(recon_batch.permute(0, 2, 1))
    return torch.cat(predictions).cpu().numpy()


# Example usage:
if __name__ == "__main__":
    import numpy as np

    # Generate some dummy data
    np.random.seed(0)
    torch.manual_seed(0)
    data = np.random.rand(100, 10, 1).astype(np.float32)

    # Convert data to PyTorch tensors and create DataLoader
    tensor_data = torch.tensor(data)
    dataset = TensorDataset(tensor_data, tensor_data)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize and train the model
    input_dim = data.shape[-1]
    timesteps = data.shape[1]
    model = Conv1D_VAE(
        input_dim=input_dim, timesteps=timesteps, intermediate_dim=32, latent_dim=100
    )
    train(model, train_loader, epochs=20, learning_rate=1e-3)

    # Make predictions
    predictions = predict(model, train_loader)
    print(predictions)
