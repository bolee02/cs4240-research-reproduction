import pickle

import torch

from autoencoder.autoencoder import AutoEncoder
from autoencoder.loss import Loss
from sindy_network import SINDy
from tqdm import tqdm

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    params = pickle.load(open('utils/model1_params.pkl', 'rb'))
    training_data = pickle.load(open('dataset_generation/training_data.pkl', 'rb'))

    encoder = AutoEncoder(params, 'encoder')
    decoder = AutoEncoder(params, 'decoder')
    sindy = SINDy(encoder, decoder, params)

    sindy = sindy.to(device=device)
    
    criterion = Loss(
        params['loss_weight_decoder'],
        params['loss_weight_sindy_z'],
        params['loss_weight_sindy_x'],
        params['loss_weight_sindy_regularization']
        )

    optimizer = torch.optim.Adam(sindy.parameters(), lr=0.01)

    loss_item = "Loss:"
    # Training loop
    num_epochs = params['max_epochs']
    with tqdm(range(num_epochs), postfix=loss_item) as epochs:
        for epoch in epochs:
            sindy.train()  # Set model to training mode
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            x, dx, dz, dz_pred, x_decode, dx_decode, sindy_coefficients, sindy_predict = sindy(torch.from_numpy(training_data['x']).to(device=device), torch.from_numpy(training_data['dx']).to(device=device),0)
            loss = criterion(x, dx, dz, dz_pred, x_decode, dx_decode, sindy_coefficients)
            epochs.set_postfix_str(f"Loss: {loss.item():.4f}")
            # Backward pass
            loss.backward()
            optimizer.step()
            
    loss.set_regularization(False)
    refinement_epochs = params['refinement_epochs']    
    with tqdm(range(refinement_epochs), postfix=loss_item) as epochs:
        for epoch in epochs:
            sindy.train()  # Set model to training mode
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            x, dx, dz, dz_pred, x_decode, dx_decode, sindy_coefficients, sindy_predict = sindy(torch.from_numpy(training_data['x']).to(device=device), torch.from_numpy(training_data['dx']).to(device=device),0)
            loss = criterion(x, dx, dz, dz_pred, x_decode, dx_decode, sindy_coefficients)
            epochs.set_postfix_str(f"Loss: {loss.item():.4f}")
            # Backward pass
            loss.backward()
            optimizer.step()