import pickle

import torch

import sys
sys.path.append("/kaggle/input/dl24-group64") #to be changed

from autoencoder.autoencoder import AutoEncoder
from autoencoder.loss import Loss
from sindy_network import SINDy
from tqdm import tqdm
import dill

def train():
    with tqdm(range(num_epochs), postfix=loss_item) as epochs:
        for epoch in epochs:
            sindy.train()  # Set model to training mode
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            x, dx, dz_predict, dz, x_decode, dx_decode, sindy_coefficients, sindy_predict, ddz, ddx, ddx_decode, z = sindy(torch.from_numpy(training_data['x']).to(device=device), torch.from_numpy(training_data['dx']).to(device=device), torch.from_numpy(training_data['ddx']).to(device=device))
            loss = criterion(x, dx, dz, dz_predict, x_decode, dx_decode, sindy_coefficients, ddz, ddx, ddx_decode, sindy.coefficient_mask)
            epochs.set_postfix_str(f"Loss: {loss.item():.4f}")
            # Backward pass
            loss.backward()
            optimizer.step()
            
            ### Every few iterations update coefficient mask, we no longer update the mask during unregularized training
            if criterion.regularization and params['sequential_thresholding'] and (epoch % params['threshold_frequency'] == 0) and (epoch > 0):
                sindy.coefficient_mask = torch.abs(sindy_coefficients) > params['coefficient_threshold']
                print('THRESHOLDING: %d active coefficients' % torch.sum(sindy.coefficient_mask))
                dill.dump(sindy, open(f'model_lorenz_1_{epoch}', 'wb'))
                
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    params = pickle.load(open('/kaggle/input/dl24-group64/utils/rd/model1_params.pkl', 'rb')) #to be changed
    training_data = pickle.load(open('/kaggle/input/dl24-group64/dataset_generation/rd/training_data.pkl', 'rb')) #to be changed
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

    optimizer = torch.optim.Adam(sindy.parameters(), lr=params['learning_rate'])

    loss_item = "Loss:"
    # Training loop
    num_epochs = params['max_epochs']
    train()
    
    ### Train whitout regularization
    num_epochs = params['refinement_epochs']
    criterion.set_regularization(False)
    train()
            
    dill.dump(sindy, open(f'model_lorenz_1', 'wb'))