from typing import List, Dict
import torch
from autoencoder.linear import LinearLayer

class AutoEncoder(torch.nn.Module):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    ELU = 'elu'
    
    
    def __init__(self, params:Dict = {}, name:str='encoder',*args, **kwargs) -> None:
        """_summary_

        Args:
            params dict: dict must contain keys activation and weights.
                Additionally, we assume that weights also contains the input and output dimension [128, 64, 32, 3]

        Raises:
            TypeError: If weghts or activation function is invalid
        """
        super().__init__(*args, **kwargs)
        self.params = params
        
        activation = self.params['activation']
        self.activation_function = self.__get_activation(activation)
        
        self.weights = [self.params['input_dim']] + self.params['widths'] + [self.params['latent_dim']] 
        self.order = self.params['model_order']
        
        if self.weights is None:
            raise TypeError("Missing weight param")   
        
        if name=='encoder':
            self.__create_encoder()
        elif name=='decoder':
            self.__create_decoder()
        
    def __create_encoder(self) -> None:
        """ Creates the encoder based on weights and activation function
        """
        layers = []
        for curr_weights, next_weights in zip(self.weights[:-1], self.weights[1:]):
            
            layers.append(LinearLayer(curr_weights, next_weights, self.activation_function, len(layers) + 2 == len(self.weights), self.order))
        self.net = torch.nn.Sequential(*layers)
        
    def __create_decoder(self) -> None:
        """Creates decoder, the weights are swapped and reversed compared to the encoder
        """
        layers = []
        for curr_weights, next_weights in zip(reversed(self.weights[1:]), reversed(self.weights[:-1])):
            layers.append(LinearLayer(curr_weights, next_weights, self.activation_function, len(layers) + 2 == len(self.weights), self.order))

        self.net = torch.nn.Sequential(*layers)
        
    def __get_activation(self, activation:str='relu') -> torch.nn.Module:
        match (activation):
            case self.RELU:
                return torch.nn.ReLU()
            case self.SIGMOID:
                return torch.nn.Sigmoid()
            case self.ELU:
                return torch.nn.ELU()
            case _:
                raise TypeError(f"Invalid activation function {activation}")
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward function of the autoencoder

        Args:
            x (List[Tensor]): either the List has 2 or 3 elements
            if it has 2 elements the model order has to be set to 1
            if it has 3 elements the model order has to be set to 2

        Returns:
            List[torch.Tensor]: returns the forward passed list whit the same number of elements as the input
        """
        return self.net(x)
    
        

if __name__ == '__main__':
    params = {'activation': 'elu', 'weights': [4, 3, 2], 'order': 1}

    encoder = AutoEncoder(params=params)
    out = encoder([torch.ones(4), torch.ones(4)])
    print(out)

    #decoder = AutoEncoder(params=params, name='decoder')
    #print(decoder(out))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        encoder.train()  # Set model to training mode
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = encoder(torch.ones((2,4)))
        loss = criterion(outputs)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")
    