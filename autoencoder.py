import torch

class AutoEncoder(torch.nn.Module):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    ELU = 'elu'
    
    
    def __init__(self, params = {}, name='encoder',*args, **kwargs) -> None:
        """_summary_

        Args:
            params dict: dict must contain keys activation and weights.
                Additionally, we assume that weights also contains the input and output dimension

        Raises:
            TypeError: If weghts or activation function is invalid
        """
        super().__init__(*args, **kwargs)
        self.params = params
        
        activation = self.params['activation']
        self.activation_function = self.__get_activation(activation)
        
        self.weights = self.params['weights']
        
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
            layers.append(torch.nn.Linear(curr_weights, next_weights))
            layers.append(self.activation_function)
        
        # Ignore last entry (array ends with activation function, should end with Linear layer)
        self.module = torch.nn.Sequential(layers[:-1])
        
    def __create_decoder(self) -> None:
        layers = []
        for curr_weights, next_weights in reversed(zip(self.weights[:-1], self.weights[1:])):
            layers.append(torch.nn.Linear(curr_weights, next_weights))
            layers.append(self.activation_function)
        
        # Ignore last entry (array ends with activation function, should end with Linear layer)
        self.module = torch.nn.Sequential(layers[:-1])
        
    def __get_activation(self, activation='relu') -> torch.nn.Module:
        match (activation):
            case self.RELU:
                return torch.nn.ReLU()
            case self.SIGMOID:
                return torch.nn.Sigmoid()
            case self.ELU:
                return torch.nn.Sigmoid()
            case _:
                raise TypeError(f"Invalid activation function {activation}")
            
    def forward(self, x):
        return self.module(x)