from typing import Callable, List
import torch
import torch.nn.functional as F


class LinearLayer(torch.nn.Linear):
    """Custom implementation of layer which includes the activation function
    and its derivative
    """
    def __init__(self, in_features: int, out_features: int, activation_function: torch.nn.Module, last:bool = False, order:int = 1, bias: bool = True, device=None, dtype=None) -> None:
        """_summary_

        Args:
            in_features (int): feature in dimension
            out_features (int): feature out dimension
            activation_function (torch.nn.Module): a torch.nn activation function
            last (bool, optional): Flag if this is the last layer, performs different computation in the forward method. Defaults to False.
            bias (bool, optional): _description_. Defaults to True.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.activation_function = activation_function
        self.activation_derivative = self.__get_activation_derivative()
        self.activation_2nd_derivative = self.__get_activation_2nd_derivative()
        self.last = last
        if order == 1:
            self.forward = self.forward_dx
        else:
            self.forward = self.forward_ddx
        
    def forward_dx(self, input: List[torch.Tensor]) -> List[torch.Tensor]:
        """_summary_

        Args:
            input (list): list containing [x, dx]
        """
        x, dx = input

        if self.last:
            x = F.linear(x, self.weight, self.bias)
            dx = F.linear(dx, self.weight, torch.zeros_like(self.bias))
        else:
            x = F.linear(x, self.weight, self.bias)
            dx = self.activation_derivative(x) * F.linear(dx, self.weight, torch.zeros_like(self.bias))
            x = self.activation_function(x)
            
        return [x, dx]
    
    def forward_ddx(self, input: List[torch.Tensor]) -> List[torch.Tensor]:
        """

        Args:
            input (list): list containing [x, dx, ddx]
        """
        x, dx, ddx = input
        
        if self.last:
            x = F.linear(x, self.weight, self.bias)
            dx = F.linear(dx, self.weight, torch.zeros_like(self.bias))
            ddx = F.linear(ddx, self.weight, torch.zeros_like(self.bias))
        else:
            x = F.linear(x, self.weight, self.bias)
            dx_ = F.linear(dx, self.weight, torch.zeros_like(self.bias))
            ddx_ = F.linear(ddx, self.weight, torch.zeros_like(self.bias))
            
            dactivation = self.activation_derivative(x)
            ddactivation = self.activation_2nd_derivative(x, dactivation)
            
            dx =  dactivation * dx_
            ddx = ddactivation * dx_ + dactivation * ddx_
            
            x = self.activation_function(x)
        
        return [x, dx, ddx]
        
        
    def __get_activation_derivative(self) -> Callable:
        if isinstance(self.activation_function, torch.nn.ReLU):
            return lambda x: torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        if isinstance(self.activation_function, torch.nn.ELU):
            return lambda x: torch.minimum(x,torch.exp(x))
        if isinstance(self.activation_function, torch.nn.Sigmoid):
            return lambda x: self.activation_function(x) * (1 - self.activation_function(x)) 
        
    def __get_activation_2nd_derivative(self) -> Callable:
        if isinstance(self.activation_function, torch.nn.ReLU):
            return lambda x, dx=0: 0
        if isinstance(self.activation_function, torch.nn.ELU):
            return lambda x, dx=0: torch.where(x > 0,torch.exp(x), torch.zeros_like(x))
        if isinstance(self.activation_function, torch.nn.Sigmoid):
            return lambda x, dx=0: self.activation_function(dx) * (1 - self.activation_function(dx)) 
        
        
if __name__ == "__main__":
    linear = LinearLayer(2,1, torch.nn.ReLU(), last=True, order=2)
    out = linear([torch.ones(2), torch.ones(2), torch.ones(2)])
    print(out)