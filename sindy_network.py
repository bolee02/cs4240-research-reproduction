from typing import Dict
import torch
from sindy.SINDy_library import *
from autoencoder.autoencoder import AutoEncoder
import math


class SINDy(torch.nn.Module):
    """
    SINDy dz predictions

    Arguments:
        params - Dictionary object containing the parameters that specify the training.
        See params.txt file for a description of the parameters.

    Returns:
        sindy_predict - tensor containing sindy's predictions for dz.
    """


    def __init__(self, encoder:AutoEncoder, decoder:AutoEncoder, params:Dict = {}, *args, **kwargs) -> None:
      super().__init__(*args, **kwargs) 
      self.params = params
      self.encoder = encoder
      self.decoder =decoder

      self.input_dim = self.params['input_dim']
      self.latent_dim = self.params['latent_dim']
      self.poly_order = self.params['poly_order']
      if 'include_sine' in self.params.keys():
        self.include_sine = self.params['include_sine']
      else:
        self.include_sine = False
      self.library_dim = self.params['library_dim']
      self.model_order = self.params['model_order']
      self.sequential_thresholding = self.params['sequential_thresholding']
      self.coefficient_initialization = self.params['coefficient_initialization']
      self.coefficient_mask = self.params['coefficient_initialization']
      self.coefficient_threshold = self.params['coefficient_threshold']
      #initialize sindy coefficients  
      self.sindy_coefficients = torch.zeros(12, self.latent_dim)#(library_size(self.latent_dim, self.poly_order), self.latent_dim))
      self.init_sindy_coefficients()


    def init_sindy_coefficients(self, name='normal', std=1., k=3):

      #self.sindy_coefficients = std*torch.randn_like(self.sindy_coefficients)
 
      if name == 'xavier':
        self.sindy_coefficients = torch.nn.init.xavier_uniform_(self.sindy_coefficients)
      elif name == 'uniform':
        self.sindy_coefficients = torch.nn.init.uniform_(self.sindy_coefficients, low=0.0, high=1.0)
      elif name == 'constant':
        self.sindy_coefficients = torch.ones_like(self.sindy_coefficients)*k
      elif name == 'normal':
        self.sindy_coefficients = torch.nn.init.normal_(self.sindy_coefficients, mean=0, std=std) 
    
      self.sindy_coefficients = self.sindy_coefficients.to(torch.float64)#.to(device='cuda')

    def forward(self, x, dx, ddx)-> torch.Tensor:

      #x = x.reshape(1, *x.shape)
      #dx = dx.reshape(1, *dx.shape)
      
      z = self.encoder(torch.cat((x, dx)))
      dz = z[z.shape[0]//2:]
      z = z[:z.shape[0]//2]
      
      #create Theta
      if self.model_order == 1:
        Theta = sindy_library_pt(z, self.latent_dim, self.poly_order, self.include_sine)
      else:
        Theta = sindy_library_pt_order2(z, dz, self.latent_dim, self.poly_order, self.include_sine)
      #apply thresholding or not
      if self.sequential_thresholding:
        '''
        tmp = torch.rand(size=(library_dim,latent_dim), dtype=torch.float32)
        mask = torch.zeros_like(tmp)
        mask = mask.where(self.coefficient_mask, tmp)
        '''
        mask = torch.where(self.sindy_coefficients > self.coefficient_threshold, self.sindy_coefficients, 0)
        sindy_predict = torch.matmul(Theta, mask*self.sindy_coefficients)
      else:
        sindy_predict = torch.matmul(Theta, self.sindy_coefficients)

      #decode
      x_decode = self.decoder(torch.cat((z, dz)))
      dx_decode = x_decode[x_decode.shape[0]//2:]
      x_decode = x_decode[:x_decode.shape[0]//2]

      if self.model_order == 1:
       dz_predict = sindy_predict
       dzz_predict = None
      else:
       ddz_predict = sindy_predict
       dz_predict = None

      return x, dx, dz_predict, dz, x_decode, dx_decode, self.sindy_coefficients, sindy_predict