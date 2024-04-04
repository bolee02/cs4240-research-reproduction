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
      self.coefficient_mask = torch.from_numpy(self.params['coefficient_mask']).to('cuda')
      self.coefficient_threshold = self.params['coefficient_threshold']
      #initialize sindy coefficients  
      self.sindy_coefficients = torch.zeros((self.library_dim, self.latent_dim))
      self.init_sindy_coefficients()
      
      if self.model_order == 1:
        self.forward = self.forward_dx
      else:
        self.forward = self.forward_ddx


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
    
      self.sindy_coefficients = torch.nn.Parameter(self.sindy_coefficients.to(torch.float64).to('cuda'))

    def forward_dx(self, x, dx, ddx)-> torch.Tensor:

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
        sindy_predict = torch.matmul(Theta, self.coefficient_mask*self.sindy_coefficients)
      else:
        sindy_predict = torch.matmul(Theta, self.sindy_coefficients)

      #decode
      x_decode = self.decoder(torch.cat((z, sindy_predict)))
      dx_decode = x_decode[x_decode.shape[0]//2:]
      x_decode = x_decode[:x_decode.shape[0]//2]

      dz_predict = sindy_predict


      return x, dx, dz_predict, dz, x_decode, dx_decode, self.sindy_coefficients, sindy_predict, torch.zeros(1), torch.zeros(1), torch.zeros(1), z
    
    def forward_ddx(self, x: torch.Tensor, dx: torch.Tensor, ddx: torch.Tensor):
      
      out = self.encoder(torch.cat((x, dx, ddx)))
      slicer = out.shape[0]//3
      z, dz, ddz = out[:slicer], out[slicer:2*slicer], out[2*slicer:] 
      
      #create Theta
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
      out_decode = self.decoder(torch.cat((z, dz, sindy_predict)))
      slicer = out_decode.shape[0]//3
      x_decode, dx_decode, ddx_decode = out_decode[:slicer], out_decode[slicer:2*slicer], out_decode[2*slicer:] 

      

      dz_predict = sindy_predict


      return x, dx, dz_predict, dz, x_decode, dx_decode, self.sindy_coefficients, sindy_predict, ddz, ddx, ddx_decode, z
    