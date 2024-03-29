from itertools import combinations_with_replacement
import math
import torch
from scipy.special import binom

def sindy_library_pt(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library for a first order system.

    Arguments:
        z - 2D torch tensor of the snapshots on which to build the library. Shape is number of
            time points by the number of state variables.
        latent_dim - Integer, number of state variables in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D torch tensor containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [torch.ones(z.size(0)).to(device='cuda')]  # initialize library
    # append state variables

    sample_list = range(latent_dim)
    list_combinations = list()

    for n in range(1, poly_order+1):
        list_combinations = list(combinations_with_replacement(sample_list, n))
        for combination in list_combinations:
            library.append(torch.prod(z[:, combination], dim=1).to(device='cuda'))

    # add sine terms if included
    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:, i]))

    return torch.stack(library, dim=1).to(device='cuda')

def library_size(latent_dim, poly_order):
    f = lambda latent_dim, poly_order: math.comb(latent_dim + poly_order -1, poly_order)
    total = 0
    for i in range(poly_order+1):
        total += f(latent_dim, i)
    return total
    
def sindy_library_pt_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library for a second-order system.

    Can combine this function and the above one when fully implemented.

    Arguments:
        z - 2D torch tensor of the snapshots on which to build the library. Shape is number of
            time points by the number of state variables.
        dz - 2D torch tensor of the time derivatives of the snapshots.
        latent_dim - Integer, number of state variables in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D torch tensor containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [torch.ones(z.size(0))]  # initialize library

    # concatenate z and dz
    z_combined = torch.cat([z, dz], dim=1)

    # append state variables and combinations
    for i in range(2 * latent_dim):
        library.append(z_combined[:, i])

    # add polynomial terms
    for order in range(2, poly_order + 1):
        poly_add(z_combined, library, order, 2 * latent_dim)

    # add sine terms if included
    if include_sine:
        for i in range(2 * latent_dim):
            library.append(torch.sin(z_combined[:, i]))

    return torch.stack(library, dim=1)


def poly_add(z, library, order, latent_dim):
    for i in range(latent_dim):
        poly_product(z, library, order, i, latent_dim)

def poly_product(z, library, order, index, latent_dim, seen_combinations=None):
    if seen_combinations is None:
        seen_combinations = set()

    if order > 1:
        for j in range(index, latent_dim):
            combination = tuple(sorted((index, j)))  # Using a tuple to make combinations hashable
            if combination not in seen_combinations:
                seen_combinations.add(combination)
                poly_product(z, library, order - 1, j, latent_dim, seen_combinations)
                library.append(torch.prod(z[:, [index, j]], dim=1))
# def poly_product(z, library, order, index, latent_dim):
#     if order > 1:
#         for j in range(index, latent_dim):  
#             poly_product(z, library, order - 1, j, latent_dim)
#             library.append(torch.prod(z[:, [index, j]], dim=1))


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l

def cust_sindy(z, latent_dim, poly_order, include_sine=False):
    library = torch.zeros_like(z)
    library[:,:] = z
    print(z.shape)
    dim = z.shape[0]
    
    if poly_order > 1:
        library = torch.cat((library, z @ z.repeat(latent_dim, 1).T))
    return library
    

def sindy_library_tf6(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [torch.ones(z.shape[0])]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(z[:,i]* z[:,j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, axis=1)