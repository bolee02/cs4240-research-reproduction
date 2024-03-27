import torch


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
    library = [torch.ones(z.size(0))]  # initialize library

    # append state variables
    for i in range(latent_dim):
        library.append(z[:, i])

    # add polynomial terms
    for order in range(2, poly_order + 1):
        poly_add(z, library, order, latent_dim)

    # add sine terms if included
    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:, i]))

    return torch.stack(library, dim=1)


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
        poly_product(z, library, order, i)


def poly_product(z, library, order, index):
    if order > 1:
        for j in range(index, len(library)):
            poly_product(z, library, order - 1, j)
            library.append(torch.prod(z[:, [index, j]], dim=1))
