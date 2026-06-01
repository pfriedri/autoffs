"""
This script implements a PyTorch only, fully-differentiable 3D B-spline interpolation.
"""

import torch

def bspline_basis(x):
    """
    Defines a centered cubic B-spline basis function (symmetric and centered around 0).
            1/6 (4 - 6x² + 3|x|³)  if |x| < 1
    B(x) =  1/6 (2 - |x|)³         if 1 <= |x| < 2
            0                      otherwise
    """
    absx = x.abs()
    res = torch.zeros_like(x)

    mask1 = absx < 1
    mask2 = (absx >= 1) & (absx < 2)

    absxmask1 = absx[mask1]
    res[mask1] = (4 - (6 - 3 * absxmask1) * absxmask1 ** 2)
    res[mask2] = ((2 - absx[mask2]) ** 3)
    res /= 6.0
    return res

def bspline_basis_derivative(x):
    """
    First derivative of the centered cubic B-spline basis function B(x).
                1/6 * (-12x + 9x|x|)                if |x| < 1
    dB(x)/dx =  1/6 * -3 * (2 - |x|)² * sign(x)     if 1 <= |x| < 2
                0                                   otherwise
    """
    absx = x.abs()
    signx = x.sign()
    res = torch.zeros_like(x)

    mask1 = absx < 1
    absx1 = absx[mask1]
    res[mask1] = (-12 * x[mask1] + 9 * x[mask1] * absx1)

    mask2 = (absx >= 1) & (absx < 2)
    absx2 = absx[mask2]
    signx2 = signx[mask2]
    res[mask2] = -3 * ((2 - absx2) ** 2) * signx2

    res /= 6.0
    return res

def bspline_basis_second_derivative(x):
    """
    Second derivative of the centered cubic B-spline basis function B(x).
                 1/6 * (9x * sign(x) + 9|x| - 12)   if |x| < 1
    d²B(x)/dx² = 1/6 * (-6|x| + 12)                 if 1 <= |x| < 2
                 0                                  otherwise
    """
    absx = x.abs()
    signx = x.sign()
    res = torch.zeros_like(x)

    mask1 = absx < 1
    absx1 = absx[mask1]
    signx1 = signx[mask1]
    res[mask1] = (9 * x[mask1] * signx1 + 9 * absx1 - 12)

    mask2 = (absx >= 1) & (absx < 2)
    absx2 = absx[mask2]
    res[mask2] = (-6 * absx2 + 12)

    res /= 6.0
    return res

def convert_to_bspline_coefficients_1d_lin(signal, dim):
    """
    Solves for the B-spline coefficients using a linear system approach.
    We set up the system A @ coeffs = signal and solve for the coefficients.
    """
    N = signal.shape[dim]
    signal = signal.clone()
    signal = signal.transpose(dim, -1)
    signal_shape = list(signal.shape)
    signal_shape[-1] = -1
    signal = signal.reshape(-1, N)

    # Construct A
    pos_bas_func = torch.arange(-1, N+1, device=signal.device)[None, :]
    eval_at = torch.arange(N, device=signal.device)[:, None]
    indices = eval_at - pos_bas_func
    B = bspline_basis(indices.float())

    # Boundary condition d²B/dx² = 0
    indices = eval_at[[0, -1], :] - pos_bas_func
    d2B = bspline_basis_second_derivative(indices.float())

    A = torch.cat([B, d2B], dim=0)

    # Add zeros to signal for zero derivatives at border
    zeros = torch.zeros(signal.size(0), 2, device=signal.device)
    signal = torch.cat([signal, zeros], dim=1)

    # Solve the linear system: A @ coeffs = signal
    coeffs = torch.linalg.solve(A, signal.T)
    coeffs = coeffs.T
    coeffs = coeffs.view(*signal_shape).transpose(-1, dim)

    return coeffs

def interpolate_bspline_1d(coeffs, new_spatial_dim, dim):
    """
    1D interpolation with B-spline coefficients.
    """
    N = new_spatial_dim
    M = coeffs.shape[dim]
    pos_bas_func = torch.arange(-1, M-1, device=coeffs.device)[None, :]
    eval_at = torch.linspace(0, M-3, N, device=coeffs.device)[:, None]
    indices = eval_at - pos_bas_func
    Ah = bspline_basis(indices.float())

    # Perform matrix multiplication for interpolation
    if dim == 4:
        yh = torch.einsum('u z, ...x y z->... x y u', Ah, coeffs)
    if dim == 3:
        yh = torch.einsum('u y, ...x y z->... x u z', Ah, coeffs)
    if dim == 2:
        yh = torch.einsum('u x, ...x y z->... u y z', Ah, coeffs)
    return yh

def bspline_interp3d(volume, scale_factor):
    """
    Performs 3D cubic B-spline interpolation on a tensor of shape (B, C, H, W, D).
    Returns the interpolated tensor with each spatial dim scaled by `scale_factor`.
    """
    assert volume.ndim == 5, "Expected input of shape (B, C, H, W, D)"
    B, C, H, W, D = volume.shape

    # Along depth (D)
    coeffs = convert_to_bspline_coefficients_1d_lin(volume, dim=4)
    new_D = int(D * scale_factor)
    volume = interpolate_bspline_1d(coeffs, new_D, dim=4)

    # Along width (W)
    coeffs = convert_to_bspline_coefficients_1d_lin(volume, dim=3)
    new_W = int(W * scale_factor)
    volume = interpolate_bspline_1d(coeffs, new_W, dim=3)

    # Along height (H)
    coeffs = convert_to_bspline_coefficients_1d_lin(volume, dim=2)
    new_H = int(H * scale_factor)
    volume = interpolate_bspline_1d(coeffs, new_H, dim=2)

    return volume
