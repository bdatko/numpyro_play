"""
Module for porting example importance sampling in pyro 
"""
import jax.numpy as jnp
import numpy as np
import torch
from jax import jit, lax

little_g = 9.8  # m/s/s
mu0 = 0.12  # actual coefficient of friction in the experiment
time_measurement_sigma = 0.02  # observation noise in seconds (known quantity)

def simulate(mu, length=2.0, phi=np.pi / 6.0, dt=0.005, noise_sigma=None):
    T = torch.zeros(())
    velocity = torch.zeros(())
    displacement = torch.zeros(())
    acceleration = torch.tensor(little_g * np.sin(phi)) - \
        torch.tensor(little_g * np.cos(phi)) * mu

    if acceleration.numpy() <= 0.0:  # the box doesn't slide if the friction is too large
        return torch.tensor(1.0e5)   # return a very large time instead of infinity

    while displacement.numpy() < length:  # otherwise slide to the end of the inclined plane
        displacement += velocity * dt
        velocity += acceleration * dt
        T += dt

    if noise_sigma is None:
        return T
    else:
        return T + noise_sigma * torch.randn(())

def _body(info):
    displacement, length, velocity, dt, acceleration, T = info
    displacement += velocity * dt
    velocity += acceleration * dt
    T += dt
    
    return displacement, length, velocity, dt, acceleration, T

def _conf(info):
    displacement, length, _, _, _, _ = info
    return displacement < length

_w = lambda info: lax.while_loop(_conf, _body, info)

@jit
def jax_simulate(mu):
    T = jnp.zeros(())
    velocity = jnp.zeros(())
    displacement = jnp.zeros(())
    length = 2.0
    phi = jnp.pi / 6.0
    dt = 0.005
    acceleration = (little_g * jnp.sin(phi)) - (little_g * jnp.cos(phi)) * mu
    info = (displacement, length, velocity, dt, acceleration, T)
    res = lax.cond(acceleration <= 0, info, lambda _: (0.,0.,0.,0.,0.,1.0e5), info, _w)
    return res[-1]
