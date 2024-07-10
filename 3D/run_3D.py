# 
import shutil
import sys
import time

import torch

from hyperparameters import *
from init_conditions import *
from io_utils import *
from mgpcg import *
from taichi_utils import *

#

dx = 1./res_y

ti.init(arch=ti.cuda, device_memory_GB=15.0, debug = False)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# uniform distribute particles
particles_per_cell_axis = 2
dist_between_neighbor = dx / particles_per_cell_axis

one_sixth = 1. / 6

# solver
boundary_types = ti.Matrix([[2, 2], [2, 2], [2, 2]], ti.i32) # boundaries: 1 means Dirichlet, 2 means Neumann
solver = MGPCG_3(boundary_types = boundary_types, N = [res_x, res_y, res_z], base_level=3)

# undeformed coordinates (cell center and faces)
X = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
X_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z))
X_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z))
X_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1))
center_coords_func(X, dx)
x_coords_func(X_x, dx)
y_coords_func(X_y, dx)
z_coords_func(X_z, dx)

# particle storage
initial_particle_num = res_x * res_y * res_z * particles_per_cell
particle_num = initial_particle_num * total_particles_num_ratio
particles_pos = ti.Vector.field(3, float, shape=particle_num)
particles_init_imp = ti.Vector.field(3, float, shape=particle_num)
particles_init_imp_grad_m = ti.Vector.field(3, float, shape=particle_num)
particles_smoke = ti.Vector.field(4, float, shape=particle_num)
particles_init_grad_smoke = ti.Matrix.field(3, 4, float, shape=particle_num)

# back flow map
T_x_init = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_x
T_y_init = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_y
T_z_init = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_z
T_x_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_x
T_y_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_y
T_z_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_z

T_x_grid = ti.Vector.field(3, float, shape=(res_x + 1, res_y, res_z))  # d_psi / d_x
T_y_grid = ti.Vector.field(3, float, shape=(res_x, res_y + 1, res_z))  # d_psi / d_y
T_z_grid = ti.Vector.field(3, float, shape=(res_x, res_y, res_z + 1))
psi_x_grid = ti.Vector.field(3, float, shape=(res_x + 1, res_y, res_z))  # x coordinate
psi_y_grid = ti.Vector.field(3, float, shape=(res_x, res_y + 1, res_z))  # y coordinate
psi_z_grid = ti.Vector.field(3, float, shape=(res_x, res_y, res_z + 1))

# paticles in each cell
cell_max_particle_num = int(cell_max_particle_num_ratio * particles_per_cell)
cell_particle_num = ti.field(int, shape=(res_x, res_y, res_z))
cell_particles_id = ti.field(int, shape=(res_x, res_y, res_z, cell_max_particle_num))

# velocity storage
u = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
w = ti.Vector.field(3, float, shape=(res_x, res_y, res_z)) # curl of u
u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
u_z = ti.field(float, shape=(res_x, res_y, res_z+1))

# P2G weight storage
p2g_weight = ti.field(float, shape=(res_x, res_y, res_z))
p2g_weight_x = ti.field(float, shape=(res_x + 1, res_y, res_z))
p2g_weight_y = ti.field(float, shape=(res_x, res_y + 1, res_z))
p2g_weight_z = ti.field(float, shape=(res_x, res_y, res_z + 1))

# APIC
init_C_x = ti.Vector.field(3, float, shape=particle_num)
init_C_y = ti.Vector.field(3, float, shape=particle_num)
init_C_z = ti.Vector.field(3, float, shape=particle_num)

# CFL related
max_speed = ti.field(float, shape=())

# smoke
smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
tmp_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))

@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
    max_speed[None] = 1.e-3 # avoid dividing by zero
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        u = 0.5 * (u_x[i, j, k] + u_x[i+1, j, k])
        v = 0.5 * (u_y[i, j, k] + u_y[i, j+1, k])
        w = 0.5 * (u_z[i, j, k] + u_z[i, j, k+1])
        speed = ti.sqrt(u ** 2 + v ** 2 + w ** 2)
        ti.atomic_max(max_speed[None], speed)

# set to undeformed config
@ti.kernel
def reset_to_identity_grid(psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(),
                    T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for I in ti.grouped(psi_x):
        psi_x[I] = X_x[I]
    for I in ti.grouped(psi_y):
        psi_y[I] = X_y[I]
    for I in ti.grouped(psi_z):
        psi_z[I] = X_z[I]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)

@ti.kernel
def reset_to_identity(psi: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for i in psi:
        psi[i] = particles_pos[i]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)

@ti.kernel
def reset_T_to_identity(T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)

# curr step should be in range(reinit_every)

@ti.kernel
def RK4_grid(psi_x: ti.template(), T_x: ti.template(), 
            u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dt: float):

    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[I] # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        T_x1 = T_x[I] + 0.5 * neg_dt * dT_x_dt1
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1 # time derivative of T
        # prepare third
        psi_x2 = psi_x[I] + 0.5 * neg_dt * u2 # advance 0.5 again
        T_x2 = T_x[I] + 0.5 * neg_dt * dT_x_dt2 
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2 # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[I] + 1.0 * neg_dt * u3
        T_x3 = T_x[I] + 1.0 * neg_dt * dT_x_dt3 # advance 1.0
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3 # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] + neg_dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] + neg_dt * 1./6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4) # advance full

@ti.kernel
def RK4_T_forward(psi: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
                  u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dt: float):
    for i in psi:
        # if particles_active[i] == 1:
            # first
            u1, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi @ T_y[i]  # time derivative of T
            dT_z_dt1 = grad_u_at_psi @ T_z[i]  # time derivative of T
            # prepare second
            psi_x1 = psi[i] + 0.5 * dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] - 0.5 * dt * dT_x_dt1
            T_y1 = T_y[i] - 0.5 * dt * dT_y_dt1
            T_z1 = T_z[i] - 0.5 * dt * dT_z_dt1
            # second
            u2, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi @ T_y1  # time derivative of T
            dT_z_dt2 = grad_u_at_psi @ T_z1  # time derivative of T
            # prepare third
            psi_x2 = psi[i] + 0.5 * dt * u2  # advance 0.5 again
            T_x2 = T_x[i] - 0.5 * dt * dT_x_dt2
            T_y2 = T_y[i] - 0.5 * dt * dT_y_dt2
            T_z2 = T_z[i] - 0.5 * dt * dT_z_dt2
            # third
            u3, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi_x2, dx)
            dT_x_dt3 = grad_u_at_psi @ T_x2  # time derivative of T
            dT_y_dt3 = grad_u_at_psi @ T_y2  # time derivative of T
            dT_z_dt3 = grad_u_at_psi @ T_z2  # time derivative of T
            # prepare fourth
            psi_x3 = psi[i] + 1.0 * dt * u3
            T_x3 = T_x[i] - 1.0 * dt * dT_x_dt3  # advance 1.0
            T_y3 = T_y[i] - 1.0 * dt * dT_y_dt3  # advance 1.0
            T_z3 = T_z[i] - 1.0 * dt * dT_z_dt3  # advance 1.0
            # fourth
            u4, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi_x3, dx)
            dT_x_dt4 = grad_u_at_psi @ T_x3  # time derivative of T
            dT_y_dt4 = grad_u_at_psi @ T_y3  # time derivative of T
            dT_z_dt4 = grad_u_at_psi @ T_z3  # time derivative of T
            # final advance
            psi[i] = psi[i] + dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
            T_x[i] = T_x[i] - dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
            T_y[i] = T_y[i] - dt * 1. / 6 * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full
            T_z[i] = T_z[i] - dt * 1. / 6 * (dT_z_dt1 + 2 * dT_z_dt2 + 2 * dT_z_dt3 + dT_z_dt4)  # advance full

@ti.kernel
def RK4_T_and_T_grad_m_forward(psi: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
                               T_x_grad_m: ti.template(), T_y_grad_m: ti.template(), T_z_grad_m: ti.template(),
                               u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dt: float):
    for i in psi:
        # if particles_active[i] == 1:
            # first
            u1, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi @ T_y[i]  # time derivative of T
            dT_z_dt1 = grad_u_at_psi @ T_z[i]  # time derivative of T
            dT_x_dt1_grad_m = grad_u_at_psi @ T_x_grad_m[i]  # time derivative of T
            dT_y_dt1_grad_m = grad_u_at_psi @ T_y_grad_m[i]  # time derivative of T
            dT_z_dt1_grad_m = grad_u_at_psi @ T_z_grad_m[i]  # time derivative of T
            # prepare second
            half_dt = 0.5 * dt
            psi_x1 = psi[i] + half_dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] - half_dt * dT_x_dt1
            T_y1 = T_y[i] - half_dt * dT_y_dt1
            T_z1 = T_z[i] - half_dt * dT_z_dt1
            T_x1_grad_m = T_x_grad_m[i] - half_dt * dT_x_dt1_grad_m
            T_y1_grad_m = T_y_grad_m[i] - half_dt * dT_y_dt1_grad_m
            T_z1_grad_m = T_z_grad_m[i] - half_dt * dT_z_dt1_grad_m
            # second
            u2, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi @ T_y1  # time derivative of T
            dT_z_dt2 = grad_u_at_psi @ T_z1  # time derivative of T
            dT_x_dt2_grad_m = grad_u_at_psi @ T_x1_grad_m  # time derivative of T
            dT_y_dt2_grad_m = grad_u_at_psi @ T_y1_grad_m  # time derivative of T
            dT_z_dt2_grad_m = grad_u_at_psi @ T_z1_grad_m  # time derivative of T
            # prepare third
            psi_x2 = psi[i] + half_dt * u2  # advance 0.5 again
            T_x2 = T_x[i] - half_dt * dT_x_dt2
            T_y2 = T_y[i] - half_dt * dT_y_dt2
            T_z2 = T_z[i] - half_dt * dT_z_dt2
            T_x2_grad_m = T_x_grad_m[i] - half_dt * dT_x_dt2_grad_m
            T_y2_grad_m = T_y_grad_m[i] - half_dt * dT_y_dt2_grad_m
            T_z2_grad_m = T_z_grad_m[i] - half_dt * dT_z_dt2_grad_m
            # third
            u3, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi_x2, dx)
            dT_x_dt3 = grad_u_at_psi @ T_x2  # time derivative of T
            dT_y_dt3 = grad_u_at_psi @ T_y2  # time derivative of T
            dT_z_dt3 = grad_u_at_psi @ T_z2  # time derivative of T
            dT_x_dt3_grad_m = grad_u_at_psi @ T_x2_grad_m  # time derivative of T
            dT_y_dt3_grad_m = grad_u_at_psi @ T_y2_grad_m  # time derivative of T
            dT_z_dt3_grad_m = grad_u_at_psi @ T_z2_grad_m  # time derivative of T
            # prepare fourth
            psi_x3 = psi[i] + dt * u3
            T_x3 = T_x[i] - dt * dT_x_dt3  # advance 1.0
            T_y3 = T_y[i] - dt * dT_y_dt3  # advance 1.0
            T_z3 = T_z[i] - dt * dT_z_dt3  # advance 1.0
            T_x3_grad_m = T_x_grad_m[i] - dt * dT_x_dt3_grad_m  # advance 1.0
            T_y3_grad_m = T_y_grad_m[i] - dt * dT_y_dt3_grad_m  # advance 1.0
            T_z3_grad_m = T_z_grad_m[i] - dt * dT_z_dt3_grad_m  # advance 1.0
            # fourth
            u4, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi_x3, dx)
            dT_x_dt4 = grad_u_at_psi @ T_x3  # time derivative of T
            dT_y_dt4 = grad_u_at_psi @ T_y3  # time derivative of T
            dT_z_dt4 = grad_u_at_psi @ T_z3  # time derivative of T
            dT_x_dt4_grad_m = grad_u_at_psi @ T_x3_grad_m  # time derivative of T
            dT_y_dt4_grad_m = grad_u_at_psi @ T_y3_grad_m  # time derivative of T
            dT_z_dt4_grad_m = grad_u_at_psi @ T_z3_grad_m  # time derivative of T
            # final advance
            one_sixth_dt = dt * one_sixth
            psi[i] = psi[i] + one_sixth_dt * (u1 + 2 * u2 + 2 * u3 + u4)
            T_x[i] = T_x[i] - one_sixth_dt * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
            T_y[i] = T_y[i] - one_sixth_dt * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full
            T_z[i] = T_z[i] - one_sixth_dt * (dT_z_dt1 + 2 * dT_z_dt2 + 2 * dT_z_dt3 + dT_z_dt4)  # advance full
            T_x_grad_m[i] = T_x_grad_m[i] - one_sixth_dt * (dT_x_dt1_grad_m + 2 * dT_x_dt2_grad_m + 2 * dT_x_dt3_grad_m + dT_x_dt4_grad_m)  # advance full
            T_y_grad_m[i] = T_y_grad_m[i] - one_sixth_dt * (dT_y_dt1_grad_m + 2 * dT_y_dt2_grad_m + 2 * dT_y_dt3_grad_m + dT_y_dt4_grad_m)  # advance full
            T_z_grad_m[i] = T_z_grad_m[i] - one_sixth_dt * (dT_z_dt1_grad_m + 2 * dT_z_dt2_grad_m + 2 * dT_z_dt3_grad_m + dT_z_dt4_grad_m)  # advance full

# u_x0, u_y0, u_z0 are the initial time quantities
# u_x1, u_y1, u_z0 are the current time quantities (to be modified)
@ti.kernel
def advect_u(u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
            u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(),
            T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
            psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    # x velocity
    for I in ti.grouped(u_x1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        u_x1[I] = T_x[I].dot(u_at_psi)
    # y velocity
    for I in ti.grouped(u_y1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_y[I], dx)
        u_y1[I] = T_y[I].dot(u_at_psi)
    # z velocity
    for I in ti.grouped(u_z1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_z[I], dx)
        u_z1[I] = T_z[I].dot(u_at_psi)


@ti.kernel
def advect_smoke(smoke0: ti.template(), smoke1: ti.template(), 
            psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    # horizontal velocity
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        psi_c = 1./6 * (psi_x[i, j, k] + psi_x[i+1, j, k] + \
                        psi_y[i, j, k] + psi_y[i, j+1, k] + \
                        psi_z[i, j, k] + psi_z[i, j, k+1])
        smoke1[i,j,k] = interp_1(smoke0, psi_c, dx)

@ti.kernel
def clamp_smoke(smoke0: ti.template(), smoke1: ti.template(),
            psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    # horizontal velocity
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        psi_c = 1./6 * (psi_x[i, j, k] + psi_x[i+1, j, k] + \
                        psi_y[i, j, k] + psi_y[i, j+1, k] + \
                        psi_z[i, j, k] + psi_z[i, j, k+1])
        mini, maxi = sample_min_max_1(smoke0, psi_c, dx)
        smoke1[i,j,k] = ti.math.clamp(smoke1[i,j,k], mini, maxi)

def stretch_T_and_advect_particles(particles_pos, T_x, T_y, T_z, u_x, u_y, u_z, dt):
    RK4_T_forward(particles_pos, T_x, T_y, T_z, u_x, u_y, u_z, dt)

def stretch_T_and_T_grad_m_and_advect_particles(particles_pos, T_x, T_y, T_z, T_x_grad_m, T_y_grad_m, T_z_grad_m, u_x, u_y, u_z, dt):
    RK4_T_and_T_grad_m_forward(particles_pos, T_x, T_y, T_z, T_x_grad_m, T_y_grad_m, T_z_grad_m, u_x, u_y, u_z, dt)

@ti.kernel
def get_particles_id_in_every_cell(cell_particles_id: ti.template(), cell_particle_num: ti.template(),
                                   particles_pos: ti.template()):
    cell_particles_id.fill(-1)
    cell_particle_num.fill(0)
    for i in particles_pos:
        cell_id = int(particles_pos[i] / dx)
        particles_index_in_cell = ti.atomic_add(cell_particle_num[cell_id], 1)
        if particles_index_in_cell < cell_max_particle_num:
            cell_particles_id[cell_id[0], cell_id[1], cell_id[2], particles_index_in_cell] = i

@ti.kernel
def compute_dT_dx(grad_T_init_x: ti.template(), grad_T_init_y: ti.template(), grad_T_init_z: ti.template(),
                          T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
                          particles_init_pos: ti.template(), cell_particles_id: ti.template(),
                          cell_particle_num: ti.template()):

    for i in grad_T_init_x:
        # if particles_active[i] == 1:
            grad_T_init_x[i] = ti.Matrix.zero(float, 3, 3)
            grad_T_init_y[i] = ti.Matrix.zero(float, 3, 3)
            grad_T_init_z[i] = ti.Matrix.zero(float, 3, 3)

            base_cell_id = int(particles_init_pos[i] / dx)

            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                    neighbor_cell_id = base_cell_id + offset
                    for k in ti.static(ti.ndrange(0, cell_particle_num[neighbor_cell_id])):
                        neighbor_particle_id = \
                            cell_particles_id[neighbor_cell_id[0], neighbor_cell_id[1], neighbor_cell_id[2], k]
                        neighbor_particle_pos = particles_init_pos[neighbor_particle_id]
                        # if particles_active[neighbor_particle_id] == 1:
                        dist_x = neighbor_particle_pos[0] - particles_init_pos[i][0]
                        dist_y = neighbor_particle_pos[1] - particles_init_pos[i][1]
                        dist_z = neighbor_particle_pos[2] - particles_init_pos[i][2]

                        dw_x = 1. / dx * dN_2(dist_x) * N_2(dist_y) * N_2(dist_z)
                        dw_y = 1. / dx * N_2(dist_x) * dN_2(dist_y) * N_2(dist_z)
                        dw_z = 1. / dx * N_2(dist_x) * N_2(dist_y) * dN_2(dist_z)
                        dw = ti.Vector([dw_x, dw_y, dw_z])

                        T = ti.Matrix.cols(
                            [T_x[neighbor_particle_id], T_y[neighbor_particle_id], T_z[neighbor_particle_id]])
                        # T_xx = ti.Vector([T[0, 0], T[1, 0], T[2, 0]])
                        # T_yy = ti.Vector([T[0, 1], T[1, 1], T[2, 1]])
                        # T_zz = ti.Vector([T[0, 2], T[1, 2], T[2, 2]])
                        grad_T_init_x[i] += dw.outer_product(T[0, :])
                        grad_T_init_y[i] += dw.outer_product(T[1, :])
                        grad_T_init_z[i] += dw.outer_product(T[2, :])

@ti.kernel
def update_particles_imp(particles_imp: ti.template(), particles_init_imp: ti.template(),
                         T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for i in particles_imp:
        # if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i], T_z[i]])
            # if forward_update_T:
            #     T = T.transpose()
            particles_imp[i] = T @ particles_init_imp[i]

@ti.kernel
def update_particles_grad_smoke(particles_grad_smoke: ti.template(), particles_init_grad_smoke: ti.template(),
                                T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for i in particles_grad_smoke:
        # if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i], T_z[i]])
            # if forward_update_T:
            #     T = T.transpose()
            particles_grad_smoke[i] = (T @ particles_init_grad_smoke[i].transpose()).transpose()

@ti.kernel
def update_particles_grad_m(C_x: ti.template(), C_y: ti.template(), C_z: ti.template(), init_C_x: ti.template(),
                            init_C_y: ti.template(), init_C_z: ti.template(), T_x: ti.template(), T_y: ti.template(),
                            T_z: ti.template(), grad_T_init_x: ti.template(), grad_T_init_y: ti.template(),
                            grad_T_init_z: ti.template(), particles_init_imp: ti.template()):
    for i in C_x:
        # if particles_active[i] == 1:
            T = ti.Matrix.rows([T_x[i], T_y[i], T_z[i]])
            T_transpose = ti.Matrix.cols([T_x[i], T_y[i], T_z[i]])
            # T_xx = ti.Vector([T[0, 0], T[1, 0], T[2, 0]])
            # T_yy = ti.Vector([T[0, 1], T[1, 1], T[2, 1]])
            # T_zz = ti.Vector([T[0, 2], T[1, 2], T[2, 2]])
            init_C = ti.Matrix.rows([init_C_x[i], init_C_y[i], init_C_z[i]])
            T_init_C_T = T_transpose @ (init_C @ T)
            C_x[i] = grad_T_init_x[i] @ particles_init_imp[i] + T_init_C_T[0, :]
            C_y[i] = grad_T_init_y[i] @ particles_init_imp[i] + T_init_C_T[1, :]
            C_z[i] = grad_T_init_z[i] @ particles_init_imp[i] + T_init_C_T[2, :]

@ti.kernel
def update_T(T_x_init: ti.template(), T_y_init: ti.template(), T_z_init: ti.template(), T_x_grad_m: ti.template(),
             T_y_grad_m: ti.template(), T_z_grad_m: ti.template()):
    for i in T_x_init:
        T_grad_m = ti.Matrix.cols([T_x_grad_m[i], T_y_grad_m[i], T_z_grad_m[i]])
        T_init = ti.Matrix.cols([T_x_init[i], T_y_init[i], T_z_init[i]])
        T = T_grad_m @ T_init
        T_x_init[i] = T[:, 0]
        T_y_init[i] = T[:, 1]
        T_z_init[i] = T[:, 2]

@ti.kernel
def P2G(particles_init_imp: ti.template(), particles_init_imp_grad_m: ti.template(), particles_pos: ti.template(),
        u_x: ti.template(), u_y: ti.template(), u_z: ti.template(), p2g_weight_x: ti.template(), p2g_weight_y: ti.template(),
        p2g_weight_z: ti.template(), T_x_grad_m: ti.template(), T_y_grad_m: ti.template(), T_z_grad_m: ti.template(),
        T_x_init: ti.template(), T_y_init: ti.template(), T_z_init: ti.template(),
        cell_particles_id: ti.template(), cell_particle_num: ti.template()):
    u_x.fill(0.0)
    u_y.fill(0.0)
    u_z.fill(0.0)
    # imp_x.fill(0.0)
    # imp_y.fill(0.0)
    # imp_z.fill(0.0)
    # psi_x_grid.fill(0.0)
    # psi_y_grid.fill(0.0)
    # psi_z_grid.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)
    p2g_weight_z.fill(0.0)

    for i in particles_init_imp:
        ### evolve grad m ###
        T_grad_m = ti.Matrix.rows([T_x_grad_m[i], T_y_grad_m[i], T_z_grad_m[i]])
        T_grad_m_transpose = ti.Matrix.cols([T_x_grad_m[i], T_y_grad_m[i], T_z_grad_m[i]])
        # T_xx = ti.Vector([T[0, 0], T[1, 0], T[2, 0]])
        # T_yy = ti.Vector([T[0, 1], T[1, 1], T[2, 1]])
        # T_zz = ti.Vector([T[0, 2], T[1, 2], T[2, 2]])
        init_C = ti.Matrix.rows([init_C_x[i], init_C_y[i], init_C_z[i]])
        T_init_C_T = T_grad_m_transpose @ (init_C @ T_grad_m)

        ### update T ###
        T_init = ti.Matrix.cols([T_x_init[i], T_y_init[i], T_z_init[i]])
        T_transpose = T_grad_m_transpose @ T_init

        ### evolve m ###
        particles_imp = T_transpose @ particles_init_imp[i]

        ### real P2G ###
        # horizontal impulse
        pos = particles_pos[i] / dx

        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5) * N_2(pos[2] - face_id[2] - 0.5)
                dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1], face_id[2] + 0.5 - pos[2]]) * dx
                p2g_weight_x[face_id] += weight
                delta = T_init_C_T[0, :].dot(dpos)
                # print(particles_imp[i][0], weight, delta)
                if use_APIC:
                    u_x[face_id] += (particles_imp[0] + delta) * weight
                else:
                    u_x[face_id] += (particles_imp[0]) * weight

                # psi_x_grid[face_id] += (psi[i] + T.transpose() @ dpos) * weight

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y and 0 <= face_id[2] < res_z:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1]) * N_2(pos[2] - face_id[2] - 0.5)
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1], face_id[2] + 0.5 - pos[2]]) * dx
                p2g_weight_y[face_id] += weight
                delta = T_init_C_T[1, :].dot(dpos)
                if use_APIC:
                    u_y[face_id] += (particles_imp[1] + delta) * weight
                else:
                    u_y[face_id] += (particles_imp[1]) * weight

                # psi_y_grid[face_id] += (psi[i] + T.transpose() @ dpos) * weight

        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] <= res_z:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1] - 0.5) * N_2(pos[2] - face_id[2])
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] + 0.5 - pos[1], face_id[2] - pos[2]]) * dx
                p2g_weight_z[face_id] += weight
                delta = T_init_C_T[2, :].dot(dpos)
                if use_APIC:
                    u_z[face_id] += (particles_imp[2] + delta) * weight
                else:
                    u_z[face_id] += (particles_imp[2]) * weight

                # psi_z_grid[face_id] += (psi[i] + T.transpose() @ dpos) * weight

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            u_x[I] /= p2g_weight_x[I]
            # scale = 1. / p2g_weight_x[I]
            # u_x[I] *= scale
            # psi_x_grid[I] *= scale
            # imp_x[I] = u_x[I]

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            u_y[I] /= p2g_weight_y[I]
            # scale = 1. / p2g_weight_y[I]
            # u_y[I] *= scale
            # psi_y_grid[I] *= scale
            # imp_y[I] = u_y[I]

    for I in ti.grouped(p2g_weight_z):
        if p2g_weight_z[I] > 0:
            u_z[I] /= p2g_weight_z[I]
            # scale = 1. / p2g_weight_z[I]
            # u_z[I] *= scale
            # psi_z_grid[I] *= scale
            # imp_z[I] = u_z[I]

@ti.kernel
def P2G_smoke(particles_smoke: ti.template(), particles_init_grad_smoke: ti.template(), particles_pos: ti.template(),
              smoke: ti.template(),  T_x_grad_m: ti.template(), T_y_grad_m: ti.template(), T_z_grad_m: ti.template(),
              T_x_init: ti.template(), T_y_init: ti.template(), T_z_init: ti.template(),p2g_weight: ti.template()):
    smoke.fill(0.)
    p2g_weight.fill(0.)

    for i in particles_smoke:
        ### update T ###
        T_grad_m_transpose = ti.Matrix.cols([T_x_grad_m[i], T_y_grad_m[i], T_z_grad_m[i]])
        T_init = ti.Matrix.cols([T_x_init[i], T_y_init[i], T_z_init[i]])
        T_transpose = T_grad_m_transpose @ T_init

        ### real P2G_smoke ###
        pos = particles_pos[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5)
        for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1] - 0.5) * N_2(pos[2] - face_id[2] - 0.5)
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] + 0.5 - pos[1], face_id[2] + 0.5 - pos[2]]) * dx
                p2g_weight[face_id] += weight
                delta = dpos @ (T_transpose @ particles_init_grad_smoke[i])
                # print(particles_imp[i][0], weight, delta)
                if use_APIC_smoke:
                    smoke[face_id] += (particles_smoke[i] + delta) * weight
                else:
                    smoke[face_id] += particles_smoke[i] * weight

    for I in ti.grouped(p2g_weight):
        if p2g_weight[I] > 0:
            smoke[I] /= p2g_weight[I]
            for j in range(4):
                if smoke[I][j] < 0.:
                    smoke[I][j] = 0.0
                if smoke[I][j] > 1.:
                    smoke[I][j] = 1.0
            # scale = 1. / p2g_weight_z[I]
            # smoke[I] *= scale

@ti.kernel
def clamp_smoke_particle(cell_particles_id: ti.template(), cell_particle_num: ti.template(),
                         particles_init_pos: ti.template(), smoke: ti.template(), init_smoke: ti.template()):
    for I in ti.grouped(cell_particle_num):
        min_smoke = ti.Vector([1000., 1000., 1000., 1000.])
        max_smoke = ti.Vector([-1000., -1000., -1000., -1000.])
        valid_count = 0
        for i in ti.static(ti.ndrange(0, cell_particle_num[I])):
            particle_id = cell_particles_id[I[0], I[1], I[2], i]
            particle_init_base_cell = int(particles_init_pos[particle_id] / dx)
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * dim)):
                particle_init_neighbor_cell = particle_init_base_cell + offset
                if 0 <= particle_init_neighbor_cell[0] < res_x and 0 <= particle_init_neighbor_cell[1] < res_y \
                        and 0 <= particle_init_neighbor_cell[2] < res_z:
                    ti.atomic_min(min_smoke, init_smoke[particle_init_neighbor_cell])
                    ti.atomic_max(max_smoke, init_smoke[particle_init_neighbor_cell])
                    valid_count += 1

        if valid_count >= 1:
            smoke[I] = ti.math.clamp(smoke[I], min_smoke, max_smoke)


# main function
def main(from_frame = 0, testing = False):
    from_frame = max(0, from_frame)
    # create some folders
    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    if from_frame <= 0:
        remove_everything_in(logsdir)

    vtkdir = "vtks"
    vtkdir = os.path.join(logsdir, vtkdir)
    os.makedirs(vtkdir, exist_ok=True)
    vort2dir = 'vort_2D'
    vort2dir = os.path.join(logsdir, vort2dir)
    os.makedirs(vort2dir, exist_ok=True)
    smoke2dir = 'smoke_2D'
    smoke2dir = os.path.join(logsdir, smoke2dir)
    os.makedirs(smoke2dir, exist_ok=True)
    ckptdir = 'ckpts'
    ckptdir = os.path.join(logsdir, ckptdir)
    os.makedirs(ckptdir, exist_ok=True)
    levelsdir = 'levels'
    levelsdir = os.path.join(logsdir, levelsdir)
    os.makedirs(levelsdir, exist_ok=True)
    modeldir = 'model' # saves the model
    modeldir = os.path.join(logsdir, modeldir)
    os.makedirs(modeldir, exist_ok=True)

    shutil.copyfile('./hyperparameters.py', f'{logsdir}/hyperparameters.py')

    if testing:
        testdir = 'test_buffer'
        testdir = os.path.join(logsdir, testdir)
        os.makedirs(testdir, exist_ok=True)
        remove_everything_in(testdir)
        GTdir = os.path.join(testdir, "GT")
        os.makedirs(GTdir, exist_ok=True)
        preddir = os.path.join(testdir, "pred")
        os.makedirs(preddir, exist_ok=True)

    # initial condition
    if from_frame <= 0:
        if case == 0:
            # u_x_numpy = np.load(os.path.join("init_conditions", "leapfrog_init_vel_x.npy")).astype(np.float32)
            # u_y_numpy = np.load(os.path.join("init_conditions", "leapfrog_init_vel_y.npy")).astype(np.float32)
            # u_z_numpy = np.load(os.path.join("init_conditions", "leapfrog_init_vel_z.npy")).astype(np.float32)
            # u_x.from_numpy(u_x_numpy.astype(np.float32))
            # u_y.from_numpy(u_y_numpy.astype(np.float32))
            # u_z.from_numpy(u_z_numpy.astype(np.float32))
            # solver.Poisson(u_x, u_y, u_z)
            init_vorts_leapfrog(X, u, smoke, tmp_smoke)
            #init_vorts_four(X, u, smoke, tmp_smoke)
            split_central_vector(u, u_x, u_y, u_z)
            solver.Poisson(u_x, u_y, u_z)
        elif case == 1:
            init_vorts_eight(X, u, smoke, tmp_smoke)
            split_central_vector(u, u_x, u_y, u_z)
            solver.Poisson(u_x, u_y, u_z)
            
    else:
        u_x.from_numpy(np.load(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame) + ".npy")))
        u_y.from_numpy(np.load(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame) + ".npy")))
        u_z.from_numpy(np.load(os.path.join(ckptdir, "vel_z_numpy_" + str(from_frame) + ".npy")))
        smoke.from_numpy(np.load(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame) + ".npy")))

    # particles_active.fill(1)

    # for visualization
    get_central_vector(u_x, u_y, u_z, u)
    curl(u, w, dx)
    w_numpy = w.to_numpy()
    w_norm = np.linalg.norm(w_numpy, axis = -1)
    w_max = max(np.abs(w_norm.max()), np.abs(w_norm.min()))
    w_min = -1 * w_max
    write_field(w_norm[:,:,res_z//2], vort2dir, from_frame, vmin = w_min, vmax = w_max)
    smoke_numpy = smoke.to_numpy()
    smoke_norm = smoke_numpy[...,-1]
    write_image(smoke_numpy[:,:,res_z//2][...,:3], smoke2dir, from_frame)
    write_w_and_smoke(w_norm, smoke_norm, vtkdir, from_frame)


    # save init 
    np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame)), u_x.to_numpy())
    np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame)), u_y.to_numpy())
    np.save(os.path.join(ckptdir, "vel_z_numpy_" + str(from_frame)), u_z.to_numpy())
    # np.save(os.path.join(ckptdir, "w_numpy_" + str(from_frame)), w_numpy)
    # np.save(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame)), smoke_numpy)
    if save_particle_pos_numpy:
        np.save(os.path.join(ckptdir, "particles_pos_numpy_" + str(from_frame)), particles_pos.to_numpy())
    # done  

    sub_t = 0. # the time since last reinit
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0 # number of reinitializations already performed
    i = -1
    ik = 0

    frame_times = np.zeros(total_steps)
    while True:
        start_time = time.time()
        i += 1
        j = i % reinit_every
        i_next = i + 1
        j_next = i_next % reinit_every
        print("[Simulate] Running step: ", i, " / substep: ", j)

        # determine dt
        calc_max_speed(u_x, u_y, u_z) # saved to max_speed[None]
        curr_dt = CFL * dx / max_speed[None]

        if save_frame_each_step:
            output_frame = True
            frame_idx += 1
        else:
            if sub_t+curr_dt >= visualize_dt: # if over
                curr_dt = visualize_dt-sub_t
                sub_t = 0. # empty sub_t
                if i <= total_steps - 1:
                    print(f'step execution time: {frame_times[i]:.6f} seconds')
                frame_idx += 1
                output_frame = True
                print(f'Visualized frame {frame_idx}')
            else:
                sub_t += curr_dt
                print(f'Visualize time {sub_t}/{visualize_dt}')
                output_frame = False
        # dts[j] = curr_dt
        # done dt

        # reinitialize flow map if j == 0:
        if j == 0:
            print("[Simulate] Reinitializing the flow map for the: ", num_reinits, " time!")

            if reinit_particle_pos or i == 0:
                init_particles_pos_uniform(particles_pos, X, res_x, res_y, particles_per_cell, dx,
                                           particles_per_cell_axis, dist_between_neighbor)
                # copy_to(particles_pos, particles_init_pos)
                ik = i

            init_particles_imp(particles_init_imp, particles_pos, u_x, u_y, u_z, init_C_x,
                               init_C_y, init_C_z, dx)
            # init_particles_smoke(particles_smoke, particles_init_grad_smoke, particles_pos, smoke, dx)

            # reset_T_to_identity(T_x, T_y, T_z)
            reset_T_to_identity(T_x_init, T_y_init, T_z_init)
            reset_T_to_identity(T_x_grad_m, T_y_grad_m, T_z_grad_m)

            # copy_to(smoke, init_smoke)

        k = (i - ik) % reinit_every_grad_m
        if k == 0:
            init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y, u_z,
                                      init_C_x, init_C_y, init_C_z, dx)
            update_T(T_x_init, T_y_init, T_z_init, T_x_grad_m, T_y_grad_m, T_z_grad_m)
            reset_T_to_identity(T_x_grad_m, T_y_grad_m, T_z_grad_m)
            # copy_to(T_x, T_x_init)
            # copy_to(T_y, T_y_init)
            # copy_to(T_z, T_z_init)

        # start midpoint
        if use_midpoint_vel:
            reset_to_identity_grid(psi_x_grid, psi_y_grid, psi_z_grid, T_x_grid, T_y_grid, T_z_grid)
            RK4_grid(psi_x_grid, T_x_grid, u_x, u_y, u_z, 0.5 * curr_dt)
            RK4_grid(psi_y_grid, T_y_grid, u_x, u_y, u_z, 0.5 * curr_dt)
            RK4_grid(psi_z_grid, T_z_grid, u_x, u_y, u_z, 0.5 * curr_dt)
            # to reduce GPU memory, use p2g_weight_[xyz] to replace tmp_u_[xyz]
            copy_to(u_x, p2g_weight_x)
            copy_to(u_y, p2g_weight_y)
            copy_to(u_z, p2g_weight_z)
            advect_u(p2g_weight_x, p2g_weight_y, p2g_weight_z, u_x, u_y, u_z, T_x_grid, T_y_grid, T_z_grid, \
                     psi_x_grid, psi_y_grid, psi_z_grid, dx)
            solver.Poisson(u_x, u_y, u_z)
        # done midpoint

        # evolve grad m
        # copy_to(particles_pos, particles_pos_backup)
        # stretch_T_and_T_grad_m_and_advect_particles(particles_pos, T_x, T_y, T_z, T_x_grad_m, T_y_grad_m, T_z_grad_m, u_x, u_y, u_z, curr_dt)
        stretch_T_and_advect_particles(particles_pos, T_x_grad_m, T_y_grad_m, T_z_grad_m, u_x, u_y, u_z, curr_dt)
        get_particles_id_in_every_cell(cell_particles_id, cell_particle_num, particles_pos)
        # compute_dT_dx(grad_T_init_x, grad_T_init_y, grad_T_init_z, T_x_grad_m, T_y_grad_m, T_z_grad_m,
        #                       particles_pos, cell_particles_id, cell_particle_num)
        # update_particles_grad_m(C_x, C_y, C_z, init_C_x, init_C_y, init_C_z, T_x_grad_m, T_y_grad_m, T_z_grad_m,
        #                         grad_T_init_x, grad_T_init_y, grad_T_init_z, particles_init_imp_grad_m)


        # evolve m
        # copy_to(particles_pos_backup, particles_pos)
        # stretch_T_and_advect_particles(particles_pos, T_x, T_y, T_z, u_x, u_y, u_z, curr_dt)
        # update_T(T_x, T_y, T_z, T_x_init, T_y_init, T_z_init, T_x_grad_m, T_y_grad_m, T_z_grad_m)
        # update_particles_imp(particles_imp, particles_init_imp, T_x, T_y, T_z)
        # update_particles_grad_smoke(particles_grad_smoke, particles_init_grad_smoke, T_x, T_y, T_z)

        P2G(particles_init_imp, particles_init_imp_grad_m, particles_pos, u_x, u_y, u_z, p2g_weight_x,
            p2g_weight_y, p2g_weight_z, T_x_grad_m, T_y_grad_m, T_z_grad_m, T_x_init, T_y_init, T_z_init,
            cell_particles_id, cell_particle_num)

        # P2G_smoke(particles_smoke, particles_init_grad_smoke, particles_pos, smoke, T_x_grad_m, T_y_grad_m,
        #           T_z_grad_m, T_x_init, T_y_init, T_z_init, p2g_weight)

        # clamp_smoke_particle(cell_particles_id, cell_particle_num, particles_init_pos, smoke, init_smoke)

        # advect_smoke(init_smoke, smoke, psi_x_grid, psi_y_grid, psi_z_grid, dx)
        # clamp_smoke(init_smoke, smoke, psi_x_grid, psi_y_grid, psi_z_grid, dx)

        solver.Poisson(u_x, u_y, u_z)

        end_time = time.time()
        frame_time = end_time - start_time
        print(f'step execution time: {frame_time:.6f} seconds')

        if i <= total_steps - 1:
            frame_times[i] = frame_time

        print("[Simulate] Done with step: ", i, " / substep: ", j, "\n", flush = True)

        if output_frame:
            # for visualization
            # write_image(levels_display[..., np.newaxis], levelsdir, frame_idx)
            get_central_vector(u_x, u_y, u_z, u)
            curl(u, w, dx)
            w_numpy = w.to_numpy()
            w_norm = np.linalg.norm(w_numpy, axis = -1)
            write_field(w_norm[:,:,res_z//2], vort2dir, frame_idx, vmin = w_min, vmax = w_max)
            smoke_numpy = smoke.to_numpy()
            smoke_norm = smoke_numpy[...,-1]
            write_image(smoke_numpy[:,:,res_z//2][...,:3], smoke2dir, frame_idx)
            write_w_and_smoke(w_norm, smoke_norm, vtkdir, frame_idx)

            if frame_idx % ckpt_every == 0:
                np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(frame_idx)), u_x.to_numpy())
                np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(frame_idx)), u_y.to_numpy())
                np.save(os.path.join(ckptdir, "vel_z_numpy_" + str(frame_idx)), u_z.to_numpy())
                # np.save(os.path.join(ckptdir, "w_numpy_" + str(frame_idx)), w_numpy)
                # np.save(os.path.join(ckptdir, "smoke_numpy_" + str(frame_idx)), smoke_numpy)
                if save_particle_pos_numpy:
                    np.save(os.path.join(ckptdir, "particles_pos_numpy_" + str(frame_idx)), particles_pos.to_numpy())

            print("[Simulate] Finished frame: ", frame_idx, " in ", i-last_output_substep, "substeps \n\n")
            last_output_substep = i

            # if reached desired number of frames
            if frame_idx >= total_frames:
                frame_time_dir = 'frame_time'
                frame_time_dir = os.path.join(logsdir, frame_time_dir)
                os.makedirs(f'{frame_time_dir}', exist_ok=True)
                np.save(f'{frame_time_dir}/frame_times.npy', frame_times)
                break

        if use_total_steps and i >= total_steps - 1:
            frame_time_dir = 'frame_time'
            frame_time_dir = os.path.join(logsdir, frame_time_dir)
            os.makedirs(f'{frame_time_dir}', exist_ok=True)
            np.save(f'{frame_time_dir}/frame_times.npy', frame_times)
            break
    
    
        
if __name__ == '__main__':
    print("[Main] Begin")
    if len(sys.argv) <= 1:
        main(from_frame = from_frame)
    else:
        main(from_frame = from_frame, testing = True)
    print("[Main] Complete")
