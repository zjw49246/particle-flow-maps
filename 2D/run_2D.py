#
from hyperparameters import *
from taichi_utils import *
from mgpcg import *
from init_conditions import *
from io_utils import *
import torch
import sys
import shutil
import time

#

dx = 1. / res_y
half_dx = 0.5 * dx
upper_boundary = 1 - half_dx
lower_boundary = half_dx
right_boundary = res_x * dx - half_dx
left_boundary = half_dx

ti.init(arch=ti.cuda, device_memory_GB=4.0, debug=False)

# uniform distribute particles
particles_per_cell_axis = int(ti.sqrt(particles_per_cell))
dist_between_neighbor = dx / particles_per_cell_axis

# solver
boundary_types = ti.Matrix([[2, 2], [2, 2]], ti.i32)  # boundaries: 1 means Dirichlet, 2 means Neumann
solver = MGPCG_2(boundary_types=boundary_types, N=[res_x, res_y], base_level=3)

# undeformed coordinates (cell center and faces)
X = ti.Vector.field(2, float, shape=(res_x, res_y))
X_horizontal = ti.Vector.field(2, float, shape=(res_x + 1, res_y))
X_vertical = ti.Vector.field(2, float, shape=(res_x, res_y + 1))
center_coords_func(X, dx)
horizontal_coords_func(X_horizontal, dx)
vertical_coords_func(X_vertical, dx)

# back flow map
T_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))  # d_psi / d_x
T_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))  # d_psi / d_y
psi_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))  # x coordinate
psi_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))  # y coordinate

# P2G weight storage
p2g_weight_x = ti.field(float, shape=(res_x + 1, res_y))
p2g_weight_y = ti.field(float, shape=(res_x, res_y + 1))

# velocity storage
u = ti.Vector.field(2, float, shape=(res_x, res_y))
w = ti.field(float, shape=(res_x, res_y))
u_x = ti.field(float, shape=(res_x + 1, res_y))
u_y = ti.field(float, shape=(res_x, res_y + 1))

# particle storage
initial_particle_num = res_x * res_y * particles_per_cell
particle_num = initial_particle_num * total_particles_num_ratio
current_particle_num = ti.field(int, shape=1)
particles_pos = ti.Vector.field(2, float, shape=particle_num)
particles_pos_backup = ti.Vector.field(2, float, shape=particle_num)
particles_imp = ti.Vector.field(2, float, shape=particle_num)
particles_init_imp = ti.Vector.field(2, float, shape=particle_num)
particles_init_imp_grad_m = ti.Vector.field(2, float, shape=particle_num)

# back flow map
T_x = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
T_y = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y
T_x_init = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
T_y_init = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y
T_x_grad_m = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
T_y_grad_m = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y
psi = ti.Vector.field(2, float, shape=particle_num)  # x coordinate

# paticles in each cell
cell_max_particle_num = int(cell_max_particle_num_ratio * particles_per_cell)
cell_particle_num = ti.field(int, shape=(res_x, res_y))
cell_particles_id = ti.field(int, shape=(res_x, res_y, cell_max_particle_num))

# APIC
C_x = ti.Vector.field(2, float, shape=particle_num)
C_y = ti.Vector.field(2, float, shape=particle_num)
init_C_x = ti.Vector.field(2, float, shape=particle_num)
init_C_y = ti.Vector.field(2, float, shape=particle_num)

# some helper storage
tmp_u_x = ti.field(float, shape=(res_x + 1, res_y))
tmp_u_y = ti.field(float, shape=(res_x, res_y + 1))

count = ti.field(int, shape=particle_num)

# CFL related
max_speed = ti.field(float, shape=())
dts = torch.zeros(1)

# smoke
init_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
tmp_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
err_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))

@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template()):
    max_speed[None] = 1.e-3  # avoid dividing by zero
    for i, j in ti.ndrange(res_x, res_y):
        u = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        v = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        speed = ti.sqrt(u ** 2 + v ** 2)
        ti.atomic_max(max_speed[None], speed)

@ti.kernel
def calc_max_imp_particles(particles_imp: ti.template()):
    max_speed[None] = 1.e-3  # avoid dividing by zero
    for i in particles_imp:
        # if particles_active[i] == 1:
            imp = particles_imp[i].norm()
            ti.atomic_max(max_speed[None], imp)


# set to undeformed config
@ti.kernel
def reset_to_identity_grid(psi_x: ti.template(), psi_y: ti.template(), T_x: ti.template(), T_y: ti.template()):
    for i, j in psi_x:
        psi_x[i, j] = X_horizontal[i, j]
    for i, j in psi_y:
        psi_y[i, j] = X_vertical[i, j]
    for i, j in T_x:
        T_x[i, j] = ti.Vector.unit(2, 0)
    for i, j in T_y:
        T_y[i, j] = ti.Vector.unit(2, 1)


@ti.kernel
def reset_to_identity(psi: ti.template(), T_x: ti.template(), T_y: ti.template()):
    for i in psi:
        psi[i] = particles_pos[i]
    for i in T_x:
        T_x[i] = ti.Vector.unit(2, 0)
    for i in T_y:
        T_y[i] = ti.Vector.unit(2, 1)

@ti.kernel
def reset_T_to_identity(T_x: ti.template(), T_y: ti.template()):
    for i in T_x:
        T_x[i] = ti.Vector.unit(2, 0)
    for i in T_y:
        T_y[i] = ti.Vector.unit(2, 1)


@ti.kernel
def check_psi_and_X(curr_step: int, psi: ti.template(), particles_pos_backup: ti.template()):
    different_X_psi_num = 0
    for i in particles_pos_backup:
        # if particles_active[i] == 1:
            diff = psi[i] - particles_pos_backup[i]
            if diff.norm() > 1e-3:
                different_X_psi_num += 1

    print(f'Step {curr_step}: {different_X_psi_num}/{particle_num} different psi and X')


# @ti.kernel
# def update_C_one_step(T_x: ti.template(), T_y: ti.template(), C_x: ti.template(), C_y: ti.template()):
#     for i in C_x:
#         if particles_active[i] == 1:
#             # new_C_x = ti.Vector([T_x[i] @ ti.Vector([C_x[i][0], C_y[i][0]]), T_x[i] @ ti.Vector([C_x[i][1], C_y[i][1]])])
#             # new_C_y = ti.Vector([T_y[i] @ ti.Vector([C_x[i][0], C_y[i][0]]), T_y[i] @ ti.Vector([C_x[i][1], C_y[i][1]])])
#             # C_x[i] = new_C_x
#             # C_y[i] = new_C_y
#             T = ti.Matrix.rows([T_x[i], T_y[i]]).transpose()
#             C = ti.Matrix.rows([C_x[i], C_y[i]])
#             new_C_x = grad_T_x[i] @ interped_imp[i] + T_x[i] @ (C @ T)
#             new_C_y = grad_T_y[i] @ interped_imp[i] + T_y[i] @ (C @ T)
#             # new_C_x = T_x[i] @ (C @ T)
#             # new_C_y = T_y[i] @ (C @ T)
#             C_x[i] = new_C_x
#             C_y[i] = new_C_y


# curr step should be in range(reinit_every)
# def backtrack_psi_grid(curr_step, velocity_buffer_dir):
#     copy_to(T_x, T_x_backup)
#     copy_to(T_y, T_y_backup)
#     reset_to_identity(psi, T_x, T_y)
#     # first step is done on grid
#     # RK4(psi, T_x, T_y, u_x, u_y, dts[curr_step].item())
#     RK4(psi, T_x, T_y, u_x, u_y, dts[0].item())
#
#     update_C_one_step(T_x, T_y, C_x, C_y)
#
#     if forward_update_T:
#         copy_to(T_x_backup, T_x)
#         copy_to(T_y_backup, T_y)
#
#         # RK4_T_forward(phi_backup, T_x, T_y, u_x, u_y, dts[curr_step].item())
#         RK4_T_forward(phi_backup, T_x, T_y, u_x, u_y, dts[0].item())
#         copy_to(particles_pos_backup, psi)
#     else:
#
#         # check_psi_and_X(curr_step, psi, particles_pos_backup)
#         # later steps are done on neural
#         for step in reversed(range(curr_step)):
#             RK4_neural(psi, T_x, T_y, step, velocity_buffer_dir)
#
#         check_psi_and_X(curr_step, psi, particles_pos_backup)


# def update_C(dt: float):
#     copy_to(T_x, T_x_backup)
#     copy_to(T_y, T_y_backup)
#     reset_to_identity(psi, T_x, T_y)
#     # first step is done on grid
#     RK4(psi, T_x, T_y, u_x, u_y, dt)
#
#     update_C_one_step(T_x, T_y, C_x, C_y)
#
#     copy_to(T_x_backup, T_x)
#     copy_to(T_y_backup, T_y)


def stretch_T_and_advect_particles(particles_pos, T_x, T_y, u_x, u_y, dt):
    RK4_T_forward(particles_pos, T_x, T_y, u_x, u_y, dt, 1)
    # copy_to(particles_pos_backup, psi)

def stretch_T(particles_pos, T_x, T_y, u_x, u_y, dt):
    RK4_T_forward(particles_pos, T_x, T_y, u_x, u_y, dt, 0)


# curr step should be in range(reinit_every)
# def march_phi_grid(curr_step):
#     RK4(phi, F_x, F_y, u_x, u_y, -1 * dts[curr_step].item())


@ti.kernel
def RK4_grid(psi_x: ti.template(), T_x: ti.template(),
             u_x0: ti.template(), u_y0: ti.template(), dt: float):
    neg_dt = -1 * dt  # travel back in time
    for i, j in psi_x:
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x[i, j], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[i, j]  # time derivative of T
        # prepare second
        psi_x1 = psi_x[i, j] + 0.5 * neg_dt * u1  # advance 0.5 steps
        T_x1 = T_x[i, j] + 0.5 * neg_dt * dT_x_dt1
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1  # time derivative of T
        # prepare third
        psi_x2 = psi_x[i, j] + 0.5 * neg_dt * u2  # advance 0.5 again
        T_x2 = T_x[i, j] + 0.5 * neg_dt * dT_x_dt2
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2  # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[i, j] + 1.0 * neg_dt * u3
        T_x3 = T_x[i, j] + 1.0 * neg_dt * dT_x_dt3  # advance 1.0
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3  # time derivative of T
        # final advance
        psi_x[i, j] = psi_x[i, j] + neg_dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[i, j] = T_x[i, j] + neg_dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full


@ti.kernel
def RK4(psi: ti.template(), T_x: ti.template(), T_y: ti.template(),
        u_x0: ti.template(), u_y0: ti.template(), dt: float):
    neg_dt = -1 * dt  # travel back in time
    for i in psi:
        # if particles_active[i] == 1:
            # first
            u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi @ T_y[i]  # time derivative of T
            # prepare second
            psi_x1 = psi[i] + 0.5 * neg_dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] + 0.5 * neg_dt * dT_x_dt1
            T_y1 = T_y[i] + 0.5 * neg_dt * dT_y_dt1
            # second
            u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi @ T_y1  # time derivative of T
            # prepare third
            psi_x2 = psi[i] + 0.5 * neg_dt * u2  # advance 0.5 again
            T_x2 = T_x[i] + 0.5 * neg_dt * dT_x_dt2
            T_y2 = T_y[i] + 0.5 * neg_dt * dT_y_dt2
            # third
            u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
            dT_x_dt3 = grad_u_at_psi @ T_x2  # time derivative of T
            dT_y_dt3 = grad_u_at_psi @ T_y2  # time derivative of T
            # prepare fourth
            psi_x3 = psi[i] + 1.0 * neg_dt * u3
            T_x3 = T_x[i] + 1.0 * neg_dt * dT_x_dt3  # advance 1.0
            T_y3 = T_y[i] + 1.0 * neg_dt * dT_y_dt3  # advance 1.0
            # fourth
            u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
            dT_x_dt4 = grad_u_at_psi @ T_x3  # time derivative of T
            dT_y_dt4 = grad_u_at_psi @ T_y3  # time derivative of T
            # final advance
            psi[i] = psi[i] + neg_dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
            T_x[i] = T_x[i] + neg_dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
            T_y[i] = T_y[i] + neg_dt * 1. / 6 * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full


@ti.kernel
def RK4_T_forward(psi: ti.template(), T_x: ti.template(), T_y: ti.template(),
                  u_x0: ti.template(), u_y0: ti.template(), dt: float, advect_psi: int):
    for i in psi:
        # if particles_active[i] == 1:
            # first
            u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi.transpose() @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi.transpose() @ T_y[i]  # time derivative of T
            # prepare second
            psi_x1 = psi[i] + 0.5 * dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] - 0.5 * dt * dT_x_dt1
            T_y1 = T_y[i] - 0.5 * dt * dT_y_dt1
            # second
            u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi.transpose() @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi.transpose() @ T_y1  # time derivative of T
            # prepare third
            psi_x2 = psi[i] + 0.5 * dt * u2  # advance 0.5 again
            T_x2 = T_x[i] - 0.5 * dt * dT_x_dt2
            T_y2 = T_y[i] - 0.5 * dt * dT_y_dt2
            # third
            u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
            dT_x_dt3 = grad_u_at_psi.transpose() @ T_x2  # time derivative of T
            dT_y_dt3 = grad_u_at_psi.transpose() @ T_y2  # time derivative of T
            # prepare fourth
            psi_x3 = psi[i] + 1.0 * dt * u3
            T_x3 = T_x[i] - 1.0 * dt * dT_x_dt3  # advance 1.0
            T_y3 = T_y[i] - 1.0 * dt * dT_y_dt3  # advance 1.0
            # fourth
            u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
            dT_x_dt4 = grad_u_at_psi.transpose() @ T_x3  # time derivative of T
            dT_y_dt4 = grad_u_at_psi.transpose() @ T_y3  # time derivative of T
            # final advance
            if advect_psi:
                psi[i] = psi[i] + dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
            T_x[i] = T_x[i] - dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
            T_y[i] = T_y[i] - dt * 1. / 6 * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full


# @ti.kernel
# def get_interped_imp_and_grad_m(dt: float):
#     for i in particles_pos:
#         if particles_active[i] == 1:
#             # new_C_x, new_C_y = interp_u_MAC_grad_updated_imp(u_x, u_y, imp_x, imp_y, phi_x_grid, phi_y_grid, particles_pos[i], dx)
#             # C_x[i] = new_C_x
#             # C_y[i] = new_C_y
#
#             a, b, new_C_x, new_C_y, interped_imp_x, interped_imp_y = interp_u_MAC_imp_and_grad_imp(u_x, u_y, imp_x,
#                                                                                                    imp_y,
#                                                                                                    particles_pos[i], dx)
#             C_x[i] = new_C_x
#             C_y[i] = new_C_y
#             interped_imp[i] = ti.Vector([interped_imp_x, interped_imp_y])


@ti.kernel
def advect_particles(dt: float):
    # different_x_phi_num = 0

    for i in particles_pos:
        # if particles_active[i] == 1:
            # new_C_x, new_C_y = interp_u_MAC_grad_updated_imp(u_x, u_y, imp_x, imp_y, phi_x_grid, phi_y_grid, particles_pos[i], dx)
            # C_x[i] = new_C_x
            # C_y[i] = ne.w_C_y

            v1, _ = interp_u_MAC_grad(u_x, u_y, particles_pos[i], dx)
            p2 = particles_pos[i] + v1 * dt * 0.5
            v2, _ = interp_u_MAC_grad(u_x, u_y, p2, dx)
            p3 = particles_pos[i] + v2 * dt * 0.5
            v3, _ = interp_u_MAC_grad(u_x, u_y, p3, dx)
            p4 = particles_pos[i] + v3 * dt
            v4, _ = interp_u_MAC_grad(u_x, u_y, p4, dx)
            particles_pos[i] += (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt

            # diff = particles_pos[i] - phi[i]
            # if diff.norm() > 1e-5:
            #     different_x_phi_num += 1

    # print(f'{different_x_phi_num}/{particle_num} different phi and x')


# @ti.kernel
# def get_grad_T(dt: float):
#     for i in particles_pos:
#         if particles_active[i] == 1:
#             C_grad_T_x, C_grad_T_y = interp_MAC_grad_T(T_x_grid, T_y_grid, particles_pos[i], dx)
#             grad_T_x[i] = C_grad_T_x
#             grad_T_y[i] = C_grad_T_y


@ti.kernel
def limit_particles_in_boundary():
    for i in particles_pos:
        # if particles_active[i] == 1:
            if particles_pos[i][0] < left_boundary:
                particles_pos[i][0] = left_boundary
            if particles_pos[i][0] > right_boundary:
                particles_pos[i][0] = right_boundary

            if particles_pos[i][1] < lower_boundary:
                particles_pos[i][1] = lower_boundary
            if particles_pos[i][1] > upper_boundary:
                particles_pos[i][1] = upper_boundary


# def RK4_neural(psi, T_x, T_y, step, velocity_buffer_dir):
#     # convert coords to torch
#     x_coord_flat = X_horizontal.to_torch().view(-1, 2)
#     y_coord_flat = X_vertical.to_torch().view(-1, 2)
#     # evaluate buffer for x and y
#     if use_neural:
#         u_x_torch = nb.pred_u_x(x_coord_flat, nb.mid_ts[step]).view(res_x + 1, res_y)
#         u_y_torch = nb.pred_u_y(y_coord_flat, nb.mid_ts[step]).view(res_x, res_y + 1)
#     else:
#         u_x_torch = torch.from_numpy(np.load(os.path.join(velocity_buffer_dir, "vel_x_numpy_" + str(step) + ".npy")))
#         u_y_torch = torch.from_numpy(np.load(os.path.join(velocity_buffer_dir, "vel_y_numpy_" + str(step) + ".npy")))
#     # convert result back to taichi
#     tmp_u_x.from_torch(u_x_torch)
#     tmp_u_y.from_torch(u_y_torch)
#     RK4(psi, T_x, T_y, tmp_u_x, tmp_u_y, dts[step].item())


# u_x0, u_y0 are the initial time quantities
# u_x1, u_y1 are the current time quantities (to be modified)
@ti.kernel
def advect_u(u_x0: ti.template(), u_y0: ti.template(),
             u_x1: ti.template(), u_y1: ti.template(),
             T_x: ti.template(), T_y: ti.template(),
             psi_x: ti.template(), psi_y: ti.template(), dx: float):
    # horizontal velocity
    for i, j in u_x1:
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x[i, j], dx)
        u_x1[i, j] = T_x[i, j].dot(u_at_psi)
    # vertical velocity
    for i, j in u_y1:
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, psi_y[i, j], dx)
        u_y1[i, j] = T_y[i, j].dot(u_at_psi)


@ti.kernel
def advect_smoke(smoke0: ti.template(), smoke1: ti.template(),
                 psi_x: ti.template(), psi_y: ti.template(), dx: float):
    # horizontal velocity
    for i, j in ti.ndrange(res_x, res_y):
        psi_c = 0.25 * (psi_x[i, j] + psi_x[i + 1, j] + psi_y[i, j] + psi_y[i, j + 1])
        smoke1[i, j] = interp_1(smoke0, psi_c, dx)


# def diffuse_sizing():
#     for _ in range(1024):
#         diffuse_grid(sizing, tmp_sizing)


# @ti.kernel
# def compute_dT_dx(grad_T_init_x: ti.template(), grad_T_init_y: ti.template(), T_x: ti.template(),
#                           T_y: ti.template(), particles_init_pos: ti.template(),
#                           cell_particles_id: ti.template(), cell_particle_num: ti.template()):
#
#     for i in grad_T_init_x:
#         weight = 0.
#         # if particles_active[i] == 1:
#         grad_T_init_x[i] = ti.Matrix.zero(float, 2, 2)
#         grad_T_init_y[i] = ti.Matrix.zero(float, 2, 2)
#
#         base_cell_id = int(particles_init_pos[i] / dx)
#
#         for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
#             # count[i] += 1
#             # neighbor_cell_id = base_cell_id + offset
#             # for k in ti.static(ti.ndrange((0, cell_max_particle_num))):
#             neighbor_cell_id = base_cell_id + offset
#             if 0 <= neighbor_cell_id[0] < res_x and 0 <= neighbor_cell_id[1] < res_y:
#                 for k in range(0, cell_particle_num[neighbor_cell_id]):
#                     # for k in ti.static(ti.ndrange((0, cell_particle_num[neighbor_cell_id]))):
#                     #     condition = k < cell_particle_num[neighbor_cell_id]
#                     #     if condition[0]:
#                     if k < cell_max_particle_num:
#                         neighbor_particle_id = cell_particles_id[neighbor_cell_id[0], neighbor_cell_id[1], k]
#                         neighbor_particle_pos = particles_init_pos[neighbor_particle_id]
#                         # if particles_active[neighbor_particle_id] == 1:
#                         dist_x = particles_init_pos[i][0] - neighbor_particle_pos[0]
#                         dist_y = particles_init_pos[i][1] - neighbor_particle_pos[1]
#                         # dist_x = id_x - neighbor_id_x
#                         # dist_y = id_y - neighbor_id_y
#                         dw_x = 1. / dx * dN_2(dist_x / dx) * N_2(dist_y / dx)
#                         dw_y = 1. / dx * N_2(dist_x / dx) * dN_2(dist_y / dx)
#                         # dw_x = 1. / dist_between_neighbor * dN_2(dist_x) * N_2(dist_y)
#                         # dw_y = 1. / dist_between_neighbor * N_2(dist_x) * dN_2(dist_y)
#                         dw = ti.Vector([dw_x, dw_y])
#
#                         weight += N_2(dist_x / dx) * N_2(dist_y / dx)
#
#                         T = ti.Matrix.cols([T_x[neighbor_particle_id], T_y[neighbor_particle_id]])
#                         grad_T_init_x[i] += dw.outer_product(T[0, :])
#                         grad_T_init_y[i] += dw.outer_product(T[1, :])
#
#             if weight > 0:
#                 grad_T_init_x[i] /= weight
#                 grad_T_init_y[i] /= weight


            # neighbor_id_x = ti.max(0, ti.min(id_x + 1, particles_x_num - 1))
            # neighbor_id_y = ti.max(0, ti.min(id_y + 1, particles_y_num - 1))
            # grad_T_init_x_dx = (T_x[id_y * particles_x_num + neighbor_id_x] - T_x[i]) / dist_between_neighbor
            # grad_T_init_x_dy = (T_x[neighbor_id_y * particles_x_num + id_x] - T_x[i]) / dist_between_neighbor
            # grad_T_init_x[i] = ti.Matrix.rows([grad_T_init_x_dx, grad_T_init_x_dy])
            # grad_T_init_y_dx = (T_y[id_y * particles_x_num + neighbor_id_x] - T_y[i]) / dist_between_neighbor
            # grad_T_init_y_dy = (T_y[neighbor_id_y * particles_x_num + id_x] - T_y[i]) / dist_between_neighbor
            # grad_T_init_y[i] = ti.Matrix.rows([grad_T_init_y_dx, grad_T_init_y_dy])



@ti.kernel
def update_particles_imp(particles_imp: ti.template(), particles_init_imp: ti.template(),
                         T_x: ti.template(), T_y: ti.template()):
    for i in particles_imp:
        # if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            # particles_imp[i] = ti.Vector([T_x[i] @ particles_init_imp[i], T_y[i] @ particles_init_imp[i]])
            particles_imp[i] = T @ particles_init_imp[i]

@ti.kernel
def update_particles_grad_m(C_x: ti.template(), C_y: ti.template(), init_C_x: ti.template(), init_C_y: ti.template(),
                            T_x: ti.template(), T_y: ti.template()):
    for i in C_x:
        # if particles_active[i] == 1:
            T = ti.Matrix.rows([T_x[i], T_y[i]])
            T_transpose = ti.Matrix.cols([T_x[i], T_y[i]])
            init_C = ti.Matrix.rows([init_C_x[i], init_C_y[i]])
            T_init_C_T = T_transpose @ (init_C @ T)
            C_x[i] = T_init_C_T[0, :]
            C_y[i] = T_init_C_T[1, :]

@ti.kernel
def update_T(T_x: ti.template(), T_y: ti.template(), T_x_init: ti.template(), T_y_init: ti.template(),
             T_x_grad_m: ti.template(), T_y_grad_m: ti.template()):
    for i in T_x:
        T_grad_m = ti.Matrix.cols([T_x_grad_m[i], T_y_grad_m[i]])
        T_init = ti.Matrix.cols([T_x_init[i], T_y_init[i]])
        T = T_grad_m @ T_init
        T_x[i] = T[:, 0]
        T_y[i] = T[:, 1]

@ti.kernel
def P2G(particles_imp: ti.template(), particles_pos: ti.template(), u_x: ti.template(), u_y: ti.template(),
        psi: ti.template(), psi_x_grid: ti.template(), psi_y_grid: ti.template(),
        p2g_weight_x: ti.template(), p2g_weight_y: ti.template()):
    u_x.fill(0.0)
    u_y.fill(0.0)
    # imp_x.fill(0.0)
    # imp_y.fill(0.0)
    psi_x_grid.fill(0.0)
    psi_y_grid.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)

    for i in particles_imp:
        # if particles_active[i] == 1:
            # horizontal impulse
            pos = particles_pos[i] / dx
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                    dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                    p2g_weight_x[face_id] += weight
                    delta = C_x[i].dot(dpos)
                    # print(particles_imp[i][0], weight, delta)
                    if use_APIC:
                        u_x[face_id] += (particles_imp[i][0] + delta) * weight
                    else:
                        u_x[face_id] += (particles_imp[i][0]) * weight

                    psi_x_grid[face_id] += psi[i] * weight

            # vertical impulse
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                    weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                    dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                    p2g_weight_y[face_id] += weight
                    delta = C_y[i].dot(dpos)
                    if use_APIC:
                        u_y[face_id] += (particles_imp[i][1] + delta) * weight
                    else:
                        u_y[face_id] += (particles_imp[i][1]) * weight

                    psi_y_grid[face_id] += psi[i] * weight

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            scale = 1. / p2g_weight_x[I]
            u_x[I] *= scale
            psi_x_grid[I] *= scale
            # imp_x[I] = u_x[I]

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            scale = 1. / p2g_weight_y[I]
            u_y[I] *= scale
            psi_y_grid[I] *= scale
            # imp_y[I] = u_y[I]


# @ti.kernel
# def reseed_particles():
#     grid_particle_num.fill(0)
#     for i in particles_pos:
#         if particles_active[i] == 1:
#             grid_id = int(particles_pos[i] / dx)
#             grid_particle_num[grid_id] += 1
#
#     for I in ti.grouped(grid_particle_num):
#         while grid_particle_num[I] < min_particles_per_cell:
#             if current_particle_num[0] >= particle_num:
#                 print("Error!!! Current particle num get max!!!")
#             else:
#                 new_particle_id = ti.atomic_add(current_particle_num[0], 1)
#                 particles_pos[new_particle_id] = X[I] + ti.Vector([(ti.random() - 0.5) for _ in ti.static(range(2))]) * dx
#                 particles_active[new_particle_id] = 1
#                 grid_particle_num[I] += 1
#                 print(f"Add particle {new_particle_id}")

# @ti.kernel
# def reseed_particles():
#     grid_particle_num.fill(0)
#     for i in particles_pos:
#         if particles_active[i] == 1:
#             grid_id = int(particles_pos[i] / dx)
#             grid_particle_num[grid_id] += 1
#
#     for I in ti.grouped(grid_particle_num):
#         while grid_particle_num[I] < min_particles_per_cell:
#             if current_particle_num[0] >= particle_num:
#                 print("Error!!! Current particle num get max!!!")
#             else:
#                 new_particle_id = ti.atomic_add(current_particle_num[0], 1)
#                 particles_pos[new_particle_id] = X[I] + ti.Vector(
#                     [(ti.random() - 0.5) for _ in ti.static(range(2))]) * dx
#                 particles_active[new_particle_id] = 1
#                 grid_particle_num[I] += 1
#                 print(f"Add particle {new_particle_id}")
#
#     grid_lock.fill(0)
#     for i in particles_pos:
#         if particles_active[i] == 1:
#             grid_id = int(particles_pos[i] / dx)
#             try_num = 0
#             while grid_particle_num[grid_id] > max_particles_per_cell and try_num < max_delete_particle_try_num:
#                 try_num += 1
#                 if grid_lock[grid_id] == 0:
#                     old_lock = ti.atomic_or(grid_lock[grid_id], 1)
#                     if old_lock == 0:
#                         if grid_particle_num[grid_id] > max_particles_per_cell:
#                             particles_active[i] = 0
#                             grid_particle_num[grid_id] -= 1
#                             print(f"Delete particle {i}")
#
#                         ti.atomic_and(grid_lock[grid_id], 0)
#                         break
#
#             if try_num >= max_delete_particle_try_num:
#                 print("Error!!! Delete particle try num get max!!!")


# main function
def main(from_frame=0, testing=False):
    from_frame = max(0, from_frame)
    # create some folders
    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    if from_frame <= 0:
        remove_everything_in(logsdir)

    vortdir = 'vorticity'
    vortdir = os.path.join(logsdir, vortdir)
    os.makedirs(vortdir, exist_ok=True)
    smokedir = 'smoke'
    smokedir = os.path.join(logsdir, smokedir)
    os.makedirs(smokedir, exist_ok=True)
    ckptdir = 'ckpts'
    ckptdir = os.path.join(logsdir, ckptdir)
    os.makedirs(ckptdir, exist_ok=True)
    levelsdir = 'levels'
    levelsdir = os.path.join(logsdir, levelsdir)
    os.makedirs(levelsdir, exist_ok=True)
    modeldir = 'model'  # saves the model
    modeldir = os.path.join(logsdir, modeldir)
    os.makedirs(modeldir, exist_ok=True)
    velocity_buffer_dir = 'velocity_buffer'
    velocity_buffer_dir = os.path.join(logsdir, velocity_buffer_dir)
    os.makedirs(velocity_buffer_dir, exist_ok=True)
    particles_dir = 'particles'
    particles_dir = os.path.join(logsdir, particles_dir)
    os.makedirs(particles_dir, exist_ok=True)

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
        leapfrog_vel_func(u, X)  # set init_m
        split_central_vector(u, u_x, u_y)
        solver.Poisson(u_x, u_y)
        # initialize smoke
        stripe_func(smoke, X, 0.25, 0.48)
    else:
        u_x.from_numpy(np.load(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame) + ".npy")))
        u_y.from_numpy(np.load(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame) + ".npy")))
        smoke.from_numpy(np.load(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame) + ".npy")))

    # active_init_particles(particles_active, initial_particle_num)
    init_particles_pos_uniform(particles_pos, X, res_x, particles_per_cell, dx,
                               particles_per_cell_axis, dist_between_neighbor)
    init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y,
                              init_C_x, init_C_y, dx)
    reset_T_to_identity(T_x_grad_m, T_y_grad_m)
    backup_particles_pos(particles_pos, particles_pos_backup)
    init_particles_imp(particles_imp, particles_init_imp, particles_pos, u_x, u_y, init_C_x, init_C_y, dx)
    current_particle_num[0] = initial_particle_num

    # for visualization
    get_central_vector(u_x, u_y, u)
    curl(u, w, dx)
    w_numpy = w.to_numpy()
    w_max = max(np.abs(w_numpy.max()), np.abs(w_numpy.min()))
    w_min = -1 * w_max
    write_field(w_numpy, vortdir, from_frame, particles_pos.to_numpy() / dx, vmin=w_min,
                vmax=w_max,
                plot_particles=plot_particles, dpi=dpi_vor)
    # write_particles(w_numpy, particles_dir, from_frame, particles_pos.to_numpy() / dx, vmin=w_min, vmax=w_max)
    write_image(smoke.to_numpy(), smokedir, from_frame)

    if save_ckpt:
        np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame)), u_x.to_numpy())
        np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame)), u_y.to_numpy())
        np.save(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame)), smoke.to_numpy())

    sub_t = 0.  # the time since last reinit
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0  # number of reinitializations already performed
    i = -1
    ik = 0

    frame_times = np.zeros(total_steps)
    # frame_energy = np.zeros(total_frames + 1)
    # u_x_np = u_x.to_numpy()
    # u_y_np = u_y.to_numpy()
    # energy = np.sum(u_x_np ** 2) / ((res_x + 1) * res_y) + np.sum(u_y_np ** 2) / (res_x * (res_y + 1))
    # frame_energy[frame_idx] = energy
    # print(f'energy of frame {frame_idx} is {energy}')

    while True:
        start_time = time.time()
        i += 1
        j = i % reinit_every
        # k = i % reinit_every_grad_m
        i_next = i + 1
        j_next = i_next % reinit_every
        print("[Simulate] Running step: ", i, " / substep: ", j)

        # determine dt
        calc_max_speed(u_x, u_y)  # saved to max_speed[None]
        curr_dt = CFL * dx / max_speed[None]

        if save_frame_each_step:
            output_frame = True
            frame_idx += 1
        else:
            if sub_t + curr_dt >= visualize_dt:  # if over
                curr_dt = visualize_dt - sub_t
                sub_t = 0.  # empty sub_t
                frame_idx += 1
                print(f'Visualized frame {frame_idx}')
                output_frame = True
            else:
                sub_t += curr_dt
                print(f'Visualize time {sub_t}/{visualize_dt}')
                output_frame = False
        # dts[j] = curr_dt
        dts[0] = curr_dt
        # done dt

        # reinitialize flow map if j == 0:
        if j == 0:
            print("[Simulate] Reinitializing the flow map for the: ", num_reinits, " time!")

            if use_reseed_particles:
                # reseed_particles()
                pass
            else:
                if reinit_particle_pos or i == 0:
                    init_particles_pos_uniform(particles_pos, X, res_x, particles_per_cell, dx,
                                               particles_per_cell_axis, dist_between_neighbor)
                    # init_particles_pos(particles_pos, X, res_x, particles_per_cell, dx)
                    # init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y,
                    #                           init_C_x, init_C_y, dx)
                    # reset_T_to_identity(T_x_grad_m, T_y_grad_m)

            ik = i

            backup_particles_pos(particles_pos, particles_pos_backup)
            init_particles_imp(particles_imp, particles_init_imp, particles_pos, u_x, u_y, init_C_x, init_C_y, dx)
            # reset_to_identity(phi, F_x, F_y)
            reset_to_identity(psi, T_x, T_y)
            reset_T_to_identity(T_x_init, T_y_init)
            # copy_to(u_x, init_u_x)
            # copy_to(u_y, init_u_y)
            # copy_to(u_x, imp_x)
            # copy_to(u_y, imp_y)
            # copy_to(smoke, init_smoke)
            # copy_to(particles_imp, particles_init_imp)
            # init_particles_imp(particles_imp, particles_init_imp, particles_pos, u_x, u_y, dx)
            # reinit neural buffer
            get_central_vector(u_x, u_y, u)
            # if use_neural:
            #     sizing_function(u, sizing, dx)
            #     diffuse_sizing()
            #     nb.reinit(sizing)
            #     nb.set_magnitude_scale(max_speed[None])
            #     print("[Neural Buffer] New magnitude scale: ", nb.magnitude_scale)
            #     # for visualization
            #     nb.paint_active(levels_display)
            # increment
            num_reinits += 1

            print(f'reinit long-distance mapping at step {i}')

        k = (i - ik) % reinit_every_grad_m
        if k == 0:
            init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y,
                                      init_C_x, init_C_y, dx)
            reset_T_to_identity(T_x_grad_m, T_y_grad_m)
            copy_to(T_x, T_x_init)
            copy_to(T_y, T_y_init)

            print(f'reinit short-distance mapping at step {i}')

            # copy_to(particles_pos, particles_init_pos_grad_m)



        # start midpoint
        # reset_to_identity(psi, T_x, T_y)
        # RK4(psi, T_x, T_y, u_x, u_y, 0.5 * curr_dt)
        if use_midpoint_vel:
            reset_to_identity_grid(psi_x_grid, psi_y_grid, T_x_grid, T_y_grid)
            RK4_grid(psi_x_grid, T_x_grid, u_x, u_y, 0.5 * curr_dt)
            RK4_grid(psi_y_grid, T_y_grid, u_x, u_y, 0.5 * curr_dt)
            copy_to(u_x, tmp_u_x)
            copy_to(u_y, tmp_u_y)
            advect_u(tmp_u_x, tmp_u_y, u_x, u_y, T_x_grid, T_y_grid, psi_x_grid, psi_y_grid, dx)
            solver.Poisson(u_x, u_y)
        # done midpoint

        # store u to nb
        # if will be reinitialize next substep, then no need.
        # if not use_neural:
        #     np.save(os.path.join(velocity_buffer_dir, "vel_x_numpy_" + str(j)), u_x.to_numpy())
        #     np.save(os.path.join(velocity_buffer_dir, "vel_y_numpy_" + str(j)), u_y.to_numpy())


        # copy_to(particles_pos, phi_backup)
        # get_interped_imp_and_grad_m(curr_dt)
        # advect_particles(curr_dt)

        # evolve grad m
        # copy_to(particles_pos, particles_pos_backup)
        stretch_T_and_advect_particles(particles_pos, T_x_grad_m, T_y_grad_m, u_x, u_y, curr_dt)
        # print(f'count: {np.sum(count.to_numpy())}')
        # print(f'grad_T_init_x: {np.sum(grad_T_init_x.to_numpy())}')
        update_particles_grad_m(C_x, C_y, init_C_x, init_C_y, T_x_grad_m, T_y_grad_m)


        # evolve m
        # copy_to(particles_pos_backup, particles_pos)
        # stretch_T_and_advect_particles(particles_pos, T_x, T_y, u_x, u_y, curr_dt)
        # get_grad_T(curr_dt)
        # update_C(curr_dt)

        # backtrack_psi_grid(j, velocity_buffer_dir)

        # limit_particles_in_boundary()
        update_T(T_x, T_y, T_x_init, T_y_init, T_x_grad_m, T_y_grad_m)
        update_particles_imp(particles_imp, particles_init_imp, T_x, T_y)
        # calc_max_imp_particles(particles_imp)  # saved to max_speed[None]
        # print(f'1.4 max imp {i}: {max_speed[None]}')

        P2G(particles_imp, particles_pos, u_x, u_y, psi, psi_x_grid, psi_y_grid, p2g_weight_x, p2g_weight_y)
        solver.Poisson(u_x, u_y)
        # advect_smoke(init_smoke, smoke, psi_x_grid, psi_y_grid, dx)

        end_time = time.time()
        frame_time = end_time - start_time
        print(f'frame execution time: {frame_time:.6f} seconds')

        if use_total_steps:
            frame_times[i] = frame_time

        # advect_u(init_u_x, init_u_y, u_x, u_y, T_x, T_y, psi_x, psi_y, dx)
        # advect_smoke(init_smoke, smoke, psi_x, psi_y, dx)
        # # # Begin BFECC
        # advect_u(u_x, u_y, err_u_x, err_u_y, F_x, F_y, phi_x, phi_y, dx)
        # advect_smoke(smoke, err_smoke, phi_x, phi_y, dx)
        # add_fields(err_u_x, init_u_x, err_u_x, -1.) # subtract init_u_x from back_u_x
        # add_fields(err_u_y, init_u_y, err_u_y, -1.)
        # add_fields(err_smoke, init_smoke, err_smoke, -1.)
        # scale_field(err_u_x, 0.5, err_u_x) # halve error
        # scale_field(err_u_y, 0.5, err_u_y)
        # scale_field(err_smoke, 0.5, err_smoke)
        # advect_u(err_u_x, err_u_y, tmp_u_x, tmp_u_y, T_x, T_y, psi_x, psi_y, dx) # advect error (tmp_u_x is the advected error)
        # advect_smoke(err_smoke, tmp_smoke, psi_x, psi_y, dx) # advect error
        # add_fields(u_x, tmp_u_x, u_x, -1.) # subtract advected_err_x from u_x
        # add_fields(u_y, tmp_u_y, u_y, -1.) # subtract advected_err_y from u_y
        # add_fields(smoke, tmp_smoke, smoke, -1.)
        #
        # solver.Poisson(u_x, u_y)

        print("[Simulate] Done with step: ", i, " / substep: ", j, "\n", flush=True)

        if output_frame:
            # for visualization
            get_central_vector(u_x, u_y, u)
            # u_x_np = u_x.to_numpy()
            # u_y_np = u_y.to_numpy()
            # energy = np.sum(u_x_np ** 2) / ((res_x + 1) * res_y) + np.sum(u_y_np ** 2) / (res_x * (res_y + 1))
            # frame_energy[frame_idx] = energy
            # print(f'energy of frame {frame_idx} is {energy}')
            # write_image(levels_display[..., np.newaxis], levelsdir, frame_idx)
            curl(u, w, dx)
            w_numpy = w.to_numpy()
            write_field(w_numpy, vortdir, frame_idx, particles_pos.to_numpy() / dx, vmin=w_min, vmax=w_max,
                        plot_particles=plot_particles, dpi=dpi_vor)
            # write_particles(w_numpy, particles_dir, frame_idx, particles_pos.to_numpy() / dx, vmin=w_min, vmax=w_max)
            # write_image(smoke.to_numpy(), smokedir, frame_idx)
            if frame_idx % ckpt_every == 0 and save_ckpt:
                np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(frame_idx)), u_x.to_numpy())
                np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(frame_idx)), u_y.to_numpy())
                np.save(os.path.join(ckptdir, "smoke_numpy_" + str(frame_idx)), smoke.to_numpy())

            print("\n[Simulate] Finished frame: ", frame_idx, " in ", i - last_output_substep, "substeps \n\n")
            last_output_substep = i

            # if reached desired number of frames
            if frame_idx >= total_frames:
                # energy_dir = 'energy'
                # energy_dir = os.path.join(logsdir, energy_dir)
                # os.makedirs(f'{energy_dir}', exist_ok=True)
                # np.save(f'{energy_dir}/energy.npy', frame_energy)
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
        main(from_frame=from_frame)
    else:
        main(from_frame=from_frame, testing=testing)
    print("[Main] Complete")
