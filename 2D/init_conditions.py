from taichi_utils import *
from single_vortex import vortex_vel_func_point, dvxdx, dvxdy, dvydx, dvydy

# single vortex fields
@ti.func
def angular_vel_func(r, rad, strength):
    r = r + 1e-6
    linear_vel = strength * 1./r * (1.-ti.exp(-(r**2)/(rad**2)))
    return 1./r * linear_vel

@ti.func
def dvdr(r, rad, strength):
    r = r + 1e-6
    result = strength * (-(2*(1-ti.exp(-(r**2)/(rad**2))))/(r**3) + (2*ti.exp(-(r**2)/(rad**2)))/(r*(rad**2)))
    return result    

# vortex velocity field
@ti.kernel
def vortex_vel_func(vf: ti.template(), pf: ti.template()):
    c = ti.Vector([0.5, 0.5])
    for i, j in vf:
        p = pf[i, j] - c
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j].y = p.x
        vf[i, j].x = -p.y
        vf[i, j] *= angular_vel_func(r, 0.02, -0.01)

# vortex velocity field
@ti.kernel
def leapfrog_vel_func(vf: ti.template(), pf: ti.template()):
    c1 = ti.Vector([0.25, 0.62])
    c2 = ti.Vector([0.25, 0.38])
    c3 = ti.Vector([0.25, 0.74])
    c4 = ti.Vector([0.25, 0.26])
    cs = [c1, c2, c3, c4]
    w1 = -0.5
    w2 = 0.5
    w3 = -0.5
    w4 = 0.5
    for i, j in vf:
        # c1
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition
        # c2
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w2 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c3
        p = pf[i, j] - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w3 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c4
        p = pf[i, j] - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w4 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition

@ti.kernel
def single_vortex_vel_func(vf: ti.template(), pf: ti.template()):
    c1 = ti.Vector([0.5, 0.5])
    # c2 = ti.Vector([0.25, 0.38])
    # c3 = ti.Vector([0.25, 0.74])
    # c4 = ti.Vector([0.25, 0.26])
    cs = [c1]
    w1 = 0.5
    # w2 = 0.5
    # w3 = -0.5
    # w4 = 0.5
    for i, j in vf:
        # c1
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition
        # # c2
        # p = pf[i, j] - c2
        # r = ti.sqrt(p.x * p.x + p.y * p.y)
        # addition = angular_vel_func(r, 0.02, -0.01) * w2 * ti.Vector([-p.y, p.x])
        # vf[i, j] += addition
        # # c3
        # p = pf[i, j] - c3
        # r = ti.sqrt(p.x * p.x + p.y * p.y)
        # addition = angular_vel_func(r, 0.02, -0.01) * w3 * ti.Vector([-p.y, p.x])
        # vf[i, j] += addition
        # # c4
        # p = pf[i, j] - c4
        # r = ti.sqrt(p.x * p.x + p.y * p.y)
        # addition = angular_vel_func(r, 0.02, -0.01) * w4 * ti.Vector([-p.y, p.x])
        # vf[i, j] += addition
# # # #

# some shapes (checkerboards...)
@ti.kernel
def checkerboard_func(qf: ti.template(), pf: ti.template()):
    thickness = 0.1
    for i, j in qf:
        p = int(pf[i,j] / thickness)
        if (p.x + p.y) % 2 > 0:
            qf[i, j] = ti.Vector([1.0, 1.0, 1.0])
        else:
            qf[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def stripe_func(qf: ti.template(), pf: ti.template(), x_start: float, x_end: float):
    for i, j in qf:
        if x_start <= pf[i,j].x <= x_end and 0.15 <= pf[i,j].y <= 0.85:
            qf[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            qf[i, j] = ti.Vector([1.0, 1.0, 1.0])

@ti.kernel
def init_impulse(imp_x: ti.template(), imp_y: ti.template(), u_x: ti.template(), u_y: ti.template()):
    for I in ti.grouped(imp_x):
        imp_x[I] = u_x[I]
    for I in ti.grouped(imp_y):
        imp_y[I] = u_y[I]

@ti.kernel
def active_init_particles(particles_active: ti.template(), initial_particle_num: int):
    particles_active.fill(0)
    for i in particles_active:
        if i < initial_particle_num:
            particles_active[i] = 1

@ti.kernel
def init_particles_pos(particles_pos: ti.template(), X: ti.template(),
                       res_x: int, particles_per_cell: int, dx: float):
    for i in particles_pos:
        # if particles_active[i] == 1:
        cell = int(i / particles_per_cell)
        id_x = cell % res_x
        id_y = cell // res_x
        particles_pos[i] = X[id_x, id_y] + ti.Vector([(ti.random() - 0.5) for _ in ti.static(range(2))]) * dx

@ti.kernel
def init_particles_pos_uniform(particles_pos: ti.template(), X: ti.template(),
                       res_x: int, particles_per_cell: int, dx: float, particles_per_cell_axis: int,
                       dist_between_neighbor: float):
    # particles_per_cell_axis = int(ti.sqrt(particles_per_cell))
    # dist_between_neighbor = dx / particles_per_cell_axis
    # for i in particles_pos:
    #     if particles_active[i] == 1:
    #         cell = int(i / particles_per_cell)
    #         id_x = cell % res_x
    #         id_y = cell // res_x
    #
    #         particle_id_in_cell = i % particles_per_cell
    #         particle_id_x_in_cell = particle_id_in_cell % particles_per_cell_axis
    #         particle_id_y_in_cell = particle_id_in_cell // particles_per_cell_axis
    #
    #         particles_pos[i] = X[id_x, id_y] - ti.Vector([0.5, 0.5]) * dx + \
    #                            ti.Vector([particle_id_x_in_cell + 0.5, particle_id_y_in_cell + 0.5]) * dist_between_neighbor

    # particles_per_cell_axis = int(ti.sqrt(particles_per_cell))
    # dist_between_neighbor = dx / particles_per_cell_axis
    particles_x_num = particles_per_cell_axis * res_x

    for i in particles_pos:
        # if particles_active[i] == 1:
            id_x = i % particles_x_num
            id_y = i // particles_x_num
            particles_pos[i] = (ti.Vector([id_x, id_y]) + 0.5) * dist_between_neighbor

@ti.kernel
def init_particles_pos_in_a_rectangle(particles_pos: ti.template(), particles_active: ti.template(), X: ti.template(),
                       initial_fluid_res_x: int, particles_per_cell: int, dx: float, fluid_origin_grid: ti.types.ndarray()):
    for i in particles_pos:
        if particles_active[i] == 1:
            cell = int(i / particles_per_cell)
            id_x = cell % initial_fluid_res_x + fluid_origin_grid[0]
            id_y = cell // initial_fluid_res_x + fluid_origin_grid[1]
            particles_pos[i] = X[id_x, id_y] + ti.Vector([(ti.random() - 0.5) for _ in ti.static(range(2))]) * dx

@ti.kernel
def backup_particles_pos(particles_pos: ti.template(), particles_pos_backup: ti.template()):
    for i in particles_pos:
        particles_pos_backup[i] = particles_pos[i]

@ti.kernel
def init_particles_imp(particles_imp: ti.template(), particles_init_imp: ti.template(), particles_pos: ti.template(),
                       u_x: ti.template(), u_y: ti.template(),
                       C_x: ti.template(), C_y: ti.template(), dx: float):
    for i in particles_imp:
        particles_imp[i], _, new_C_x, new_C_y = interp_u_MAC_grad_imp(u_x, u_y, particles_pos[i], dx)
        # C_x[i] = new_C_x
        # C_y[i] = new_C_y
        particles_init_imp[i] = particles_imp[i]

@ti.kernel
def init_particles_imp_th(particles_imp: ti.template(), particles_init_imp: ti.template(), particles_pos: ti.template(),
                       u_x: ti.template(), u_y: ti.template(),
                       C_x: ti.template(), C_y: ti.template(), dx: float):
    for i in particles_imp:
        particles_imp[i] = vortex_vel_func_point(particles_pos[i])
        # C_x[i] = new_C_x
        # C_y[i] = new_C_y
        particles_init_imp[i] = particles_imp[i]

@ti.kernel
def init_particles_imp_grad_m(particles_imp: ti.template(), particles_pos: ti.template(),
                                u_x: ti.template(), u_y: ti.template(),
                                C_x: ti.template(), C_y: ti.template(), dx: float):
    for i in particles_imp:
        particles_imp[i], _, new_C_x, new_C_y = interp_u_MAC_grad_imp(u_x, u_y, particles_pos[i], dx)
        C_x[i] = new_C_x
        C_y[i] = new_C_y

@ti.kernel
def init_particles_imp_grad_m_th(particles_imp: ti.template(), particles_pos: ti.template(),
                                u_x: ti.template(), u_y: ti.template(),
                                C_x: ti.template(), C_y: ti.template(), dx: float):
    for i in particles_imp:
        particles_imp[i] = vortex_vel_func_point(particles_pos[i])
        C_x[i] = ti.Vector([dvxdx(particles_pos[i]), dvxdy(particles_pos[i])])
        C_y[i] = ti.Vector([dvydx(particles_pos[i]), dvydy(particles_pos[i])])

@ti.kernel
def init_particles_imp_one_step(particles_imp: ti.template(), particles_pos: ti.template(),
                                u_x: ti.template(), u_y: ti.template(), imp_x: ti.template(), imp_y: ti.template(),
                                C_x: ti.template(), C_y: ti.template(), dx: float):
    for i in particles_imp:
        particles_imp[i], _, new_C_x, new_C_y = interp_u_MAC_grad_imp(u_x, u_y, imp_x, imp_y, particles_pos[i], dx)
        C_x[i] = new_C_x
        C_y[i] = new_C_y