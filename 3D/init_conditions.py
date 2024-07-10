from taichi_utils import *
import math

# 3D specific
# w: vortex strength
# rad: radius of torus
# delta: thickness of torus
# c: central position
# unit_x, unit_y: the direction
@ti.kernel
def add_vortex_ring(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), num_samples: int):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point controls part of the curve
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)

# 3D specific
# w: vortex strength
# rad: radius of torus
# delta: thickness of torus
# c: central position
# unit_x, unit_y: the direction
@ti.kernel
def add_vortex_ring_and_smoke(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: ti.types.vector(3, float), num_samples: int):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point controls part of the curve
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
            smokef[i,j,k][3] += curve_length * (ti.exp(-(r/delta) ** 3))
    for i, j, k in smokef:
        if smokef[i,j,k][3] > 0.002:
            # if color[0] == 0. and color[1] == 0. and color[2] == 0.:
            #     smokef[i,j,k][3] = 1.0
            # else:
            #     smokef[i,j,k][3] = 0.0
            #     #smokef[i,j,k][3] = 4 * color[0] + 3 * color[1] + 2 * color[2]
            # smokef[i,j,k].xyz = color
            smokef[i, j, k][3] = 1.0
            smokef[i, j, k].xyz = color
        else:
            smokef[i,j,k] = ti.Vector([0.,0.,0.,0.])


@ti.kernel
def add_vortex_tube_and_smoke(w: float, delta: float, c: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: ti.types.vector(3, float), num_samples: int):
    curve_length = 1./num_samples
    for i, j, k in vf:
        for l in range(num_samples):
            sign = -1* w / ti.abs(w)
            p_sampled = c + (l+0.5) * ti.Vector([0., 1./num_samples, 0.])
            p_sampled.x += sign * 0.01 * ti.cos(2.*math.pi*p_sampled.y)
            p_next = c + (l+1+0.5) * ti.Vector([0., 1./num_samples, 0.])
            p_next.x += sign * 0.01 * ti.cos(2*math.pi*p_next.y)
            w_vector = (p_next-p_sampled).normalized()

            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * w_vector
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
            smokef[i,j,k][3] += curve_length * (ti.exp(-(r/delta) ** 3))
    for i, j, k in smokef:
        if smokef[i,j,k][3] > 0.002:
            smokef[i,j,k][3] = 1.0
            smokef[i,j,k].xyz = color

def init_vorts_leapfrog(X, u, smoke1, smoke2):
    # # same plane
    # add_vortex_ring(w = 2.e-2, rad = 0.24, delta = 0.016, c = ti.Vector([0.20,0.5,0.5]),
    #         unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
    #         pf = X, vf = u, num_samples = 500)

    # add_vortex_ring(w = 2.e-2, rad = 0.40, delta = 0.016, c = ti.Vector([0.20,0.5,0.5]),
    #         unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
    #         pf = X, vf = u, num_samples = 500)

    # # front and back
    # add_vortex_ring(w = 2.e-2, rad = 0.20, delta = 0.016, c = ti.Vector([0.23,0.5,0.5]),
    #         unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
    #         pf = X, vf = u, num_samples = 500)

    # add_vortex_ring(w = 2.e-2, rad = 0.20, delta = 0.016, c = ti.Vector([0.35,0.5,0.5]),
    #         unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
    #         pf = X, vf = u, num_samples = 500)

    # front and back revised
    radius = 0.21
    x_gap = 0.625 * radius
    x_start = 0.16
    delta = 0.08 * radius
    w = radius * 0.1
    add_vortex_ring_and_smoke(w = w, rad = radius, delta = delta, c = ti.Vector([x_start,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke1, color = ti.Vector([1., 0, 0]), num_samples = 2000)

    add_vortex_ring_and_smoke(w = w, rad = radius, delta = delta, c = ti.Vector([x_start+x_gap,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke2, color = ti.Vector([0, 0, 1.]), num_samples = 2000)

    add_fields(smoke1, smoke2, smoke1, 1.0)

    #leapfrog_smoke_func(smoke1, X)


def init_vorts_eight(X, u, smoke1, smoke2):
    smoke1.fill(0.)
    smoke2.fill(0.)
    size = 0.08
    cos45 = ti.cos(math.pi / 4)
    tan_theta = ti.sqrt(2) / 2
    cot_theta = ti.sqrt(2)
    x_offset = 0.16
    y_offset = 0.16
    z_offset = 0.16 / cos45 * tan_theta

    add_vortex_ring_and_smoke(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 - x_offset, 0.5 - y_offset, 0.5 - z_offset]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([-cos45, -cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke1, color=ti.Vector([1, 0., 0.]), num_samples=500)

    add_vortex_ring_and_smoke(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 + x_offset, 0.5 - y_offset, 0.5 - z_offset]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([cos45, -cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0, 1, 0.]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 - x_offset, 0.5 + y_offset, 0.5 - z_offset]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([-cos45, cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0., 0., 1]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 + x_offset, 0.5 + y_offset, 0.5 - z_offset]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([cos45, cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0., 0., 0.]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 - x_offset, 0.5 - y_offset, 0.5 + z_offset]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([cos45, cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0., 1, 1]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 + x_offset, 0.5 - y_offset, 0.5 + z_offset]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([-cos45, cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([1, 0., 1]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 - x_offset, 0.5 + y_offset, 0.5 + z_offset]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([cos45, -cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([1, 1, 0.]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 + x_offset, 0.5 + y_offset, 0.5 + z_offset]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([-cos45, -cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([1, 1, 1]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)

@ti.kernel
def add_vertical_vel(u: ti.template()):
    for I in ti.grouped(u):
        u[I].x = 0.
        u[I].y = 0.
        u[I].z = 0.3

def init_vorts_tube(X, u, smoke1, smoke2):
    smoke1.fill(0.)
    smoke2.fill(0.)
    
    add_vortex_tube_and_smoke(w = 2.e-2, delta = 0.02, c = ti.Vector([0.45,0., 0.5]),
        pf = X, vf = u, smokef = smoke1, color = ti.Vector([1, 0, 0]), num_samples = 500)

    add_vortex_tube_and_smoke(w = -2.e-2, delta = 0.02, c = ti.Vector([0.55,0., 0.5]),
        pf = X, vf = u, smokef = smoke2, color = ti.Vector([0, 1, 0]), num_samples = 500)
    
    add_fields(smoke1, smoke2, smoke1, 1.0)


# some shapes (checkerboards...)

@ti.kernel
def stripe_func(qf: ti.template(), pf: ti.template(), x_start: float, x_end: float):
    for I in ti.grouped(qf):
        if x_start <= pf[I].x <= x_end and 0.15 <= pf[I].y <= 0.85 and 0.15 <= pf[I].z <= 0.85:
            qf[I] = ti.Vector([1.0, 1.0, 1.0])
        else:
            qf[I] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def double_stripe_func(qf: ti.template(), pf: ti.template(), 
                x_start1: float, x_end1: float,
                x_start2: float, x_end2: float):
    for I in ti.grouped(qf):
        if x_start1 <= pf[I].x <= x_end1 and 0.4 <= pf[I].y <= 0.6 and 0.4 <= pf[I].z <= 0.6:
            qf[I] = ti.Vector([1.0, 0., 0.])
        elif x_start2 <= pf[I].x <= x_end2 and 0.4 <= pf[I].y <= 0.6 and 0.4 <= pf[I].z <= 0.6:
            qf[I] = ti.Vector([0., 0., 1.0])
        else:
            qf[I] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def leapfrog_smoke_func(qf: ti.template(), pf: ti.template()):
    for I in ti.grouped(qf):
        if 0.2 <= pf[I].x <= 0.3 and 0.1 <= pf[I].y <= 0.9 and 0.1 <= pf[I].z <= 0.9:
            center = ti.Vector([0.2, 0.5, 0.5])
            radius = 0.3
            dist = (pf[I]-center).norm()
            if dist < radius:
                qf[I] = ti.Vector([1.0, 0.0, 0.0, 1.0])
            else:
                qf[I] = ti.Vector([0.0, 1.0, 0.0, 2.0])
        else:
            qf[I] = ti.Vector([0.0, 0.0, 0.0, 0.0])

@ti.kernel
def init_particles_pos_uniform(particles_pos: ti.template(), X: ti.template(),
                       res_x: int, res_y: int, particles_per_cell: int, dx: float, particles_per_cell_axis: int,
                       dist_between_neighbor: float):

    particles_x_num = particles_per_cell_axis * res_x
    particles_y_num = particles_per_cell_axis * res_y

    for i in particles_pos:
        # if particles_active[i] == 1:
            id_x = i % particles_x_num
            id_yz = i // particles_x_num
            id_y = id_yz % particles_y_num
            id_z = id_yz // particles_y_num
            particles_pos[i] = (ti.Vector([id_x, id_y, id_z]) + 0.5) * dist_between_neighbor

@ti.kernel
def init_particles_imp(particles_init_imp: ti.template(), particles_pos: ti.template(),
                       u_x: ti.template(), u_y: ti.template(), u_z:ti.template(), C_x: ti.template(),
                       C_y: ti.template(), C_z: ti.template(), dx: float):
    for i in particles_init_imp:
        particles_init_imp[i], _, new_C_x, new_C_y, new_C_z = interp_u_MAC_grad_imp(u_x, u_y, u_z, particles_pos[i], dx)
        # C_x[i] = new_C_x
        # C_y[i] = new_C_y
        # C_z[i] = new_C_z
        # particles_init_imp[i] = particles_imp[i]

@ti.kernel
def init_particles_imp_grad_m(particles_imp: ti.template(), particles_pos: ti.template(), u_x: ti.template(),
                              u_y: ti.template(), u_z: ti.template(), C_x: ti.template(), C_y: ti.template(),
                              C_z: ti.template(), dx: float):
    for i in particles_imp:
        particles_imp[i], _, new_C_x, new_C_y, new_C_z = interp_u_MAC_grad_imp(u_x, u_y, u_z, particles_pos[i], dx)
        C_x[i] = new_C_x
        C_y[i] = new_C_y
        C_z[i] = new_C_z

@ti.kernel
def init_particles_smoke(particles_smoke: ti.template(), particles_grad_smoke: ti.template(),
                         particles_pos: ti.template(), smoke: ti.template(), dx: float):
    for i in particles_smoke:
        particles_smoke[i], particles_grad_smoke[i] = interp_u_MAC_smoke(smoke, particles_pos[i], dx)