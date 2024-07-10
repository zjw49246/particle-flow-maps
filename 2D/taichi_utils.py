import taichi as ti
import torch

eps = 1.e-6
data_type = ti.f32
torch_type = torch.float32

@ti.kernel
def scale_field(a: ti.template(), alpha: float, result: ti.template()):
    for i, j in result:
        result[i, j] = alpha * a[i,j]

@ti.kernel
def add_fields(f1: ti.template(), f2: ti.template(), dest: ti.template(), multiplier: float):
    for I in ti.grouped(dest):
        dest[I] = f1[I] + multiplier * f2[I]

@ti.kernel
def copy_to(source: ti.template(), dest: ti.template()):
    for I in ti.grouped(source):
        dest[I] = source[I]

@ti.func
def sample(qf: ti.template(), u: float, v: float):
    u_dim, v_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim-1))
    j = ti.max(0, ti.min(int(v), v_dim-1))
    return qf[i, j]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.kernel
def center_coords_func(pf: ti.template(), dx: float):
    for i, j in pf:
        pf[i, j] = ti.Vector([i + 0.5, j + 0.5]) * dx

@ti.kernel
def horizontal_coords_func(pf: ti.template(), dx: float):
    for i, j in pf:
        pf[i, j] = ti.Vector([i, j + 0.5]) * dx

@ti.kernel
def vertical_coords_func(pf: ti.template(), dx: float):
    for i, j in pf:
        pf[i, j] = ti.Vector([i + 0.5, j]) * dx

@ti.kernel
def get_central_vector(horizontal: ti.template(), vertical: ti.template(), central: ti.template()):
    for i, j in central:
        central[i,j].x = 0.5 * (horizontal[i+1, j] + horizontal[i, j])
        central[i,j].y = 0.5 * (vertical[i, j+1] + vertical[i, j])

@ti.kernel
def split_central_vector(central: ti.template(), horizontal: ti.template(), vertical: ti.template()):
    for i, j in horizontal:
        r = sample(central, i, j)
        l = sample(central, i-1, j)
        horizontal[i,j] = 0.5 * (r.x + l.x)
    for i, j in vertical:
        t = sample(central, i, j)
        b = sample(central, i, j-1)
        vertical[i,j] = 0.5 * (t.y + b.y)

@ti.kernel
def sizing_function(u: ti.template(), sizing: ti.template(), dx: float):
    u_dim, v_dim = u.shape
    for i, j in u:
        u_l = sample(u, i-1, j)
        u_r = sample(u, i+1, j)
        u_t = sample(u, i, j+1)
        u_b = sample(u, i, j-1)
        partial_x = 1./(2*dx) * (u_r - u_l)
        partial_y = 1./(2*dx) * (u_t - u_b)
        if i == 0:
            partial_x = 1./(2*dx) * (u_r - 0)
        elif i == u_dim - 1:
            partial_x = 1./(2*dx) * (0 - u_l)
        if j == 0:
            partial_y = 1./(2*dx) * (u_t - 0)
        elif j == v_dim - 1:
            partial_y = 1./(2*dx) * (0 - u_b)

        sizing[i, j] = ti.sqrt(partial_x.x ** 2 + partial_x.y ** 2 + partial_y.x ** 2 + partial_y.y ** 2)

@ti.kernel
def diffuse_grid(value: ti.template(), tmp: ti.template()):
    for i, j in tmp:
        tmp[i,j] = 0.25 * (sample(value, i+1, j) + sample(value, i-1, j)\
                + sample(value, i, j+1) + sample(value, i, j-1))
    for i, j in tmp:
        value[i,j] = ti.max(value[i,j], tmp[i,j])

@ti.kernel
def curl(vf: ti.template(), cf: ti.template(), dx: float):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        cf[i, j] = (vr.y - vl.y - vt.x + vb.x) / (2*dx)

# # # # interpolation kernels # # # # 

@ti.func
def N_1(x):
    return 1.0-ti.abs(x)
    
@ti.func
def dN_1(x):
    result = -1.0
    if x < 0.:
        result = 1.0
    return result

@ti.func
def interp_1(vf, p, dx, BL_x = 0.5, BL_y = 0.5):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(0., ti.min(u, u_dim-1-eps))
    t = ti.max(0., ti.min(v, v_dim-1-eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)

    interped = a * N_1(fu) * N_1(fv) + \
            b * N_1(fu-1) * N_1(fv) + \
            c * N_1(fu) * N_1(fv-1) + \
            d * N_1(fu-1) * N_1(fv-1)
    
    return interped

@ti.func
def interp_grad_1(vf, p, dx, BL_x = 0.5, BL_y = 0.5):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(0., ti.min(u, u_dim-1-eps))
    t = ti.max(0., ti.min(v, v_dim-1-eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)

    # comp grad while bilerp
    partial_x = 1./dx * (a * dN_1(fu) * N_1(fv) + \
                    b * dN_1(fu-1) * N_1(fv) + \
                    c * dN_1(fu) * N_1(fv-1) + \
                    d * dN_1(fu-1) * N_1(fv-1))
    partial_y = 1./dx * (a * N_1(fu) * dN_1(fv) + \
                        b * N_1(fu-1) * dN_1(fv) + \
                        c * N_1(fu) * dN_1(fv-1) + \
                        d * N_1(fu-1) * dN_1(fv-1))

    interped = a * N_1(fu) * N_1(fv) + \
            b * N_1(fu-1) * N_1(fv) + \
            c * N_1(fu) * N_1(fv-1) + \
            d * N_1(fu-1) * N_1(fv-1)
    
    return interped, ti.Vector([partial_x, partial_y])


@ti.func
def N_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = 3.0/4.0 - abs_x ** 2
    elif abs_x < 1.5:
        result = 0.5 * (3.0/2.0-abs_x) ** 2
    return result
    
@ti.func
def dN_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = -2 * abs_x
    elif abs_x < 1.5:
        result = 0.5 * (2 * abs_x - 3)
    if x < 0.0: # if x < 0 then abs_x is -1 * x
        result *= -1
    return result

@ti.func
def d2N_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = -2
    elif abs_x < 1.5:
        result = 1
    return result


@ti.func
def interp_grad_2(vf, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    partial_x = 0.
    partial_y = 0.
    interped = 0.

    new_C = ti.Matrix.zero(float, 2, 2)

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            partial_x += 1. / dx * (value * dN_2(x_p_x_i) * N_2(y_p_y_i))
            partial_y += 1. / dx * (value * N_2(x_p_x_i) * dN_2(y_p_y_i))
            interped += value * N_2(x_p_x_i) * N_2(y_p_y_i)
            dpos = ti.Vector([-x_p_x_i, -y_p_y_i])
            vector_value = ti.Vector([value, 0.0])
            # if is_y:
            #     vector_value = ti.Vector([0.0, value])
            new_C += 4 * N_2(x_p_x_i) * N_2(y_p_y_i) * vector_value.outer_product(dpos) / dx

    return interped, ti.Vector([partial_x, partial_y])


@ti.func
def interp_u_MAC_grad_backup(u_x, u_y, p, dx):
    u_x_p, grad_u_x_p, C_x = interp_grad_2(u_x, p, dx, BL_x=0.0, BL_y=0.5)
    u_y_p, grad_u_y_p, C_y = interp_grad_2(u_y, p, dx, BL_x=0.5, BL_y=0.0)
    return ti.Vector([u_x_p, u_y_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p]), C_x, C_y

@ti.func
def interp_u_MAC_grad(u_x, u_y, p, dx):
    u_x_p, grad_u_x_p = interp_grad_2(u_x, p, dx, BL_x=0.0, BL_y=0.5)
    u_y_p, grad_u_y_p = interp_grad_2(u_y, p, dx, BL_x=0.5, BL_y=0.0)
    return ti.Vector([u_x_p, u_y_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p])

@ti.func
def interp_grad_2_imp(vf, p, dx, BL_x = 0.5, BL_y = 0.5, is_y=False):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    partial_x = 0.
    partial_y = 0.
    interped = 0.

    new_C = ti.Vector([0.0, 0.0])
    # interped_imp = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            # imp_value = sample(imp, iu + i, iv + j)
            dw_x = 1./dx * dN_2(x_p_x_i) * N_2(y_p_y_i)
            dw_y = 1./dx * N_2(x_p_x_i) * dN_2(y_p_y_i)
            partial_x += value * dw_x
            partial_y += value * dw_y
            interped += value * N_2(x_p_x_i) * N_2(y_p_y_i)
            # dpos = ti.Vector([-x_p_x_i, -y_p_y_i])
            # vector_value = ti.Vector([imp_value, 0.0])
            # # vector_value = ti.Vector([value, 0.0])
            # if is_y:
            #     vector_value = ti.Vector([0.0, imp_value])
            #     # vector_value = ti.Vector([0.0, value])
            new_C += ti.Vector([dw_x, dw_y]) * value
            # interped_imp += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)
    
    return interped, ti.Vector([partial_x, partial_y]), new_C

@ti.func
def interp_u_MAC_grad_imp(u_x, u_y, p, dx):
    u_x_p, grad_u_x_p, C_x = interp_grad_2_imp(u_x, p, dx, BL_x = 0.0, BL_y = 0.5, is_y=False)
    u_y_p, grad_u_y_p, C_y = interp_grad_2_imp(u_y, p, dx, BL_x = 0.5, BL_y = 0.0, is_y=True)
    return ti.Vector([u_x_p, u_y_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p]), C_x, C_y


@ti.func
def interp_grad_2_imp_and_grad_imp(vf, imp, p, dx, BL_x=0.5, BL_y=0.5, is_y=False):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    partial_x = 0.
    partial_y = 0.
    interped = 0.

    new_C = ti.Vector([0.0, 0.0])
    interped_imp = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            imp_value = sample(imp, iu + i, iv + j)
            dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i)
            dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i)
            partial_x += value * dw_x
            partial_y += value * dw_y
            interped += value * N_2(x_p_x_i) * N_2(y_p_y_i)
            # dpos = ti.Vector([-x_p_x_i, -y_p_y_i])
            # vector_value = ti.Vector([imp_value, 0.0])
            # # vector_value = ti.Vector([value, 0.0])
            # if is_y:
            #     vector_value = ti.Vector([0.0, imp_value])
            #     # vector_value = ti.Vector([0.0, value])
            new_C += ti.Vector([dw_x, dw_y]) * imp_value
            interped_imp += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped, ti.Vector([partial_x, partial_y]), new_C, interped_imp


@ti.func
def interp_u_MAC_imp_and_grad_imp(u_x, u_y, imp_x, imp_y, p, dx):
    u_x_p, grad_u_x_p, C_x, interped_imp_x = interp_grad_2_imp_and_grad_imp(u_x, imp_x, p, dx, BL_x=0.0, BL_y=0.5, is_y=False)
    u_y_p, grad_u_y_p, C_y, interped_imp_y = interp_grad_2_imp_and_grad_imp(u_y, imp_y, p, dx, BL_x=0.5, BL_y=0.0, is_y=True)
    return ti.Vector([u_x_p, u_y_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p]), C_x, C_y, interped_imp_x, interped_imp_y


@ti.func
def interp(field, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = field.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    interped = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            imp_value = sample(field, iu + i, iv + j)
            interped += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped


@ti.func
def interp_grad_2_updated_imp(imp, phi_grid, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = imp.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    new_C = ti.Vector([0.0, 0.0])

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            updated_imp_pos = sample(phi_grid, iu + i, iv + j)
            imp_value = interp(imp, updated_imp_pos, dx, BL_x, BL_y)
            dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i)
            dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i)
            new_C += ti.Vector([dw_x, dw_y]) * imp_value

    return new_C


@ti.func
def interp_u_MAC_grad_updated_imp(u_x, u_y, imp_x, imp_y, phi_x_grid, phi_y_grid, p, dx):
    C_x = interp_grad_2_updated_imp(imp_x, phi_x_grid, p, dx, BL_x=0.0, BL_y=0.5)
    C_y = interp_grad_2_updated_imp(imp_y, phi_y_grid, p, dx, BL_x=0.5, BL_y=0.0)
    return C_x, C_y

@ti.func
def interp_grad_T(T, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = T.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    grad_T = ti.Matrix.zero(float, 2, 2)

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            T_value = sample(T, iu + i, iv + j)
            dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i)
            dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i)
            dw = ti.Vector([dw_x, dw_y])
            grad_T += dw.outer_product(T_value)

    return grad_T

@ti.func
def interp_MAC_grad_T(T_x, T_y, p, dx):
    grad_T_x = interp_grad_T(T_x, p, dx, BL_x=0.0, BL_y=0.5)
    grad_T_y = interp_grad_T(T_y, p, dx, BL_x=0.5, BL_y=0.0)
    return grad_T_x, grad_T_y

# # # # ti and torch conversion # # # # 

@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]

@ti.kernel
def torch2ti(data: ti.types.ndarray(), field: ti.template()):
    for I in ti.grouped(data):
        field[I] = data[I]

@ti.kernel
def torch2ti_grad(grad: ti.types.ndarray(), field: ti.template()):
    for I in ti.grouped(grad):
        field.grad[I] = grad[I]

@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]

@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4

    