# some hyperparameters
dim = 2
testing = False
use_neural = False
save_ckpt = True
save_frame_each_step = False
use_BFECC = False
use_midpoint_vel = True
use_APIC = True
forward_update_T = True
plot_particles = False
use_reseed_particles = False
reinit_particle_pos = True
dpi_vor = 512 if plot_particles else 512 // 8

# encoder hyperparameters
min_res = (128, 32)
num_levels = 4
feat_dim = 2
activate_threshold = 0.03
# neural buffer hyperparameters
N_iters = 2000
N_batch = 40000 #25000
success_threshold = 3.e-8
# simulation hyperparameters
res_x = 1024
res_y = 256
visualize_dt = 0.1
reinit_every = 20
reinit_every_grad_m = 8
ckpt_every = 1
CFL = 1.0
from_frame = 0
total_frames = 5000
use_total_steps = False
total_steps = 1
exp_name = "2D_leapfrog_1024x256_reinit-20-8"

particles_per_cell = 16
total_particles_num_ratio = 1
cell_max_particle_num_ratio = 2.0
# min_particles_per_cell_ratio = 0.75
# min_particles_per_cell = int(particles_per_cell * min_particles_per_cell_ratio)
min_particles_per_cell_ratio = 1
min_particles_per_cell = int(particles_per_cell * min_particles_per_cell_ratio)
max_particles_per_cell_ratio = 1.6
max_particles_per_cell = int(particles_per_cell * max_particles_per_cell_ratio)

max_delete_particle_try_num = 100000
