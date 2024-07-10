import numpy as np
import imageio.v2 as imageio
import os
import shutil
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# remove everything in dir
def remove_everything_in(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# for writing images
def to_numpy(x):
    return x.detach().cpu().numpy()
    
def to8b(x):
    return (255*np.clip(x,0,1)).astype(np.uint8)

def comp_vort(vel_img): # compute the curl of velocity
    W, H, _ = vel_img.shape
    dx = 1./H
    u = vel_img[...,0]
    v = vel_img[...,1]
    dvdx = 1/(2*dx) * (v[2:, 1:-1] - v[:-2, 1:-1])
    dudy = 1/(2*dx) * (u[1:-1, 2:] - u[1:-1, :-2])
    vort_img = dvdx - dudy
    return vort_img

def write_image(img_xy, outdir, i):
    img = np.flip(img_xy.transpose([1,0,2]), 0)
    # take the predicted c map
    img8b = to8b(img)
    save_filepath = os.path.join(outdir, '{:04d}.png'.format(i))
    imageio.imwrite(save_filepath, img8b)

def write_field(img, outdir, i, particles_pos, cell_type=None, vmin=0, vmax=1, plot_particles=False, dpi=512//8):
    array = img[:,:,np.newaxis]
    scale_x = array.shape[0]
    scale_y = array.shape[1]
    array = np.transpose(array, (1, 0, 2)) # from X,Y to Y,X
    x_to_y = array.shape[1]/array.shape[0]
    y_size = 7
    fig = plt.figure(num=1, figsize=(x_to_y * y_size + 1, y_size), clear=True)
    ax = fig.add_subplot()
    ax.set_xlim([0, array.shape[1]])
    ax.set_ylim([0, array.shape[0]])
    cmap = 'jet'
    p = ax.imshow(array, alpha = 0.4, cmap=cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    # if plot_particles:
    #     active_particles_pos = particles_pos
    #     ax.scatter(active_particles_pos[:, 0], active_particles_pos[:, 1], facecolors='black', s=0.0001)
    # plt.text(0.87 * scale_x, 0.87 * scale_y, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:04d}.png'.format(i)), dpi = dpi)
    plt.close()

def write_particles(img, outdir, i, particles_pos, vmin = 0, vmax = 1):
    crop_x = 16
    range_x = [0.5, 1.5]
    crop_y = 5
    range_y = [2.75, 4]
    array = img[:, :, np.newaxis]
    scale_x = array.shape[0]
    scale_y = array.shape[1]
    array = np.transpose(array, (1, 0, 2)) # from X,Y to Y,X
    x_to_y = array.shape[1]/array.shape[0]
    y_size = 7
    fig = plt.figure(num=1, figsize=((x_to_y * y_size + 1) / crop_x * (range_x[1] - range_x[0]),
                                     y_size / crop_y * (range_y[1] - range_y[0])), clear=True)
    fig.subplots_adjust(left=-0.5, right=1, top=1, bottom=0)
    ax = fig.add_subplot()
    # ax.set_xlim([array.shape[1] / crop_x * range_x[0], array.shape[1] / crop_x * range_x[1]])
    # ax.set_ylim([array.shape[0] / crop_y * range_y[0], array.shape[0] / crop_y * range_y[1]])
    ax.set_xlim([0, array.shape[1] / crop_x * (range_x[1] - range_x[0])])
    ax.set_ylim([0, array.shape[0] / crop_y * (range_y[1] - range_y[0])])
    cmap = 'jet'
    p = ax.imshow(array[int(array.shape[0] / crop_y * range_y[0]) : int(array.shape[0] / crop_y * range_y[1]),
                  int(array.shape[1] / crop_x * range_x[0]) : int(array.shape[1] / crop_x * range_x[1]),
                  :], alpha = 0.4, cmap=cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    particles_pos = particles_pos[(particles_pos[:, 1] > array.shape[0] / crop_y * range_y[0]) &
                                  (particles_pos[:, 1] < array.shape[0] / crop_y * range_y[1]) &
                                  (particles_pos[:, 0] > array.shape[1] / crop_x * range_x[0]) &
                                  (particles_pos[:, 0] < array.shape[1] / crop_x * range_x[1])]

    # contour_width = 2
    # x_array = np.array(particles_pos[:, 0])
    # y_array = np.array(particles_pos[:, 1])
    # centers = np.column_stack((x_array, y_array))
    # circles = patches.Circle(centers, contour_width, edgecolor='black', facecolor='none', lw=contour_width)
    # ax.add_collection(patches.PathCollection(circles))

    # ax.scatter(particles_pos[:, 0],
    #            particles_pos[:, 1],
    #            marker='o', facecolors='black', edgecolors='black', s=0.0001, linewidths=1)

    ax.scatter(particles_pos[:, 0] - array.shape[1] / crop_x * range_x[0],
               particles_pos[:, 1] - array.shape[0] / crop_y * range_y[0],
               marker='o', facecolors='black', edgecolors='black', s=0.0001, linewidths=1)

    plt.text(0.87 * scale_x / crop_x, 0.87 * scale_y / crop_y, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:04d}.png'.format(i)), dpi = 512)
    plt.close()
