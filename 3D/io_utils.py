import numpy as np
import torch
import imageio.v2 as imageio
import os
import shutil
import glob
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import vtk

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

def comp_vort_3d(vel_img):
    W, H, D, _ = vel_img.shape
    dx = 1./H
    u = vel_img[...,0]
    v = vel_img[...,1]
    w = vel_img[...,2]
    du_dz = u[1:-1,1:-1,2:] - u[1:-1,1:-1,:-2]
    du_dy = u[1:-1,2:,1:-1] - u[1:-1,:-2,1:-1]

    dv_dx = v[2:,1:-1,1:-1] - v[:-2,1:-1,1:-1]
    dv_dz = v[1:-1,1:-1,2:] - v[1:-1,1:-1,:-2]

    dw_dx = w[2:,1:-1,1:-1] - w[:-2,1:-1,1:-1]
    dw_dy = w[1:-1,2:,1:-1] - w[1:-1,:-2,1:-1]

    vort_img = torch.cat([(dw_dy - dv_dz)[...,None], (du_dz - dw_dx)[...,None], (dv_dx - du_dy)[...,None]], dim = -1)
    return vort_img / (2 * dx)

def write_image(img_xy, outdir, i):
    img = np.flip(img_xy.transpose([1,0,2]), 0)
    # take the predicted c map
    img8b = to8b(img)
    save_filepath = os.path.join(outdir, '{:03d}.jpg'.format(i))
    imageio.imwrite(save_filepath, img8b)

def write_field(img, outdir, i, vmin = 0, vmax = 1):
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
    plt.text(0.87 * scale_x, 0.87 * scale_y, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:03d}.jpg'.format(i)), dpi = 512//8)

def write_vtk(w_numpy, outdir, i):
    data = w_numpy.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("Value")
    imageData.GetPointData().SetScalars(vtkDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_w_and_smoke(w_numpy, smoke_numpy, outdir, i):
    data = w_numpy.squeeze()
    smoke_data = smoke_numpy.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("vorticity")
    imageData.GetPointData().SetScalars(vtkDataArray)

    # add smoke
    smokeDataArray = numpy_support.numpy_to_vtk(smoke_data.ravel(order = "F"), deep=True)
    smokeDataArray.SetName("smoke")
    imageData.GetPointData().AddArray(smokeDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()