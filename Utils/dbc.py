import math

import skimage
from skimage.color import rgb2gray
import numpy as np

import torch


import math
import numpy as np
from skimage.color import rgb2gray
import skimage

def differential_box_counting(image, block, scale, box_size, slide_style=0, lac_type='grayscale'):
    """
    Differential box-counting algorithm for computing lacunarity on a normal NumPy RGB image.

    Parameters:
    -----------
    image: ndarray
        The input RGB image as a NumPy array.
    block: int
        The size of the block in pixels.
    scale: int
        The window size in pixels for computing lacunarity (w x w).
    box_size: int
        The size of the cube (r x r x r).
    slide_style: int
        How the boxes slide across the window.
        For glide: specify a slide_style of 0
        For block: specify a slide_style of -1
        For skip: specify the number of pixels to skip (i.e., a positive integer).
    lac_type: str
        Two options are available: grayscale or binary.
        Lacunarity calculations are slightly different for these.

    Returns:
    --------
    out: ndarray
        The lacunarity image.
    """
    assert box_size < scale
    assert scale % box_size == 0

    # Restrict bands
    image = image[:, :, 0:3]

    if lac_type == "grayscale":
        image = skimage.img_as_ubyte(rgb2gray(image))

    # Move the window (scale) over the 2D image block by block
    out_image = []
    for i in range(0, image.shape[0], block):
        outrow = []
        if i >= scale and i <= image.shape[0] - scale:
            for j in range(0, image.shape[1], block):
                if j >= scale and j <= image.shape[1] - scale:
                    block_arr = image[i:i+block, j:j+block]
                    center_i = int(i + block / 2)
                    center_j = int(j + block / 2)

                    if block % 2 != 0 and scale % 2 == 0:
                        scale_arr = image[center_i-int(scale/2):center_i+int(scale/2),
                                          center_j-int(scale/2):center_j+int(scale/2)]
                    else:
                        scale_arr = image[center_i-int(scale/2):center_i+int(scale/2)+1,
                                          center_j-int(scale/2):center_j+int(scale/2)+1]

                    # Now slide the box over the window
                    n_mr = {}  # Dictionary to count the number of boxes of size r and mass m (a histogram)
                    total_boxes_in_window = 0
                    ii = 0
                    while ii + box_size <= len(scale_arr):
                        jj = 0
                        while jj + box_size <= len(scale_arr[0]):
                            total_boxes_in_window += 1
                            box = scale_arr[ii:ii+box_size, jj:jj+box_size]
                            max_val = np.amax(box)
                            min_val = np.amin(box)
                            u = math.ceil(min_val / box_size)  # Box with minimum pixel value
                            v = math.ceil(max_val / box_size)  # Box with maximum pixel value
                            n_ij = v - u + 1  # Relative height of column at ii and jj

                            if n_ij not in n_mr:
                                n_mr[n_ij] = 1
                            else:
                                n_mr[n_ij] += 1

                            # Move the box based on the glide_style
                            if slide_style == 0:  # Glide
                                jj += 1
                            elif slide_style == -1:  # Block
                                jj += box_size
                            else:  # Skip
                                jj += box_size + slide_style
                        if slide_style == 0:  # Glide
                            ii += 1
                        elif slide_style == -1:  # Block
                            ii += box_size
                        else:  # Skip
                            ii += box_size + slide_style

                    num = 0
                    denom = 0.000001
                    for masses in n_mr:
                        q_mr = n_mr[masses] / total_boxes_in_window
                        num += (masses * masses) * q_mr
                        denom += masses * q_mr
                    denom = denom**2
                    lac = num / denom
                    outrow.append(lac)
            out_image.append(outrow)

    return np.array(out_image)

# Example usage:
# lacunarity_result = differential_box_counting(rgb_image, block=16, scale=64, box_size=8)
