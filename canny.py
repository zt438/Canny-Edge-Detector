'''
canny.py
- Zhen Tang

Implementation of Canny's Edge Detector
'''

from PIL import Image
import numpy as np
import math
import argparse
import sys

# defining the convolutions masks and the normalization factor
GAUSSIAN_MASK = np.array([[ 1,  1,  2,  2,  2,  1,  1],
                          [ 1,  2,  2,  4,  2,  2,  1],
                          [ 2,  2,  4,  8,  4,  2,  2],
                          [ 2,  4,  8, 16,  8,  4,  2],
                          [ 2,  2,  4,  8,  4,  2,  2],
                          [ 1,  2,  2,  4,  2,  2,  1],
                          [ 1,  1,  2,  2,  2,  1,  1]])
GAUSSIAN_MASK_NORM = 140

SOBEL_X_MASK = np.array([[-1,  0,  1],
                         [-2,  0,  2],
                         [-1,  0,  1]])
SOBEL_Y_MASK = np.array([[ 1,  2,  1],
                         [ 0,  0,  0],
                         [-1, -2, -1]])
SOBEL_MASK_NORM = 4

# the thresholds for double thresholding
T_1 = 7
T_2 = 2 * T_1

# apply_mask() - applies the mask to the image centered at the location
def apply_mask(img_arr, pixel_i, pixel_j, mask):
    start_i = pixel_i - int(mask.shape[0] / 2)
    start_j = pixel_j - int(mask.shape[1] / 2)
    
    # return np.nan if undefined due to out of bound
    if (start_i < 0 or start_j < 0
       or (pixel_i + int(mask.shape[0] / 2)) >= img_arr.shape[0]
       or (pixel_j + int(mask.shape[1] / 2)) >= img_arr.shape[1]):
        return np.nan
    
    result = 0
    
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            # return np.nan if mask lies in undefined region
            if (np.isnan(img_arr[start_i + i, start_j + j])):
                return np.nan
            
            result += (mask[i, j] * img_arr[start_i + i, start_j + j])
    
    return result

# convolution() - performs discrete convolution with the mask
def convolution(img_arr, mask):
    result = np.zeros_like(img_arr).astype(float)

    for i in range(0, img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            result[i, j] = apply_mask(img_arr, i, j, mask)

    return result

# gaussian_smoothing() - perform convolution with the gaussian mask
def gaussian_smoothing(img_arr):
    smoothed = convolution(img_arr, GAUSSIAN_MASK) / GAUSSIAN_MASK_NORM
    output_version = np.copy(smoothed)

    # replaced np.nan values with 0, cast to int and save the result
    output_version[np.isnan(output_version)] = 0
    output = Image.fromarray(np.rint(output_version).astype(np.uint8), 'L')
    output.save('gaussian_smooth.bmp')

    return smoothed

# gradient_response() - calculate vertical and horizontal gradient responses
def gradient_response(img_arr):
    horizontal = convolution(img_arr, SOBEL_X_MASK) / SOBEL_MASK_NORM
    vertical = convolution(img_arr, SOBEL_Y_MASK) / SOBEL_MASK_NORM
    
    # create a copy to cast to int for output image
    # this is done such that the exact values from the calculation are used
    # for the remaining calculations
    horizontal_copy = np.copy(horizontal)
    vertical_copy = np.copy(vertical)
    
    # replaced np.nan values with 0 and cast the rounded absolute value to uint8
    horizontal_copy[np.isnan(horizontal_copy)] = 0
    output_horizontal = Image.fromarray((np.rint(np.absolute(horizontal_copy)).astype(np.uint8)), 'L')
    output_horizontal.save('horizontal_gradient.bmp')
    
    vertical_copy[np.isnan(vertical_copy)] = 0
    output_vertical = Image.fromarray(np.rint(np.absolute(vertical_copy)).astype(np.uint8), 'L')
    output_vertical.save('vertical_gradient.bmp')
    
    return (horizontal, vertical)

# gradient_magnitude() 
# - computed as the square root of the sum of the squares of 
#   the horizontal and vertical gradients
def gradient_magnitude(horizontal, vertical):
    magnitude = np.empty_like(horizontal)
    
    for i in range(0, horizontal.shape[0]):
        for j in range(0, horizontal.shape[1]):
            # output is undefined if mask lies in undefined region
            if (np.isnan(horizontal[i, j]) or np.isnan(vertical[i, j])):
                magnitude[i, j] = np.nan
            else:
                horizontal_square = horizontal[i, j] * horizontal[i, j]
                vertical_square = vertical[i, j] * vertical[i, j]
                magnitude[i, j] = math.sqrt(horizontal_square + vertical_square)
    
    # normalizes the magnitude
    # maximum possible value of magnitude is sqrt(2*255*255)
    # normalization factor of sqrt(2)
    magnitude /= math.sqrt(2)
    
    mag_copy = np.copy(magnitude)
    # output the normalized magnitude image
    mag_copy[np.isnan(mag_copy)] = 0
    output_mag = Image.fromarray(np.rint(mag_copy).astype(np.uint8), 'L')
    output_mag.save('gradient_magnitude.bmp')
    
    return magnitude

# gradient_angle() - computed as arctan(gradient_y / gradient_x)
def gradient_angle(horizontal, vertical):
    angle = np.empty_like(horizontal)

    for i in range(0, horizontal.shape[0]):
        for j in range(0, horizontal.shape[1]):
            if (np.isnan(horizontal[i, j]) or np.isnan(vertical[i, j])):
                angle[i, j] = np.nan
            # no angle if both horizontal and vertical gradient are 0
            elif (horizontal[i, j] == 0 and vertical[i, j] == 0):
                angle[i, j] = np.nan
            # math.atan2 handles cases in which the horizontal gradient is 0
            else:
                angle[i, j] = math.atan2(vertical[i, j], horizontal[i, j]) * 180 / math.pi

    return angle

# Non-maxima Suppression
def non_maxima_suppression(magnitude, angle):
    suppressed = np.empty_like(magnitude)
    
    for i in range(0, magnitude.shape[0]):
        for j in range(0, magnitude.shape[1]):
            # location where gradient undefined has output 0
            if np.isnan(magnitude[i, j]):
                suppressed[i, j] = 0
            else:
                # offset by 22.5 to simplify sector computation
                current_angle = (angle[i, j] + 22.5) % 180
                # sector 0
                if (current_angle < 45):
                    neighbor1 = magnitude[i, j - 1]
                    neighbor2 = magnitude[i, j + 1]
                # sector 1
                elif (current_angle < 90):
                    neighbor1 = magnitude[i - 1, j + 1]
                    neighbor2 = magnitude[i + 1, j - 1]
                # sector 2
                elif (current_angle < 135):
                    neighbor1 = magnitude[i - 1, j]
                    neighbor2 = magnitude[i + 1, j]
                # sector 3
                else:
                    neighbor1 = magnitude[i - 1, j - 1]
                    neighbor2 = magnitude[i + 1, j + 1]
                # location where neighbor has undefined gradient has output 0
                if (np.isnan(neighbor1) or np.isnan(neighbor2)):
                    suppressed[i, j] = 0
                elif (magnitude[i, j] > neighbor1 and magnitude[i, j] > neighbor2):
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0
                    
    # output the post suppression image
    output = Image.fromarray(np.rint(suppressed).astype(np.uint8), 'L')
    output.save('post_suppression_magnitude.bmp')
    
    return suppressed

# double_threshold()
def double_threshold(suppressed, angle):
    result = np.zeros_like(suppressed)
    
    for i in range(0, suppressed.shape[0]):
        for j in range(0, suppressed.shape[1]):
            # set values less than T_1 to 0, values greater than T_2 to 255
            if suppressed[i, j] < T_1:
                result[i, j] = 0
            elif suppressed[i, j] > T_2:
                result[i, j] = 255
            else:
                # values between thresholds to 255 if neighbor > T_2, else 0
                neighbor_check = False
                for neighbor_i in range(i - 1, i + 2):
                    for neighbor_j in range(j - 1, j + 2):
                        # out of bound check
                        if (neighbor_i < 0 or \
                            neighbor_i >= suppressed.shape[0] or \
                            neighbor_j < 0 or \
                            neighbor_j >= suppressed.shape[1]):
                            continue
                        # checking angle difference > 315 because the 
                        # true difference is < 45 in that case
                        if (suppressed[neighbor_i, neighbor_j] > T_2 and \
                            (abs(angle[i, j] - angle[neighbor_i, neighbor_j]) <= 45.0 or \
                            abs(angle[i, j] - angle[neighbor_i, neighbor_j]) >= 315.0)):
                            neighbor_check = True
                    if (neighbor_check):
                        break
                
                result[i, j] = 255 if neighbor_check else 0
    
    output = Image.fromarray(result.astype(np.uint8), 'L')
    output.save('binary_edge_map.bmp')

    return result

def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Uses Canny\' edge detector to create a binary edge map of the input image')
    parser.add_argument('image', help='path to input image')

    args = parser.parse_args()
    print('Processing %s...' % args.image)

    # read the image as black and white image with 8-bit pixels into an array
    try:
        im = Image.open(args.image).convert('L')
    except IOError as err:
        print('Failed to open %s' % args.image)
        sys.exit(0)

    image_arr = np.asarray(im)

    smooth_img = gaussian_smoothing(image_arr)
    print('>>> Gaussian Smoothing Complete')
    
    (gradient_x, gradient_y) = gradient_response(smooth_img)
    print('>>> Gradient Response Calculated')
    
    img_magnitude = gradient_magnitude(gradient_x, gradient_y)
    print('>>> Gradient Magnitude Calculated')
    angle = gradient_angle(gradient_x, gradient_y)

    img_magnitude = non_maxima_suppression(img_magnitude, angle)
    print('>>> Non-Maxima Suppression Complete')

    double_threshold(img_magnitude, angle)
    print('>>> Double Thresholding Complete')

if __name__ == "__main__":
    main()

