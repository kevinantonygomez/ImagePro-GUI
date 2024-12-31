'''
    This script contains several image processing functions, including SSIF, median blur noise reduction, 
    contrast/saturation modification, brightness/lightness adjustment, histogram equalization, CLAHE, and upscaling. 
    These functions use OpenCV and NumPy
    Author: Kevin Antony Gomez
'''

import cv2
import numpy as np

def SSIF(I:np.ndarray, radius:int=11, epsilon:float=0.01, kappa:float=7.5, scale:float=1.0) -> np.ndarray:
    '''
    Python implementation for 'A Guided Edge-Aware Smoothing-Sharpening Filter Based on Patch Interpolation Model 
    and Generalized Gamma Distribution'
    @article{SSIF_Deng2021,
	    author={G. {Deng} and F. J. {Galetto} and M. {Al-nasrawi} and W. {Waheed}},
 	    journal={IEEE Open Journal of Signal Processing}, 
 	    title={A guided edge-aware smoothing-sharpening filter based on patch interpolation model and generalized Gamma distribution}, 
 	    year={2021},
 	    doi={10.1109/OJSP.2021.3063076}
    } 
    Args:
        I (np.ndarray): Input BGR image
        radius (int): Radius of the filter patch
        epsilon (float): Small constant to avoid division by zero in calculations
        kappa (float): Controls the influence of local variations in image details
        scale (float): user-defined scale param
    Returns:
        J (np.ndarray): Output image after applying the SSIF filter
    '''
    if kappa == 0:
        return I
    I = I / 255.0
    G = I.copy()
    patch_size = 2 * radius + 1
    h = np.ones((patch_size, patch_size)) / (patch_size * patch_size)
    mu = cv2.filter2D(I, -1, h, borderType=cv2.BORDER_REFLECT)
    nu = cv2.filter2D(G, -1, h, borderType=cv2.BORDER_REFLECT)
    phi = cv2.filter2D(I * G, -1, h, borderType=cv2.BORDER_REFLECT) - mu * nu
    var_sigma = np.maximum(0, cv2.filter2D(G * G, -1, h, borderType=cv2.BORDER_REFLECT) - nu * nu)
    a = phi / (var_sigma + epsilon)
    Beta = (a + np.sign(phi) * np.sqrt(a**2 + 4 * kappa * epsilon / (var_sigma + epsilon))) / 2
    w = var_sigma / (scale * np.mean(var_sigma))
    w = 1 / (1 + w**2)
    normalize_factor = cv2.filter2D(w, -1, h, borderType=cv2.BORDER_REFLECT)
    A = cv2.filter2D(Beta * w, -1, h, borderType=cv2.BORDER_REFLECT)
    B = cv2.filter2D((mu - Beta * nu) * w, -1, h, borderType=cv2.BORDER_REFLECT)
    J = (G * A + B) / normalize_factor
    J = np.clip(J * 255, 0, 255).astype(np.uint8)
    return J

    
def denoise_medianBlur(img_bgr:np.ndarray, k_size:int=5) -> np.ndarray:
    '''
    Denoises an image using a median blur filter
    Args:
      img_bgr (np.ndarray): Input BGR image
      k_size (int): Size of the kernel (must be an odd number). Default is 5.
        Larger values will result in more aggressive smoothing
    Returns:
      np.array: Denoised image
    '''
    return cv2.medianBlur(img_bgr, ksize=k_size)
    

def hist_eq_ycrcb(img_bgr:np.ndarray) -> np.ndarray:
    '''
    Applies histogram equalization to the Y channel of an image in YCrCb color space.
    This enhances the contrast of the image by equalizing the histogram of the luminance (Y) channel
    Args:
      img_bgr (np.ndarray): Input BGR image
    Returns:
      np.ndarray: Histogram equalized image
    '''
    ycrcb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    # convert back to BGR color-space from YCrCb
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


def clahe_ycrcb(img_bgr:np.ndarray, clip_limit:int) -> np.ndarray:
    '''
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the Y channel of an image 
    in YCrCb color space
    Args:
      img_bgr (np.ndarray): Input BGR image
      clip_limit (int): Threshold for contrast limiting
    Returns:
      np.ndarray: Enhanced image
    '''
    if clip_limit == 0:
        return img_bgr
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4,4))
    ycrcb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


def saturate(img_bgr:np.ndarray, value:float) -> np.ndarray:
    '''
    Modifies the saturation of an image by scaling the saturation channel in HSV color space
    Args:
      img_bgr (np.ndarray): Input BGR image
      value (float): Factor to scale the saturation channel
    Returns:
      np.ndarray: Saturated image
    '''
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, value)  # Scale saturation
    s = np.clip(s, 0, 255)  # Ensure values stay within valid range
    # Merge channels and convert back to BGR
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    return enhanced_image


def contrast(img_bgr:np.ndarray, value:float) -> np.ndarray:
    '''
    Adjusts the contrast of an image by applying a contrast factor
    Args:
      img_bgr (np.ndarray): Input BGR image
      value (float): Contrast factor
    Returns:
      np.ndarray: Contrast-adjusted image
    '''
    image = np.float32(img_bgr)
    image = (image - 128) * value + 128 # Apply contrast factor
    # Clip values to stay within the range [0, 255]
    image = np.clip(image, 0, 255)
    image = np.uint8(image) # Convert back to uint8
    return image


def brighten(img_bgr:np.ndarray, value:int) -> np.ndarray:
    '''
    Brightens or darkens an image by adding a brightness factor
    Args:
      img_bgr (np.ndarray): Input BGR image
      value (int): Brightness factor
    Returns:
      np.ndarray: Brightened/darkened image
    '''
    image = np.float32(img_bgr)
    # Add brightness factor (this will brighten or darken the image)
    image = image + value
    # Clip the values to stay within the valid range [0, 255]
    image = np.clip(image, 0, 255)
    # Convert back to uint8
    image = np.uint8(image)
    return image


def lighten(img:np.ndarray, value:int) -> np.ndarray:
    '''
    Lightens an image by increasing the value channel in HSV color space
    Args:
      img_bgr (np.ndarray): Input BGR image
      value (int): Amount to add to the value channel
    Returns:
      np.ndarray: Lightened image
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    v[:] = cv2.add(v, value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def upscale_lancoz(img:np.ndarray, value:float) -> np.ndarray:
    '''
    Upscales an image using Lanczos interpolation.
    Args:
      img_bgr (np.ndarray): Input BGR image
      value (float): Scaling factor
    Returns:
      np.ndarray: Upscaled image
    '''
    image_height, image_width, _ = img.shape
    new_width = int(image_width * value)
    new_height = int(image_height * value)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
