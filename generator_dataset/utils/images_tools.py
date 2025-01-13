import numpy as np
import functools

@functools.cache
def gaussian_kernel(size=5, dtype=np.float32, normalize=False):
    """
    Generates a Gaussian kernel of size (size, size) that can be used for operations
    such as convolutions for smoothing or blurring images.

    The kernel is computed using the Gaussian distribution formula, with the standard
    deviation determined by the size of the kernel.

    Parameters:
    -----------
    size : int, optional
        The size of the square kernel (must be an odd number). The default is 5.
    
    dtype : data type, optional
        The data type of the resulting kernel. The default is np.float32.
    
    normalize : bool, optional
        If True, normalizes the kernel so that the sum of its elements equals 1. This 
        ensures that the kernel preserves the brightness of the image when convolved. 
        The default is False.
    
    Returns:
    --------
    numpy.ndarray
        A 2D array representing the Gaussian kernel. It has dimensions (size, size) 
        and the specified data type.
    
    Example:
    --------
    kernel = gaussian_kernel(size=7, normalize=True)
    """
    sigma = size / 5.0
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    
    if normalize:
        kernel /= np.sum(kernel)
    
    return kernel.astype(dtype)

def overlay_image_with_blend(base_image, overlay_image, position=(0, 0), alpha_mask=None, center=True, blend_mode=0):
    """
    Overlay `overlay_image` onto `base_image` at the specified position `(x, y)` and blend using the `alpha_mask`.
    
    Parameters:
    - base_image (np.ndarray): The base image onto which the overlay will be applied.
    - overlay_image (np.ndarray): The image to overlay on top of `base_image`.
    - position (tuple): The (x, y) coordinates where the top-left corner of `overlay_image` will be placed. Default is (0, 0).
    - alpha_mask (np.ndarray or float, optional): A mask that controls the transparency of the overlay image. 
      It must have the same height and width as `overlay_image`, with values in the range [0, 1]. If `None`, no mask is applied.
    - center (bool, optional): If True, the overlay image is centered on the `(x, y)` position. Default is True.
    - blend_mode (int, optional): Defines the blending mode to apply. Possible values are:
        - 0: Replace (alpha blending)
        - 1: Additive blending
        - 2: Subtractive blending
        - 3: Multiply blending
        - 4: Divide blending
        
    Returns:
    - np.ndarray: The resulting image after the overlay and blending operation.
    """
    
    # Ensure base_image and overlay_image are numpy arrays
    base_image = base_image.copy()
    
    h_base, w_base = base_image.shape[:2]
    h_overlay, w_overlay = overlay_image.shape[:2]
    
    # Compute position for centering the overlay image
    x, y = position
    if center:
        x -= w_overlay // 2
        y -= h_overlay // 2

    # Define the valid range for overlapping areas
    y1_base, y2_base = max(0, y), min(h_base, y + h_overlay)
    x1_base, x2_base = max(0, x), min(w_base, x + w_overlay)
    
    y1_overlay, y2_overlay = max(0, -y), min(h_overlay, h_base - y)
    x1_overlay, x2_overlay = max(0, -x), min(w_overlay, w_base - x)
    
    # Exit early if no valid overlap
    if y1_base >= y2_base or x1_base >= x2_base or y1_overlay >= y2_overlay or x1_overlay >= x2_overlay:
        return base_image

    # Crop both base image and overlay image to the overlapping region
    base_crop = base_image[y1_base:y2_base, x1_base:x2_base]
    overlay_crop = overlay_image[y1_overlay:y2_overlay, x1_overlay:x2_overlay]

    # Handle alpha mask
    if alpha_mask is None:
        alpha = 1.0
    elif isinstance(alpha_mask, np.ndarray):
        alpha = alpha_mask[y1_overlay:y2_overlay, x1_overlay:x2_overlay, np.newaxis]
    else:
        alpha = alpha_mask
    alpha_inv = 1.0 - alpha
    
    # Blend mode dictionary for easier lookup
    blend_modes = {
        0: lambda base, overlay, alpha, alpha_inv: base * alpha_inv + overlay * alpha,
        1: lambda base, overlay, alpha: base + overlay * alpha,
        2: lambda base, overlay, alpha: base - overlay * alpha,
        3: lambda base, overlay, alpha: base * (overlay * alpha),
        4: lambda base, overlay, alpha: base / (overlay * alpha)
    }
    
    # Get the blending function based on the blend_mode
    if blend_mode not in blend_modes:
        raise ValueError(f"Invalid blend_mode: {blend_mode}. Choose a value between 0 and 4.")
    
    # Perform the blending operation
    blended_crop = blend_modes[blend_mode](base_crop, overlay_crop, alpha, alpha_inv)
    
    # Update the base image with the blended result
    base_crop[:] = blended_crop
    
    return base_image