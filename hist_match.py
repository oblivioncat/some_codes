import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

if __name__=='__main__':

    # im_raw = tiff.imread('../sample_library/2011_1.tif')
    im_data=tiff.imread('./2011clip.tif')

    ref_data = tiff.imread('./2011_ref2.tif')

    # patch,cord = crop(im_data)
    # cord = [1129, 15615, 757, 36095]
    # match=np.zeros([cord[1]-cord[0], cord[3]-cord[2],3])
    match = np.zeros([im_data.shape[0],im_data.shape[1],3])
    # patch = im_data[cord[0]:cord[1], cord[2]:cord[3],:]
    for i in range(3):
        # match[:,:,i]=hist_match(patch[:,:,i],ref_data[:,:,i])
        match[:,:,i] = hist_match(im_data[:,:,i],ref_data[:,:,i])
    # im_raw[cord[0]:cord[1], cord[2]:cord[3], :] = match
    # im_data.transpose((2,0,1))
    misc.imsave('./2011hist_match.tif',match)
