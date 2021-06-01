"""
Convenience functions for registration related image methods

Author: Fabian A. Preiss.
"""
from typing import Dict, Hashable, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.linalg import norm
from scipy.fftpack import fft2, fftshift
from skimage.filters import difference_of_gaussians, window
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, radon, warp, warp_polar


class ImageMethods:
    """Collection of static methods for image analysis and manipulation."""

    @staticmethod
    def norm_rel_l2(image: np.ndarray, ref_image: np.ndarray) -> float:
        r"""
        Compute a relative similarity measurement between two images.

        Interpretes the images as a vector and calculates the L2 norm of the differences relative to the
        reference image `ref_image`.

        Notes
        -----
        L2 norm  Implemented analog to `NormRel_L2` [#f1]_.

        .. math::
            \frac{\left\Vert \mathtt{ref\_image}-\mathtt{image}\right\Vert _{F}}{\left\Vert \mathtt{ref\_image}\right\Vert _{F}}

        Where :math:`\left\Vert \ldots\right\Vert _{F}` denotes the Frobenius norm of a matrix.

        References
        ----------
        .. [#f1] `NVIDIA Performance Primitives (NPP) - Image Norms
                 <https://docs.nvidia.com/cuda/archive/9.2/npp/group__image__norm.html>`_
        """
        return cast(float, norm(ref_image - image) / norm(ref_image))

    @staticmethod
    def abs_diff(image: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
        r"""
        Absolute value of the difference between two images.

        Notes
        -----
        .. math::
            \mathtt{abs}\left(\mathtt{ref\_image}-\mathtt{image}\right)
        """
        return cast(np.ndarray, abs(ref_image - image))

    @staticmethod
    def sqr_diff(image: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
        r"""
        Squared difference between two images.

        Notes
        -----
        .. math::
            \left(\mathtt{ref\_image}-\mathtt{image}\right)^{2}
        """
        return cast(np.ndarray, (ref_image - image) ** 2)

    @staticmethod
    def exp_filter(
        image: np.ndarray, signal_noise_ratio: Optional[float] = None
    ) -> np.ndarray:
        """
        Remap the values of the image such that bright pixels are given an exponentially higher weight.

        Maps the min value of the input image to 1/signal_noise_ratio and the max value to 1.
        """
        if signal_noise_ratio is not None:
            norm_image = np.interp(
                image, (image.min(), image.max()), (-np.log(signal_noise_ratio), 0)
            )
            return cast(np.ndarray, np.exp(norm_image))
        return image

    @staticmethod
    def sinogram(image, theta=None, exp_filter_val=None, circle=False) -> np.ndarray:
        """Computes the radon transformation and optionally applies the `exp_filter` afterwards.

        Examples
        --------
        .. plot:: tutorial/pyplots/sinogram.py
           :include-source:
        """
        sinogram = radon(image, theta=theta, circle=circle)
        if exp_filter_val is not None:
            sinogram = ImageMethods.exp_filter(sinogram, exp_filter_val)
        return cast(np.ndarray, sinogram)

    @staticmethod
    def sinogram_project(
        image, theta=None, exp_filter_val=None, circle=False
    ) -> np.ndarray:
        """Projects the sinogram onto the axis of the theta angle.

        Examples
        --------
        .. plot:: tutorial/pyplots/sinogram_project.py
           :include-source:
        """
        sinogram = ImageMethods.sinogram(
            image, theta=theta, exp_filter_val=exp_filter_val, circle=circle
        )
        projection = np.sum(sinogram, axis=0)
        return cast(np.ndarray, projection)

    @staticmethod
    def max_sinogram_angle(
        image, theta=None, exp_filter_val=None, circle=False, precision=0.1
    ) -> float:
        """Iterative solver, works well in special cases. Implementation very crude."""
        if theta is None:
            theta = np.arange(0, 180, 1)
        projection = ImageMethods.sinogram_project(
            image, theta=theta, exp_filter_val=exp_filter_val, circle=circle
        )
        max_index = np.where(np.max(projection) == projection)
        angles = theta[max_index]
        if len(max_index[0]) > 1:
            print(
                f"Warning: angle solution not unique returning first match, possible angles: {angles}"
            )
        delta_theta = theta[1] - theta[0]
        if delta_theta > precision:
            theta = np.linspace(
                angles[0] - delta_theta * 2, angles[0] + delta_theta * 2, 10
            )
            return ImageMethods.max_sinogram_angle(
                image,
                theta=theta,
                exp_filter_val=exp_filter_val,
                precision=precision,
                circle=circle,
            )
        return cast(float, angles[0] % 180)

    @staticmethod
    def compute_rts(
        image: np.ndarray,
        angle: float = 0,
        scale: float = 1,
        translation: Sequence[float] = (0.0, 0.0),
        inverse: bool = False,
        preserve_range: bool = True,
        order: int = 5,
    ) -> np.ndarray:
        """
        Rotate, translate and scale image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image to transform
        angle : float, optional
            The rotation angle in degrees for the transform
        scale : float, optional
            The scaling factor used in the transform
        translation : (float, float), optional
            x,y-translations used in the transform
        inverse : bool
            Apply the backwards transformation for given parameters

        Returns
        -------
        numpy.ndarray
            modified image with same shape as initial image
        """
        # AffineTransform rotates around (x,y)=(0,0) instead of
        # image center => calculate coordinate transformation
        _translation = np.array(translation, dtype="float64")
        tform = AffineTransform(scale=scale, rotation=np.deg2rad(angle))
        # change in ordering required, as numpy.ndarray uses matrix-like row/column indexing
        shape_half = (np.array(image.shape) / 2)[::-1]
        _translation = shape_half + tform.params[:2, :2] @ (_translation - shape_half)
        tform.translation[:] = _translation
        if not inverse:
            rts_image = warp(
                image, tform, order=order, preserve_range=preserve_range
            )  # ,mode='wrap')
        else:
            rts_image = warp(
                image, tform.inverse, order=order, preserve_range=preserve_range
            )  # ,mode='wrap')
        return cast(np.ndarray, rts_image)

    @staticmethod
    def compute_dgfw(
        image: np.ndarray,
        gaussdiff: Sequence[float] = (5, 20),
        windowweight: float = 1,
        windowtype: str = "hann",
    ) -> np.ndarray:
        """Image Difference of Gaussian Filter + Window.

        Parameters
        ----------
        image : numpy.ndarray
            The input image to filter
        gaussdiff : (float, float), optional
            The low and high standard deviations for the gaussian difference band pass filter
        windowweight : float, optional
            weighting factor scaling beween windowed image and image
        windowtype : str, optional
            see `skimage.filters.window` for possible choices

        Returns
        -------
        numpy.ndarray
            modified image with bandpass filter and window applied

        Notes
        -----
        Applying this bandpass and window filter prevents artifacts from image boundaries and
        noise from contributing significantly to the fourier transform. The gaussian
        difference filter can be tuned such that the features relevant for the identification
        of the rotation angle are at the center of the band pass filter.
        """
        image_dgfw = (
            difference_of_gaussians(image, *gaussdiff)
            if gaussdiff[0] < gaussdiff[1]
            else image * 1.0
        )
        image_dgfw *= windowweight * window(windowtype, image.shape) + (
            1 - windowweight
        )
        return cast(np.ndarray, image_dgfw)

    @staticmethod
    def compute_afts(image: np.ndarray) -> np.ndarray:
        """Compute FFT magnitude, shifted with low fequencies in center.

        Parameters
        ----------
        image : numpy.ndarray
            The input image for the fourier transform

        Returns
        -------
        numpy.ndarray
            FFT magnitude, center shifted
        """
        image_fs = np.abs(fftshift(fft2(image)))
        return cast(np.ndarray, image_fs)

    @staticmethod
    def compute_log_polar_tf(
        image: np.ndarray, wrexp: float = 3, order: int = 5
    ) -> np.ndarray:
        """Compute log-scaled polar coordinate transform of center shifted FFT.

        Parameters
        ----------
        image : numpy.ndarray
            The input image to transform (expects center shifted FFT magnitude)
        wrexp : float, optional
            Cutoff exponent factor for higher frequencies, larger wrexp => faster computation
            min value: 1

        Returns
        -------
        numpy.ndarray
            log-scaled polar transformed of the input image
        """
        warp_radius = ImageMethods.compute_warp_radius(min(image.shape), wrexp)
        # NOTE: we can use 5th order approximation here, as a the computation time on our image
        # only increases lineary with the interpolation order, whereas the mean squared error
        # decreases pretty much exponentially
        warped = warp_polar(
            image,
            radius=warp_radius,
            output_shape=image.shape,
            scaling="log",
            order=order,
        )
        # FFT magnitude is symmetric => div by 2
        return cast(np.ndarray, warped[: image.shape[0] // 2, :])

    @staticmethod
    def compute_warp_radius(image_diameter: int, wrexp: float = 1.0) -> int:
        """Compute the warp radius from image and warp radius exponent *wrexp*.

        Parameters
        ----------
        image_diameter : int
            The length of the smallest image dimension
        wrexp : float, optional
            Cutoff exponent factor for higher frequencies, larger wrexp => faster computation
            min value: 1

        Returns
        -------
        int
            The cutoff radius for the log-ploar transform of the image
        """
        return int(image_diameter // (2 ** max(wrexp, 1.0)))

    # NOTE: ugly return params
    @staticmethod
    def recover_rs(
        image_warped_fs: np.ndarray,
        rts_warped_fs: np.ndarray,
        image_shape: Sequence[int],
        upsampl: int = 10,
        wrexp: float = 3,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Hashable]]:
        """Recover the rotation and scaling transformation from given input.

        Parameters
        ----------
        image_warped_fs : np.ndarray
            log-polar warped fourier transformed of original input image
        rts_warped_fs : np.ndarray
            log-polar warped fourier transformed of modified input image
        image_shape : Sequence[int]
            image dimensions of original input image
        upsampl : int, optional
            Upsampling factor. 1 => no upsampling, 20 => precision to 1/20 of a pixel
        wrexp : float, optional
            Cutoff exponent factor for higher frequencies, larger wrexp => faster computation
            min value: 1

        Returns
        -------
        numpy.ndarray
            Vector of recovered rotation angle and error in degrees
        numpy.ndarray
            Vector of recovered scaling factor and error
        dict
            Dict containing the phase_cross_correlation parameters

        Notes
        -----
        The errors are a lower estimate under ideal assumptions and can be much larger depending on the data.
        """
        warp_radius = ImageMethods.compute_warp_radius(min(image_shape), wrexp)
        shifts, error, phasediff = phase_cross_correlation(
            image_warped_fs, rts_warped_fs, upsample_factor=upsampl
        )
        phaseparams = {"shifts": shifts, "error": error, "phasediff": phasediff}
        shift_err_r = np.array([shifts[0], error])
        shift_err_c = np.array([shifts[1], error])
        recovered_rotation = cast(np.ndarray, (360 / image_shape[0]) * shift_err_r)
        klog = image_shape[1] / np.log(warp_radius)
        recovered_scale = cast(np.ndarray, np.exp(shift_err_c / klog))
        return recovered_rotation, recovered_scale, phaseparams
