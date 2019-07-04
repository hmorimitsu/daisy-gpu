import numpy as np
import time
import tensorflow as tf


class DaisyTF(object):
    """ Computes dense DAISY [1] descriptors from a batch of images. It uses
    Tensorflow to perform operations on GPU (if available) to significantly
    speedup the process. This implementation is based on and borrows some
    parts of code from the scikit-image version available at
    https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_daisy.py.

    This code is able to process a batch of images simultaneously for better
    performance. The most expensive operation when running in GPU mode is the
    allocation of the space for the descriptors on the GPU. However, this step
    is only performed when the shape of the input batch changes. Subsequent
    calls using batches with the same shape as before will reuse the memory and
    will, therefore, be much faster.

    Usage:
        from daisy_tf import DaisyTF

        daisy = DaisyTF()
        imgs = [
            read_some_image,
            read_another_image
        ]
        imgs2 = [
            read_yet_another_image,
            read_even_one_more_image
        ]
        imgs_tf, descs_tf = daisy.extract_descriptor(imgs[0].shape[:2])
        with tf.Session() as sess:
            # This first call can be slower, due to memory alloc
            descs = sess.run(
                descs_tf,
                feed_dict={
                    imgs_tf:
                    np.stack(imgs, axis=0)[:, None].astype(np.float32)})
            # Subsequent calls are faster if images retain same shape
            descs2 = sess.run(
                descs_tf,
                feed_dict={
                    imgs_tf:
                    np.stack(imgs2, axis=0)[:, None].astype(np.float32)})

        # descs[0] is the descriptor of images[0] and so on.

    Args:
        (This explanation below is taken directly from scikit-image).
        step : int, optional
            Distance between descriptor sampling points.
        radius : int, optional
            Radius (in pixels) of the outermost ring.
        rings : int, optional
            Number of rings.
        histograms  : int, optional
            Number of histograms sampled per ring.
        orientations : int, optional
            Number of orientations (bins) per histogram.
        normalization : [ 'l1' | 'l2' | 'daisy' | 'off' ], optional
            How to normalize the descriptors
            * 'l1': L1-normalization of each descriptor.
            * 'l2': L2-normalization of each descriptor.
            * 'daisy': L2-normalization of individual histograms.
            * 'off': Disable normalization.
        sigmas : 1D array of float, optional
            Standard deviation of spatial Gaussian smoothing for the center
            histogram and for each ring of histograms. The array of sigmas
            should be sorted from the center and out. I.e. the first sigma
            value defines the spatial smoothing of the center histogram and the
            last sigma value defines the spatial smoothing of the outermost
            ring. Specifying sigmas overrides the following parameter.
                ``rings = len(sigmas) - 1``
        ring_radii : 1D array of int, optional
            Radius (in pixels) for each ring. Specifying ring_radii overrides
            the following two parameters.
                ``rings = len(ring_radii)``
                ``radius = ring_radii[-1]``
            If both sigmas and ring_radii are given, they must satisfy the
            following predicate since no radius is needed for the center
            histogram.
                ``len(ring_radii) == len(sigmas) + 1``

    References:
    [1] E. Tola; V. Lepetit; P. Fua : Daisy: An Efficient Dense Descriptor
        Applied to Wide Baseline Stereo; IEEE TPAMI. 2010.
        DOI : 10.1109/TPAMI.2009.77.
    """
    def __init__(self,
                 step=4,
                 radius=15,
                 rings=3,
                 histograms=8,
                 orientations=8,
                 normalization='l1',
                 sigmas=None,
                 ring_radii=None):
        self.step = step
        self.radius = radius
        self.rings = rings
        self.histograms = histograms
        self.orientations = orientations
        self.normalization = normalization
        self.sigmas = sigmas
        self.ring_radii = ring_radii

        # Validate parameters.
        if sigmas is not None and ring_radii is not None \
                and len(sigmas) - 1 != len(ring_radii):
            raise ValueError('`len(sigmas)-1 != len(ring_radii)`')
        if ring_radii is not None:
            self.rings = len(ring_radii)
            self.radius = ring_radii[-1]
        if sigmas is not None:
            self.rings = len(sigmas) - 1
        if sigmas is None:
            self.sigmas = [
                radius * (i + 1) / float(2 * self.rings)
                for i in range(self.rings)]
            self.sigmas = [self.sigmas[0]] + self.sigmas
        if ring_radii is None:
            self.ring_radii = [
                radius * (i + 1) / float(self.rings)
                for i in range(self.rings)]
        if normalization not in ['l1', 'l2', 'daisy', 'off']:
            raise ValueError('Invalid normalization method.')

        self.sigmas = np.array(self.sigmas)

        self.orientation_kappa = self.orientations / np.pi
        orientation_angles = [2 * o * np.pi / self.orientations - np.pi
                              for o in range(self.orientations)]
        self.orientation_angles = tf.constant(
            np.array(
                orientation_angles).astype(np.float32)[None, :, None, None])

        self.gauss_dx, self.gauss_dy = self._compute_gaussian_kernels(
            self.sigmas, self.orientations)

    def extract_descriptor(self,
                           image_shape):
        """ Main function of this class, which extracts the descriptors from
        a batch of images.

        Args:
            image_shape : Tuple(int, int).
                A tuple (height, width) indicating the size of the input
                images.

        Returns:
            (This explanation below is modified from scikit-image).
            descs : 4D array of floats
                Grid of DAISY descriptors for the given image as an array
                dimensionality  (N, P, Q, R) where
                ``N = len(images)``
                ``P = ceil((M - radius*2) / step)``
                ``Q = ceil((N - radius*2) / step)``
                ``R = (rings * histograms + 1) * orientations``
        """
        images = tf.placeholder(
            tf.float32, shape=[None, 1, image_shape[0], image_shape[1]])
        images /= 255.0

        dx = images[:, :, :, 1:] - images[:, :, :, :-1]
        dx = tf.pad(
            dx,
            tf.constant([[0, 0], [0, 0], [0, 0], [0, 1]]))
        dy = images[:, :, 1:, :] - images[:, :, :-1, :]
        dy = tf.pad(
            dy,
            tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]))

        # Compute gradient orientation and magnitude and their contribution
        # to the histograms.
        grad_mag = tf.sqrt(dx ** 2 + dy ** 2)
        grad_ori = tf.atan2(dy, dx)
        hist = tf.exp(self.orientation_kappa * tf.cos(
            grad_ori - self.orientation_angles))
        hist *= grad_mag

        # Smooth orientation histograms for the center and all rings.
        hist_smooth = self._compute_ring_histograms(hist)

        # Assemble descriptor grid.
        theta = np.array([2 * np.pi * j / self.histograms
                          for j in range(self.histograms)])
        desc_dims = (self.rings * self.histograms + 1) * self.orientations
        desc_shape = (images.shape[2] - 2 * self.radius,
                      images.shape[3] - 2 * self.radius)
        idx = self.orientations
        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)
        descs = [hist_smooth[
            :, 0, :, self.radius:-self.radius, self.radius:-self.radius]]
        for i in range(self.rings):
            for j in range(self.histograms):
                y_min = self.radius + tf.cast(tf.round(
                    self.ring_radii[i] * sin_theta[j]), tf.int32)
                y_max = desc_shape[0] + y_min
                x_min = self.radius + tf.cast(tf.round(
                    self.ring_radii[i] * cos_theta[j]), tf.int32)
                x_max = desc_shape[1] + x_min
                descs.append(hist_smooth[
                    :, i + 1, :, y_min:y_max, x_min:x_max])
                idx += self.orientations
        descs = tf.concat(descs, axis=1)
        descs = descs[:, :, ::self.step, ::self.step]
        descs = tf.transpose(descs, [0, 2, 3, 1])

        # Normalize descriptors.
        if self.normalization != 'off':
            descs += 1e-10
            if self.normalization == 'l1':
                descs /= tf.reduce_sum(descs, axis=3, keepdims=True)
            elif self.normalization == 'l2':
                descs /= tf.sqrt(tf.reduce_sum(
                    tf.pow(descs, 2), axis=3, keepdims=True))
            elif self.normalization == 'daisy':
                for i in range(0, desc_dims, self.orientations):
                    norms = tf.sqrt(tf.reduce_sum(
                        tf.pow(descs[:, :, :, i:i + self.orientations], 2),
                        axis=3, keepdims=True))
                    descs[:, :, :, i:i + self.orientations] /= norms

        return images, descs

    def _compute_one_gaussian_kernel(self,
                                     length,
                                     sigma):
        """ Computes a 1D convolution gaussian kernel.

        Args:
            length : int
                Length of the convolution kernel vector.
            sigma : float
                Standard deviation of the gaussian distribution.

        Returns:
            g : 1D array of float
                The Gaussian kernel vector.
        """
        sigma = max(sigma, 1.0)
        k = np.zeros(length, np.float32)
        half_idx = np.arange(length//2 + 1)
        k[:length//2+1] = half_idx[::-1]
        k[length//2:] = half_idx
        g = ((1.0 / (sigma * np.sqrt(2*np.pi))) *
             np.exp(-0.5 * np.power(((k) / sigma), 2)))
        g = g / g.sum()
        return g

    def _compute_gaussian_kernels(self,
                                  sigmas,
                                  in_channels):
        """ Computes the x and y directional convolution Gaussian kernels.
        The kernels computed by this function over tensors whose channels are a
        stack of all orientations and rings. More specifically, the input
        tensor should have a shape (N, C, H, W) where the channels C are
        defined as:
            ``C = self.rings * self.orientations``
        The channels should be arranged in the following order:
            input[:, 0, :, :] -> orientation 0, ring 0
            input[:, 1, :, :] -> orientation 1, ring 0
            input[:, 2, :, :] -> orientation 2, ring 0
            ...
            input[:, self.orientations, :, :] -> orientation 1, ring 0
            input[:, self.orientations+1, :, :] -> orientation 1, ring 1
            ...

        Returns:
            gx : 4D tf.Tensor
                A horizontal convolution gaussian kernel tensor.
            gy : 4D tf.Tensor
                A vertical convolution gaussian kernel tensor.
        """
        sigmas = self.sigmas.astype(np.int32)
        max_radius = sigmas[-1]
        gx = np.zeros(
            (1, 2*max_radius+1, self.orientations*len(sigmas),
             self.orientations*len(sigmas)), np.float32)
        gy = np.zeros(
            (2*max_radius+1, 1, self.orientations*len(sigmas),
             self.orientations*len(sigmas)), np.float32)
        for i in range(len(sigmas)):
            k = self._compute_one_gaussian_kernel(2*max_radius+1, sigmas[i])
            for j in range(self.orientations):
                gx[0, :, j*len(sigmas) + i, j*len(sigmas) + i] = k
                gy[:, 0, j*len(sigmas) + i, j*len(sigmas) + i] = k
        gx = tf.constant(gx)
        gy = tf.constant(gy)
        return gx, gy

    def _compute_ring_histograms(self,
                                 hist):
        """ Applies Gaussian convolutions of different sizes to obtain the
        histograms for each ring at each orientation.

        Args:
            hist : 4D tf.Tensor
                Raw histograms computed at all orientations. Its shape must
                be (N, C, H, W), where C = self.orientations.

        Returns:
            hist_smooth : 5D tf.Tensor
                The histograms smoothed with different Gaussian kernel for
                all rings and all orientations. Its shape will be
                (N, R, C, H, W), where R = self.rings + 1.
        """
        radius = self.gauss_dx.shape[1] // 2
        hist_smooth = tf.tile(hist, [1, len(self.sigmas), 1, 1])
        hist_smooth = tf.pad(
            hist_smooth, [[0, 0], [0, 0], [0, 0], [radius, radius]], 'REFLECT')
        hist_smooth = tf.nn.conv2d(
            hist_smooth, self.gauss_dx, padding='VALID', data_format='NCHW')
        hist_smooth = tf.pad(
            hist_smooth, [[0, 0], [0, 0], [radius, radius], [0, 0]], 'REFLECT')
        hist_smooth = tf.nn.conv2d(
            hist_smooth, self.gauss_dy, padding='VALID', data_format='NCHW')
        hist_smooth = tf.reshape(
            hist_smooth,
            [-1, self.rings+1, self.orientations,
             hist.get_shape()[2].value, hist.get_shape()[3].value])
        return hist_smooth
