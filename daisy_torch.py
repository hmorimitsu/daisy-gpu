import numpy as np
import torch
import torch.nn.functional as F


class DaisyTorch(object):
    """ Computes dense DAISY [1] descriptors from a batch of images. It uses
    PyTorch to perform operations on GPU (if available) to significantly
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
        from daisy_torch import DaisyTorch

        daisy = DaisyTorch()
        imgs = [
            read_some_image,
            read_another_image
        ]
        descs = daisy.extract_descriptor(imgs) # This first call can be
                                               # slower, due to memory alloc
        imgs2 = [
            read_yet_another_image,
            read_even_one_more_image
        ]
        descs2 = daisy.extract_descriptor(imgs2) # Subsequent calls are faster,
                                                 # if images retain same shape

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
        fp16 : boolean : optional
            If True, use half precision floting point tensors.

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
                 ring_radii=None,
                 fp16=False,
                 return_numpy=False):
        self.step = step
        self.radius = radius
        self.rings = rings
        self.histograms = histograms
        self.orientations = orientations
        self.normalization = normalization
        self.sigmas = sigmas
        self.ring_radii = ring_radii
        self.fp16 = fp16
        self.return_numpy = return_numpy

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

        self.cuda = torch.cuda.is_available()

        self.sigmas = np.array(self.sigmas)

        self.orientation_kappa = self.orientations / np.pi
        orientation_angles = [2 * o * np.pi / self.orientations - np.pi
                              for o in range(self.orientations)]
        self.orientation_angles = torch.from_numpy(
            np.array(orientation_angles)[None, :, None, None])
        if self.fp16:
            self.orientation_angles = self.orientation_angles.half()
        else:
            self.orientation_angles = self.orientation_angles.float()
        if self.cuda:
            self.orientation_angles = self.orientation_angles.cuda()

        self.gauss_dx, self.gauss_dy = self._compute_gaussian_kernels(
            self.sigmas, self.orientations)

        self.dx = None
        self.dy = None
        self.descs = None
        self.max_batch_size = 1

    def extract_descriptor(self,
                           images):
        """ Main function of this class, which extracts the descriptors from
        a batch of images.

        Args:
            images : list of 2D array of int or float.
                List of images to form the batch. All images must have be
                grayscale (only two dimensions) and its values are assumed
                to be in the interval [0, 255]. All images must have the same
                size.

        Returns:
            (This explanation below is modified from scikit-image).
            descs : 4D array of floats
                Grid of DAISY descriptors for the given image as an array
                dimensionality  (N, R, P, Q) where
                ``N = len(images)``
                ``R = (rings * histograms + 1) * orientations``
                ``P = ceil((M - radius*2) / step)``
                ``Q = ceil((N - radius*2) / step)``
        """
        images = np.stack(images, axis=0)[:, None]
        images = torch.from_numpy(images.astype(np.float32)) / 255.0
        if self.fp16:
            images = images.half()
        else:
            images = images.float()
        if self.cuda:
            images = images.cuda()

        self.batch_size = images.shape[0]
        self.max_batch_size = max(self.max_batch_size, self.batch_size)

        if (self.dx is None or self.dx.shape[0] < self.max_batch_size or
                self.dx.shape[2] != images.shape[2] or
                self.dx.shape[3] != images.shape[3]):
            shape = (self.max_batch_size,) + images.shape[1:]
            self.dx = torch.zeros(shape)
            if self.fp16:
                self.dx = self.dx.half()
            else:
                self.dx = self.dx.float()
            if self.cuda:
                self.dx = self.dx.cuda()
            self.dy = torch.zeros(shape)
            if self.fp16:
                self.dy = self.dy.half()
            else:
                self.dy = self.dy.float()
            if self.cuda:
                self.dy = self.dy.cuda()
        dx = self.dx[:self.batch_size]
        dx[:, :, :, :-1] = (images[:, :, :, 1:] - images[:, :, :, :-1])
        dy = self.dy[:self.batch_size]
        dy[:, :, :-1, :] = (images[:, :, 1:, :] - images[:, :, :-1, :])

        # Compute gradient orientation and magnitude and their contribution
        # to the histograms.
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2)
        grad_ori = torch.atan2(dy, dx)
        hist = torch.exp(self.orientation_kappa * torch.cos(
            grad_ori - self.orientation_angles))
        hist *= grad_mag

        # Smooth orientation histograms for the center and all rings.
        hist_smooth = self._compute_ring_histograms(hist)

        # Assemble descriptor grid.
        theta = np.array([2 * np.pi * j / self.histograms
                          for j in range(self.histograms)])
        desc_dims = (self.rings * self.histograms + 1) * self.orientations
        desc_shape = (self.max_batch_size, desc_dims,
                      images.shape[2] - 2 * self.radius,
                      images.shape[3] - 2 * self.radius)
        if self.descs is None or self.descs.shape != desc_shape:
            self.descs = torch.empty(desc_shape)
            if self.fp16:
                self.descs = self.descs.half()
            else:
                self.descs = self.descs.float()
            if self.cuda:
                self.descs = self.descs.cuda()
        descs = self.descs[:self.batch_size]
        descs[:, :self.orientations, :, :] = hist_smooth[
            :, 0, :, self.radius:-self.radius, self.radius:-self.radius]
        idx = self.orientations
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        for i in range(self.rings):
            for j in range(self.histograms):
                y_min = self.radius + int(round(
                    self.ring_radii[i] * sin_theta[j]))
                y_max = descs.shape[2] + y_min
                x_min = self.radius + int(round(
                    self.ring_radii[i] * cos_theta[j]))
                x_max = descs.shape[3] + x_min
                # print(i, j, y_min, y_max, x_min, x_max)
                descs[:, idx:idx + self.orientations, :, :] = hist_smooth[
                    :, i + 1, :, y_min:y_max, x_min:x_max]
                idx += self.orientations
        descs = descs[:, :, ::self.step, ::self.step]

        # Normalize descriptors.
        if self.normalization != 'off':
            if self.fp16:
                descs += 1e-3
            else:
                descs += 1e-10
            if self.normalization == 'l1':
                descs /= torch.sum(descs, dim=1, keepdim=True)
            elif self.normalization == 'l2':
                descs /= torch.sqrt(torch.sum(
                    torch.pow(descs, 2), dim=1, keepdim=True))
            elif self.normalization == 'daisy':
                for i in range(0, desc_dims, self.orientations):
                    norms = torch.sqrt(torch.sum(
                        torch.pow(descs[:, i:i + self.orientations], 2),
                        dim=1, keepdim=True))
                    descs[:, i:i + self.orientations] /= norms

        if self.return_numpy:
            descs = descs.detach().cpu().numpy()

        return descs

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
            gx : 4D torch.Tensor
                A horizontal convolution gaussian kernel tensor.
            gy : 4D torch.Tensor
                A vertical convolution gaussian kernel tensor.
        """
        sigmas = self.sigmas.astype(np.int32)
        max_radius = sigmas[-1]
        gx = np.zeros(
            (self.orientations*len(sigmas), self.orientations*len(sigmas),
             1, 2*max_radius+1), np.float32)
        gy = np.zeros(
            (self.orientations*len(sigmas), self.orientations*len(sigmas),
             2*max_radius+1, 1), np.float32)
        for i in range(len(sigmas)):
            k = self._compute_one_gaussian_kernel(2*max_radius+1, sigmas[i])
            for j in range(self.orientations):
                gx[j*len(sigmas) + i, j*len(sigmas) + i, 0] = k
                gy[j*len(sigmas) + i, j*len(sigmas) + i, :, 0] = k
        gx = torch.from_numpy(gx)
        gy = torch.from_numpy(gy)
        if self.fp16:
            gx = gx.half()
            gy = gy.half()
        else:
            gx = gx.float()
            gy = gy.float()
        if self.cuda:
            gx = gx.cuda()
            gy = gy.cuda()
        return gx, gy

    def _compute_ring_histograms(self,
                                 hist):
        """ Applies Gaussian convolutions of different sizes to obtain the
        histograms for each ring at each orientation.

        Args:
            hist : 4D torch.Tensor
                Raw histograms computed at all orientations. Its shape must
                be (N, C, H, W), where C = self.orientations.

        Returns:
            hist_smooth : 5D torch.Tensor
                The histograms smoothed with different Gaussian kernel for
                all rings and all orientations. Its shape will be
                (N, R, C, H, W), where R = self.rings + 1.
        """
        radius = self.gauss_dx.shape[3] // 2
        hist_smooth = hist.repeat(1, len(self.sigmas), 1, 1)
        hist_smooth = F.pad(
            hist_smooth, (radius, radius, 0, 0), mode='reflect')
        hist_smooth = F.conv2d(hist_smooth, self.gauss_dx)
        hist_smooth = F.pad(
            hist_smooth, (0, 0, radius, radius), mode='reflect')
        hist_smooth = F.conv2d(hist_smooth, self.gauss_dy)
        hist_smooth = hist_smooth.reshape(
            hist.shape[0], self.rings+1, self.orientations,
            hist.shape[2], hist.shape[3])
        return hist_smooth
