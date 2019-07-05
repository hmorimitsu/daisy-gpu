# daisy-gpu

Implementation of the DAISY descriptor [[1]](#references) in GPU using deep learning libraries.
Codes are provided for PyTorch, Tensorflow 1, and Tensorflow 2.

This implementation is based on and borrows some
parts of code from the scikit-image version available at
[https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_daisy.py](https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_daisy.py).

This code is able to process a batch of images simultaneously for better
performance. The most expensive operation when running in GPU mode is the
allocation of the space for the descriptors on the GPU. However, this step
is only performed when the shape of the input batch changes. Subsequent
calls using batches with the same shape as before will reuse the memory and
will, therefore, be much faster.

## Requirements

### Base

- [Python 3](https://www.python.org/) (Tested on 3.7)
- [Numpy](https://www.numpy.org/)

### Deep learning libraries

- [PyTorch](https://pytorch.org/) >= 1.0.0 (Tested on 1.1.0) **or**
- [Tensorflow 1.X](https://www.tensorflow.org/) (Tested on 1.14) **or**
- [Tensorflow 2.X](https://www.tensorflow.org/) (Tested on 2.0.0-beta1)

## Usage

### PyTorch version

```python
from daisy_torch import DaisyTorch

daisy = DaisyTorch()
imgs = [
    read_some_image,
    read_another_image
]
descs = daisy.extract_descriptor(imgs) # This first call can be
                                       # slower, due to memory allocation
imgs2 = [
    read_yet_another_image,
    read_even_one_more_image
]
descs2 = daisy.extract_descriptor(imgs2) # Subsequent calls are faster,
                                         # if images retain same shape

# descs[0] is the descriptor of imgs[0] and so on.
```

### Tensorflow 1 version

```python
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
```

### Tensorflow 2 version
```python
from daisy_tf2 import DaisyTF2

daisy = DaisyTF2()
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
```

## Benchmark

- Machine configuration:
  - Intel i7 8750H
  - NVIDIA GeForce GTX1070
  - Images 1024 x 436
  - Descriptor size 200

Batch Size|PyTorch<br />Time CPU(ms)|PyTorch<br />Time GPU(ms)<sup>1</sup>|TF2<br />Time GPU(ms)<sup>1</sup>|PyTorch<br />Time GPU(ms)<sup>2</sup>|TF1<br />Time GPU(ms)<sup>2</sup>|TF2<br />Time GPU(ms)<sup>2</sup>
-|------|---|---|-----|-----|-----
1| 428.8|2.4|1.5| 35.5| 21.0| 32.5
2| 786.5|3.2|2.7| 68.4| 39.8| 58.0
4|1973.1|5.1|4.1|127.2| 77.2|114.2
8|3042.5|8.9|6.4|250.8|151.4|227.1

<sup>1</sup> NOT including time to transfer result from GPU to CPU

<sup>2</sup> Including time to transfer result from GPU to CPU

These times are the median of 5 runs measured after a warm up run to allocate the descriptor space in memory
(read the [introduction](#daisy-pytorch)).

## References

[1] E. Tola; V. Lepetit; P. Fua : Daisy: An Efficient Dense Descriptor Applied to Wide Baseline Stereo;
IEEE TPAMI. 2010. DOI : 10.1109/TPAMI.2009.77.