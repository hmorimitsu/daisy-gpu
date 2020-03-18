# daisy-gpu

Implementation of the DAISY descriptor [[1]](#references) on GPU using deep learning libraries.
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

Code for SIFT Flow descriptors on GPU is also available at [https://github.com/hmorimitsu/sift-flow-gpu](https://github.com/hmorimitsu/sift-flow-gpu).

## Requirements

### Base

- [Python 3](https://www.python.org/) (Tested on 3.7)
- [Numpy](https://www.numpy.org/)

### Deep learning libraries

- [PyTorch](https://pytorch.org/) >= 1.0.0 (Tested on 1.4.0) **or**
- [Tensorflow 1.X](https://www.tensorflow.org/) (Tested on 1.14) **or**
- [Tensorflow 2.X](https://www.tensorflow.org/) (Tested on 2.1.0)

## Usage

### PyTorch version

A simple example is shown below. A more complete practical usage is available as a [Jupyter demo notebook](demo_notebook_torch.ipynb)

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

**Update:** The times reported previously were incorrect, because they were being computed without synchronizing the GPU.
The table below was updated with the corrected values.

- Machine configuration:
  - Intel i7 8750H
  - NVIDIA GeForce GTX1070
  - Images 1024 x 436
  - Descriptor size 200

Batch Size|PyTorch<br />Time CPU(ms)|PyTorch<br />Time GPU(ms)<sup>1</sup>|PyTorch FP16<br />Time GPU(ms)<sup>1</sup>|TF2<br />Time GPU(ms)<sup>1</sup>|PyTorch<br />Time GPU(ms)<sup>2</sup>|TF1<br />Time GPU(ms)<sup>2</sup>|TF2<br />Time GPU(ms)<sup>2</sup>
-|------|-----|-----|-----|-----|-----|-----
1| 309.8| 27.9| 25.0| 21.3| 37.9| 26.5| 31.6
2| 534.9| 39.8| 34.8| 38.0| 57.1| 48.2| 63.4
4| 998.3| 79.6| 67.1| 75.3|113.5| 92.6|123.6
8|2009.8|158.3|134.9|150.4|226.4|187.0|251.1

<sup>1</sup> NOT including time to transfer the result from GPU to CPU

<sup>2</sup> Including time to transfer the result from GPU to CPU

These times are the median of 5 runs measured after a warm up run to allocate the descriptor space in memory
(read the [introduction](#daisy-pytorch)).

## References

[1] E. Tola; V. Lepetit; P. Fua : Daisy: An Efficient Dense Descriptor Applied to Wide Baseline Stereo;
IEEE TPAMI. 2010. DOI : 10.1109/TPAMI.2009.77.
