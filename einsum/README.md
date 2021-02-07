#### `Ein`stein `Sum`mation (einsum)

**Intro** : Since the description of einsum is skimpy in torch documentation<sup id="torcheinsumdoc">[1](#fn1)</sup>, we will first understand [**`numpy.einsum()`**][3] based on the semantics of which **`torch.einsum()`**<sup id="torcheinsum">[2](#fn2)</sup>  is implemented. Then, we will compare and contrast how **`torch.einsum()`**<sup id="torcheinsum">[2](#fn2)</sup>  behaves when compared to [`numpy.einsum()`][3].

--------------------------

**Understanding** : Grasping the idea of einsum is easy if you understand it intuitively. As an example, let's start with a simple description involving *matrix multiplication* which is ubiquitous in deep learning.

---------

To use [**`numpy.einsum()`**][3] or **`torch.einsum()`**<sup id="torcheinsum">[2](#fn2)</sup> , all you have to do is to pass the so-called *subscripts string* as an argument, followed by your _input arrays_ (or equivalently tensors in PyTorch).

Let's say you have two 2D arrays, **`A`** and **`B`**, and you want to perform matrix multiplication. So, you do:

```python
numpy.einsum("ij, jk -> ik", A, B)    # A, B are 2D array-like
torch.einsum('ij, jk -> ik', aten, bten)  # aten, bten are 2D torch tensors
```

In the above lines of code, the first part of the *subscript string* **`ij`** corresponds to the first array (i.e., **`A`**) while the **`jk`** in the *subscript string* corresponds to the second array (i.e., **`B`**). Also, the most important thing one must note here is that the *number of characters* in each *subscript string* **must** match the dimensions of the array/tensor. (i.e. two chars for 2D arrays/tensors, three chars for 3D arrays/tensors, and so on.) And if you repeat the characters in the first half of the *subscript strings* (**`j`** in our case), then that means you want the `ein`*sum* to happen along those dimensions. Thus, they will be sum-reduced (i.e. that dimension will be _gone_ in the output).

The *subscript string* after this **`->`**, corresponds to the resultant array/tensor. If you leave it empty, then everything will be summed and a single scalar value is returned as the result. Else the resultant array will have dimensions according to the *subscript string*. In our example, it'll be **`ik`**. This is intuitive because we know that for matrix multiplication the constraint is that the number of columns in array **`A`** has to match the number of rows in array **`B`** , which is what is happening here (i.e., we encode this knowledge by repeating the char **`j`** in the *subscript string*).

**Differences** :

  - NumPy allows both small case and capitalized letters `[a-zA-Z]` for the "*subscript string*" whereas PyTorch allows only small case letters `[a-z]`.

  - NumPy accepts nd-arrays, plain Python lists (or equivalently tuples), list of lists (or equivalently tuple of tuples, list of tuples, tuple of lists) or even PyTorch tensors as *operands* (i.e. inputs). This is because the *operands* have only to be *array_like* and not strictly NumPy nd-arrays. On the contrary, PyTorch expects the *operands* (i.e. inputs) strictly to be PyTorch tensors. It will throw a `TypeError` if you pass either plain Python lists, tuples (or its combinations) or NumPy nd-arrays.

  - NumPy supports lot of keyword arguments (e.g. `optimize`) in addition to the `nd-array` inputs while PyTorch doesn't offer any such flexibility yet.

    

Now that we have an understanding of how to frame the *subscript strings* for the desired operation, it is time to delve into some exercises. In the jupyter notebook, we will see implementation of some tensor and linear algebra operations, both in PyTorch and NumPy:

---------------------------------

<b id="fn1">1. https://pytorch.org/docs/stable/generated/torch.einsum.html  </b> [:arrow_heading_up:](#torcheinsum) 

<b id="fn2">2. https://pytorch.org/docs/stable/_modules/torch/functional.html#einsum </b> [:arrow_heading_up:](#torcheinsum) 

[3]: https://numpy.org/devdocs/reference/generated/numpy.einsum.html
[4]: https://en.wikipedia.org/wiki/Frobenius_inner_product