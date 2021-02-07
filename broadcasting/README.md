### Broadcasting

----------------

**Intro** : Understanding broadcasting<sup id="numpybroadcastingdoc">[1](#fn1)</sup> equips programmers with skills for writing memory-efficient code. The idea to always have in mind, as far as broadcasting is concerned, is that we do not *always* have to store duplicate values in tensors for the purpose of achieving some operations, for instance *hadamard product*<sup id="hadamardproductwiki">[3](#fn3)</sup> . First we will develop an intuitive understanding before diving into details.

--------------------------

**Intuition** :

<img src="http://jalammar.github.io/images/numpy/numpy-matrix-broadcast.png" style="zoom:120%;" /> 



In the above figure<sup id="visualnumpy">[4](#fn4)</sup> ,  we are adding two tensors of *non-matching shape*. This is possible in PyTorch<sup id="pytorchbroadcastingdoc">[2](#fn2)</sup> because  it will take care of value (*non-*)duplication under the hood, as long as the following **conditions** are met:

- the tensors are at least 1D each
- while comparing the values in the shape tuple across all tensors, from right-to-left, the values (i.e., dimensions) should be one of the following:
  - same
  - one of them is singleton
  - one of them is non-existent 

If these constraints are met, then the tensors can be *broadcasted*<sup id="jakevdpbcast">[5](#fn5)</sup> for performing the requested operation.



---------------------------------

<b id="fn1">1. https://numpy.org/doc/stable/user/basics.broadcasting.html  </b> [:arrow_heading_up:](#numpybroadcastingdoc) 

<b id="fn2">2. https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics  </b> [:arrow_heading_up:](#pytorchbroadcastingdoc) 

<b id="fn3">3. https://en.wikipedia.org/wiki/Hadamard_product_(matrices) </b> [:arrow_heading_up:](#hadamardproductwiki) 

<b id="fn4">4. http://jalammar.github.io/visual-numpy/ </b> [:arrow_heading_up:](#visualnumpy) 

<b id="fn5">5. https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html </b> [:arrow_heading_up:](#jakevdpbcast) 