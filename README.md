.. image:: http://img.shields.io/badge/arXiv-1806.10206-orange.svg?style=flat :target: http://arxiv.org/abs/1806.10206

Deep Feature Factorization (DFF)
--------------------------------

**DFF** is the application of non-negative matrix faxtorization (NMF) to the ReLU feature activations of a deep neural network. In the case of CNNs trained on images, the resulting factors decompose an image batch into semenatic parts with a high degree of invariance to complex transformations.

This implementation relies on Pytorch and includes a GPU implementation of NMF with multiplicative updates (Lee and Seung, 2001).

 Edo Collins, Radhakrishna Achanta and Sabine Süsstrunk. *Deep Feature Factorization for Concept Discovery*.  European Conference on Computer Vision (ECCV) 2018.
