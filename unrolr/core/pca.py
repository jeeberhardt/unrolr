#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2020
# Unrolr
#
# Principal Component Analysis
# Author: Jérôme Eberhardt <qksoneo@gmail.com>
#
# License: MIT


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from scipy import linalg

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2020, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


class PCA:

    def __init__(self, n_components=None):
        """Create DihedralPCa object

        Args:
            n_components (int, None): Number of components to keep. if n_components is not set all components are kept.

        """
        self._n_components = n_components
        self.components = None
        self.singular_values = None

    def fit_transform(self, X):
        """Fit the model with X and apply the dimensionality reduction on X.

        Args:
            X (ndarray): array-like, shape (n_samples, n_features)

        Returns:
            ndarray: final embedding (n_samples, n_components)

        """
        # Centering the data
        X -= np.mean(X, axis=0)  
        # Compute covariance matrix
        cov = np.cov(X, rowvar=False)
        # PCA!!!
        singular_values , components = linalg.eigh(cov)

        # Sort by singular values
        idx = np.argsort(singular_values)[::-1]
        self.components = components[:, idx].T
        self.singular_values = singular_values[idx]

        if self._n_components is None:
            embedding = np.dot(X, self.components)
        else:
            embedding = np.dot(X, self.components[:int(self._n_components)].T)

        return embedding
