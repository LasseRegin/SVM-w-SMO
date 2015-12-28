# SVM

Simple implementation of a Support Vector Machine using the Sequential Minimal Optimization (SMO) algorithm for training.

## Supported python versions:
* Python 2.7
* Python 3.4

## Python package dependencies
* Numpy        (http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)

# Documentation

Setup model (following parameters are default)

```python

from SVM import SVM
model = SVM(max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001)
```

Train model

```python
model.fit(X, y)
```

Predict new observations

```python
y_hat = model.predict(X_test)
```
