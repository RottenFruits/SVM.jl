SVM.jl
======

# THIS PACKAGE IS FORKED FROM https://github.com/JuliaStats/SVM.jl, AND ADAPTING JULIA 1.1.1.



# SVMs in Julia

Native Julia implementations of standard SVM algorithms.
Currently, there are textbook style implementations of
two popular linear SVM algorithms:

* Pegasos (Shalev-Schwartz et al., 2007)
* Dual Coordinate Descent (Hsieh et al., 2008)

The `svm` function is a wrapper for `pegasos`, but it is
possible to call `cddual` explicitly. See the source code
for the hyperparameters of the `cddual` function.

# Usage

The demo below shows how SVMs work:

```julia
# To show how SVMs work, we'll use Fisher's iris data set
using SVM
using RDatasets
using Random

# We'll learn to separate setosa from other species
iris = dataset("datasets", "iris")

# SVM format expects observations in columns and features in rows
#X = array(iris[:, 1:4])'
p, n = size(X)

X = convert(Array, iris[:, 1:4])'

# SVM format expects positive and negative examples to +1/-1
y = [species == "setosa" ? 1.0 : -1.0 for species in iris[:Species]]

# Select a subset of the data for training, test on the rest.
train = bitrand(n)

# We'll fit a model with all of the default parameters
model = svm(X[:,train], y[train])

# And now evaluate that model on the testset
accuracy = sum(predict(model, X[:, .!train]) .== Y[.!train])/sum(.!train)
```

You may specify non-default values for the various parameters:

```julia
# The algorithm processes minibatches of data of size k
model = svm(X, y, k = 150)

# Weight regularization is controlled by lambda
model = svm(X, y, lambda = 0.1)

# The algorithm performs T iterations
model = svm(X, y, T = 1000)
```

