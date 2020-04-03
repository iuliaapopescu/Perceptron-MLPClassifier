# Training a perceptron on 3d points

- Step 1: Load data

```
X_train = np.loadtxt('data/3d-points/x_train.txt')

y_train = np.loadtxt('data/3d-points/y_train.txt', 'int') 

X_test = np.loadtxt('data/3d-points/x_test.txt')

y_test = np.loadtxt('data/3d-points/y_test.txt', 'int') 
```

- Step 2: Normalize data

```
scaler = preprocessing.StandardScaler()

scaler.fit(X_train)

X_train_sc = scaler.transform(X_train)

X_test_sc = scaler.transform(X_test)
```

- Step 3: Define the perceptron model
```
perceptron_model = Perceptron(eta0=0.1, tol=1e-5)
```

- Step 4: Train the model
```
perceptron_model.fit(X_train_sc, y_train)
```

- Step 5: Get the accuracy on the test set
```
accuracy = perceptron_model.score(X_test_sc, y_test)
```

- Step 6: Get the number of iterations until convergence
```
no_iter = perceptron_model.n_iter_
```

- Step 7: Get constants in decision function
```
bias = perceptron_model.intercept_
```

- Step 8: Get weights assigned to the features
```
Weights = perceptron_model.coef_.reshape(3, 1)
```

- Step 9: Plot result
```
plot3d_data_and_decision_function(X_train_sc, y_train, Weights, bias)
```

![Train set and the separating hyperplane](https://github.com/iuliaapopescu/Perceptron-MLPClassifier/blob/master/Figure_1.png)

# Training a neural network using the MLPClassifier model from sklearn library

## Train a MLPClassifier model and test it using the following configurations:

a. Activation function=‘tanh’, hidden_layer_sizes=(1), learning_rate_init=0.01, momentum=0, max_iter=200 (default)

b. Activation function=‘tanh’, hidden_layer_sizes=(10), learning_rate_init=0.01, momentum=0, max_iter=200 (default)

c. Activation function=‘tanh’, hidden_layer_sizes=(10), learning_rate_init=0.00001, momentum=0, max_iter=200 (default)

d. Activation function=‘tanh’, hidden_layer_sizes=(10), learning_rate_init=10, momentum=0, max_iter=200 (default)

e. Activation function=‘tanh’, hidden_layer_sizes=(10), learning_rate_init=0.01, momentum=0, max_iter=20

f. Activation function=‘tanh’, hidden_layer_sizes=(10, 10), learning_rate_init=0.01, momentum=0, max_iter=2000

g. Activation function=‘relu’, hidden_layer_sizes=(10, 10), learning_rate_init=0.01, momentum=0, max_iter=2000

h. Activation function=‘relu’, hidden_layer_sizes=(100, 100), learning_rate_init=0.01, momentum=0, max_iter=2000

i. Activation function=‘relu’, hidden_layer_sizes=(100, 100), learning_rate_init=0.9, momentum=0, max_iter=2000

j. Activation function=‘relu’, hidden_layer_sizes=(100, 100), learning_rate_init=0.9, momentum=0, max_iter=2000, alpha=0.005

- a)
`clf = MLPClassifier(hidden_layer_sizes=(1,), 
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`

- b)
`clf = MLPClassifier(hidden_layer_sizes=(10,), 
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`
- c)
`clf = MLPClassifier(hidden_layer_sizes=(10,), 
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.00001,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`
- d)
`clf = MLPClassifier(hidden_layer_sizes=(10,), 
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=10,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`
- e)
`clf = MLPClassifier(hidden_layer_sizes=(10,), 
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0,
                    max_iter=20)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`
- f)
`clf = MLPClassifier(hidden_layer_sizes=(10, 10), 
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0,
                    max_iter=2000)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`
- g)
`clf = MLPClassifier(hidden_layer_sizes=(10, 10), 
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0,
                    max_iter=2000)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`
- h)
`clf = MLPClassifier(hidden_layer_sizes=(100, 100), 
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`
- i)
`clf = MLPClassifier(hidden_layer_sizes=(100, 100), 
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0.9,
                    max_iter=2000)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`
- j)
`clf = MLPClassifier(hidden_layer_sizes=(100, 100), 
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0.9,
                    max_iter=2000,
                    alpha=0.005)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
`

## Final results

 | Idx |   Accuracy for train set |   Accuracy for test set |   No. of iterations until convergence |   No. of maximum iterations |
 | :-: |        :--------:        |         :--------:      |               :--------:              |           :--------:        |
 |   1 |                    0.226 |                   0.19  |                                   200 |                         200 |
 |   2 |                    0.935 |                   0.848 |                                   200 |                         200 |
 |   3 |                    0.151 |                   0.162 |                                   200 |                         200 |    
 |   4 |                    0.955 |                   0.784 |                                   105 |                         200 |
 |   5 |                    0.548 |                   0.51  |                                    20 |                          20 |
 |   6 |                    0.599 |                   0.56  |                                    37 |                        2000 |
 |   7 |                    0.993 |                   0.874 |                                   866 |                        2000 |
 |   8 |                    0.994 |                   0.862 |                                   200 |                        2000 |
 |   9 |                        1 |                   0.882 |                                   110 |                        2000 |     
 |  10 |                        1 |                   0.896 |                                   108 |                        2000 |
