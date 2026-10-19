# Lab 5: Linear Classifier & Multiclass SVM

In this lab you will implement a complete multiclass linear classifier trained
with Stochastic Gradient Descent on CIFAR-10, including the vectorized hinge
loss, its analytical gradient, minibatch sampling and a grid search over
hyperparameters.

## 1. Setup

- Python 3.8 or higher with numpy. No other library is needed.
- The four scripts in this package: main.py, data_utils.py, classifiers.py,
  hyperparameter_tuning.py.

## 2. Getting the CIFAR-10 data

main.py reads the CIFAR-10 python version from a folder named
`cifar-10-batches-py` sitting next to the scripts. Get it like this:

1. Download `cifar-10-python.tar.gz` (about 163 MB) from
   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
2. Extract it. This produces one folder `cifar-10-batches-py` containing
   `data_batch_1` ... `data_batch_5`, `test_batch` and `batches.meta`.
3. Move that folder into `lab_week05/`, next to main.py:

```text
lab_week05/
├── main.py
├── data_utils.py
├── classifiers.py
├── hyperparameter_tuning.py
├── README.md
└── cifar-10-batches-py/
    ├── data_batch_1 ... data_batch_5
    ├── test_batch
    └── batches.meta
```

The folder name must stay exactly `cifar-10-batches-py` and it must sit next
to main.py, otherwise the loader cannot find it.

## 3. The five TODOs

| # | File | Function | What you implement |
|---|------|----------|--------------------|
| 1 | data_utils.py | preprocess_with_bias_trick | append a constant 1 to every sample so the bias folds into W |
| 2 | classifiers.py | svm_loss_vectorized | the score matrix for the whole batch in one matrix multiplication |
| 3 | classifiers.py | svm_loss_vectorized | the gradient without loops, using a binary mask over positive margins |
| 4 | classifiers.py | LinearClassifier.train | minibatch sampling with np.random.choice |
| 5 | hyperparameter_tuning.py | hyperparameter_tuning | the grid search loop over learning rates and regularization strengths |

svm_loss_naive is given to you in full as the reference implementation. Your
vectorized version must return the same loss and gradient.

## 4. Running

From inside `lab_week05/` run:

```bash
python main.py
```

A correct run shows, among other prints:

- loss difference between the naive and the vectorized version below 1e-9
- gradient difference below 1e-9
- the training loss decreasing over the 1500 iterations
- validation accuracies from the grid search, with the best one around 0.25
  to 0.40 for this setup

The whole run takes a few minutes up to half an hour depending on your
machine, mostly spent in the grid search.

## 5. Submission requirements

Submissions close at the due date shown on the Canvas assignment page. Late
uploads are not accepted after the assignment locks.

Your submission is one zip file containing exactly these items:

1. The four completed .py files: main.py, data_utils.py, classifiers.py,
   hyperparameter_tuning.py.
2. A plain text file named output.txt holding the complete console output of
   one run of `python main.py` from start to finish. Save it for example with
   `python main.py > output.txt 2>&1`, or copy the full terminal text into a
   .txt file.

No report is required for this lab.
