import time

import scipy.sparse
import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split


def select_n_users(X, n):
    users = np.random.randint(X.shape[0], size=n)
    Y = X[users, :].tocoo()
    return scipy.sparse.csr_matrix(
        (Y.data, (users[Y.row], Y.col)),
        shape=X.shape)


print('Loading data...')
X = scipy.sparse.load_npz('lastfm.npz')
X_val = select_n_users(X, 1000)

print('Creating model...')
model = LightFM(32,
                learning_schedule='adadelta',
                loss='warp',
                rho=0.95,
                epsilon=1e-6,
                max_sampled=1000)

print('Training...')
for i in range(5):
    t1 = time.time()
    model.fit_partial(X, num_threads=8)
    t2 = time.time()
    print(f'Epoch time: {(t2 - t1)/60:.2f}min', end=" ")
    print(f'precision@5: {np.mean(precision_at_k(model, X_val, k=5)):.3f}')
