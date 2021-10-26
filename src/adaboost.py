from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import random
from line_search import backtracking

random.seed(42)


class Adaboost:
    def __init__(self, num_basis=100, num_sample=30, num_iteration=100):
        self.num_basis = num_basis
        self.num_sample = num_sample
        self.num_target = None
        self.stumps = None
        self.w = np.ones(self.num_basis)
        self.num_iteration = num_iteration
        self.path = []
        self.fun_his = []

    def one_hot_encoder(self, y):
        ret = np.zeros((y.size, self.num_target))
        ret[np.arange(y.size), y] = 1.0
        return ret

    def grad_i(self, ith, X, y):
        """
        numerically compute ith gradient
        """
        f_old = self.loss(X, y)
        w_old = self.w
        eps = 1e-3
        dw = np.zeros_like(w_old)
        dw[ith] += eps
        w_new = self.w + dw
        self.w = w_new
        f_new = self.loss(X, y)
        self.w = w_old
        df_dw_i = (f_new - f_old) / eps
        grad = np.zeros_like(self.w)
        grad[ith] += df_dw_i
        return grad

    def make_decision_stump(self, X, y):
        """
        return list of decision stump classifier 
        """
        len_data = len(X)
        stumps = []
        for i in range(self.num_basis):
            choiced_idx = np.random.choice(range(len_data), self.num_sample)
            X_small = X[choiced_idx]
            y_small = y[choiced_idx]
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
            tree.fit(X_small, y_small)
            stumps.append(tree)
        return stumps

    def f(self, w, X, y):
        w_old = self.w
        self.w = w
        loss = self.loss(X, y)
        self.w = w_old
        return loss

    def df(self, w, X, y, ith):
        w_old = self.w
        self.w = w
        grad = self.grad_i(ith, X, y)
        self.w = w_old
        return grad

    def fit(self, X, y):
        self.num_target = len(np.unique(y))
        self.w /= (self.num_target * self.num_basis)
        stumps = self.make_decision_stump(X, y)
        self.stumps = stumps
        for i in range(self.num_iteration):
            ith = np.random.choice(range(len(self.w)))
            grad = self.grad_i(ith, X, y)
            def f(w): return self.f(w, X=X, y=y)
            def ddf(w): return self.df(w, X=X, y=y, ith=ith)
            w0 = self.w
            alpha = backtracking(f, ddf, w0, -grad)
            self.w = w0 - alpha * grad
            self.path.append((self.w))
            self.fun_his.append(self.loss(X, y))

    def prob(self, X):
        prob_y = np.zeros((len(X), self.num_target))
        for i, fun in enumerate(self.stumps):
            prob_y += self.w[i] * self.one_hot_encoder(fun.predict(X))
        return prob_y

    def loss(self, X, y):
        """
        return sum_{k=1}^{N} exp(- t_k * log(y_k))
        """
        N = len(X)
        true_y = self.one_hot_encoder(y)
        pred_y = self.prob(X)
        cross_entropy = -np.log((true_y * pred_y).sum(axis=1))
        return np.exp(cross_entropy).sum() / N

    def predict(self, X):
        y_pred = np.zeros((len(X), self.num_target))
        for i, fun in enumerate(self.stumps):
            y_pred += self.w[i] * self.one_hot_encoder(fun.predict(X))
        return np.argmax(y_pred, axis=1)


        ####### validation ################
from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    test_proportion = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=1)
    clf = Adaboost()
    clf.fit(X_train, y_train)
    print(clf.loss(X_train, y_train))
    print(clf.loss(X_test, y_test))
