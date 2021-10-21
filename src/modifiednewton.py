import numpy as np
from line_search import backtracking


class Optimizer:
    def __init__(self, f, df, ddf, alpha=1e-3, eps=1e-6, path=None, rho=10, mu=0.1):
        self.f = f
        self.df = df
        self.ddf = ddf
        self.alpha = alpha
        self.eps = eps
        self.path = path
        self.opt_sol = None
        self.opt_val = None
        self.rho = rho
        self.mu = mu

    def optimize(self, init_sol):
        x = init_sol
        path = []
        path.append(x)
        grad = self.df(x)
        hesse = self.ddf(x)
        while (grad**2).sum() > self.eps**2:
            grad = self.df(x)
            hesse = self.ddf(x)
            hesse_bar = hesse
            mu = self.mu
            while True:
                try:
                    L = np.linalg.cholesky(hesse_bar)
                except np.linalg.LinAlgError as err:
                    mu *= self.rho
                    hesse_bar = hesse + np.eye(len(hesse_bar)) * mu
                    continue
                else:
                    break
            d = np.linalg.solve(L.T, np.linalg.solve(L, grad))
            alpha = backtracking(self.f, self.df, x, -d)
            x = x - alpha * d
            path.append(x)
        self.path = np.array(path)
        self.opt_sol = x
        self.opt_val = self.f(x)


if __name__ == '__main__':
    n_size = 10
    A = np.random.random((n_size, n_size))
    A = A.T @ A
    x0 = np.random.random(n_size)
    print(x0)

    optimizer = Optimizer(lambda x: x @ A @ x, lambda x: A @ x, lambda x: A)
    optimizer.optimize(x0)
    opt_val = optimizer.opt_val
    opt_sol = optimizer.opt_sol
    print(opt_val, opt_sol)
