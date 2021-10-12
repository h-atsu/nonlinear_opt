import numpy as np


def armijo_condtion(f, df, xk, dk, c1, alpha):
    """Parameters
    ----------
    f : callable
        Function to be minimized.
    df : callable
        Drivative of object function.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    c1 : float, optional
        Value to control stopping criterion.
    alpha : float
        Current step size
    """
    flag_armijo: bool = (f(xk + alpha * dk) <= f(xk) + c1 * alpha * (df(xk) @ dk))
    return flag_armijo


def wolf_condtion(f, df, xk, dk, c1, c2, alpha):
    """Parameters
    ----------
    f : callable
        Function to be minimized.
    df : callable
        Drivative of object function.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    c1 : float, optional
        Value to control stopping criterion (armijo condition).
    c2 :
        Value to control stopping criterion (curvature condition).
    alpha : float
        Current step size
    """
    assert 0 < c1 < c2 < 1
    flag_armijo: bool = armijo_condtion(f, df, xk, dk, c1, alpha)
    flag_curvature: bool = (df(xk + alpha * dk) @ dk >= c2 * df(xk) @ dk)
    return flag_armijo and flag_curvature


def strong_wolf_condtion(f, df, xk, dk, c1, c2, alpha):
    """Parameters
    ----------
    f : callable
        Function to be minimized.
    df : callable
        Drivative of object function.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    c1 : float, optional
        Value to control stopping criterion (armijo condition).
    c2 :
        Value to control stopping criterion (curvature condition).
    alpha : float
        Current step size
    """
    assert 0 < c1 < c2 < 1
    flag_armijo: bool = f(xk + alpha * dk) <= c1 * alpha * (df(xk) @ dk)
    flag_curvature_strong: bool = abs(df(xk + alpha * dk) @ dk) >= abs(c2 * df(xk) @ dk)
    return flag_armijo and flag_curvature_strong


def goldstein_condtion(f, df, xk, dk, c, alpha):
    """Parameters
    ----------
    f : callable
        Function to be minimized.
    df : callable
        Drivative of object function.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    c : float, optional
        Value to control stopping criterion.
    alpha : float
        Current step size
    """
    flag_goldstein = (f(xk) + (1 - c) * alpha * df(xk) @ dk <= f(xk + alpha * dk) <= f(xk) + c * alpha * df(xk) @ dk)
    return flag_goldstein


def backtracking(f, df, xk, dk, rho=4 / 5, alpha0=1):
    """Parameters
    ----------
    f : callable
        Function to be minimized.
    df : callable
        Drivative of object function.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    rho : float, optional
        shrinkage rate
    alpha0 : float
        initial value of alpha

    return 
    ------
    opmized step size alpha

    Notes
    ------
    only support wolf method yet
    """
    alpha = alpha0
    iter_num = 0
    while not armijo_condtion(f, df, xk, dk, c1=0.1, alpha=alpha):

        alpha *= rho
        iter_num += 1
        assert iter_num < 100

    return alpha


if __name__ == '__main__':
    x = np.array([0])
    print(type(x))
    print(armijo_condtion(lambda x: (x - 1) * x, lambda x: 2 * x - 1, np.array([0]), np.array([1]), 0, np.array([1.1])))
