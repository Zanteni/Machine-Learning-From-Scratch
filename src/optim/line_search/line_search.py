"""
line_search/line_search.py
Algorithmes de recherche de pas.
"""
import numpy as np

def armijo_backtracking(loss_fn, theta: np.ndarray, d: np.ndarray, g: np.ndarray, alpha_0: float = 1.0, rho: float = 0.5, c: float = 1e-4) -> float:
    # TODO: Réduire alpha par alpha * rho tant que f(θ + alpha*d) > f(θ) + c * alpha * (g.T @ d)
    slope = (d.T@g).item()
    if slope>=-1e-10:
        print(f"Waring {d} is not strick  descent direction (slope = {slope})")
        return alpha_0
    
    alpha = alpha_0
    f_alpha = loss_fn(theta)
    while loss_fn(theta+alpha*d)>f_alpha+c*alpha*slope:
        alpha*=rho
        if alpha<1e-10:
            break
    return alpha


def goldstein_line_search(loss_fn, theta: np.ndarray, d: np.ndarray, g: np.ndarray, alpha_0: float = 1.0) -> float:
    # TODO: Encadrement du pas respectant les deux bornes de Goldstein.
    slope = (d.T@g).item()
    if slope>=-1e-10:
        print(f"Waring {d} is not strick  descent direction (slope = {slope})")
        return alpha_0
    alpha = alpha_0
    c = 1e-4
    a = 0.0
    b = np.inf
    f_theta = loss_fn(theta)
    for _ in range(100):
        f_next = loss_fn(theta+alpha*d)
        if f_next>f_theta+c*alpha*slope:#large step
            b = alpha
            alpha = 0.5*(a+b)
        elif f_next<f_theta +(1-c)*alpha*slope: # inf dound of Goldstein is not resoected small  step
            a = alpha
            if b==np.inf:
                alpha*=2
            else:
                alpha=0.5*(a+b)
        else:
            break
    return alpha



    

def exact_line_search_quadratic(Q: np.ndarray, g: np.ndarray, d: np.ndarray) -> float:
    # TODO: Pas optimal exact analytique : alpha = - (g.T @ d) / (d.T @ Q @ d)
    alpha = -(g.T@d)/(d.T@Q@d)
    return alpha.item()