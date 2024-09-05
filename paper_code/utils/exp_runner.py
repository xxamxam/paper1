from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

from .timer import timer
from ..altorithm import gamma_for_budget
from ..simple_example.exact_solver import gamma_exact

def compute_for_budgets(c_x, x_s, v_s, L_s, max_budgets, budget_init, cases, exact = False, norm = 1, iterations = 1, n_jobs = 5):
    if exact:
        df = delayed(timer(gamma_exact))
    else:
        df = delayed(timer(gamma_for_budget))
    def _get_rez_():
        if exact:   
            if budget_init is not None:
                iter = zip(max_budgets, budget_init)
            else:
                iter = max_budgets          
            return Parallel(n_jobs=10)(df(bb, c_x, v_s, cases) \
                         for bb in iter)
        return Parallel(n_jobs=n_jobs)(df(budget, c_x, x_s, v_s, L_s, norm = norm) \
                                for budget in max_budgets)
         
    b_s = []
    gammas = []
    times = []

    if iterations == 1:    
        rez = _get_rez_() 

        for b, gamma, t in rez:
            b_s.append(b)
            gammas.append(gamma)
            times.append(t)
        return np.array(b_s), np.array(gammas), np.array(times)
    else:
        # just run several times 
        for _ in tqdm(range(iterations)):
            rez = _get_rez_() 

            tmp_t = []
            for _, _, t in rez:
                tmp_t.append(t)
            times.append(tmp_t)

        return None, None, np.array(times)
    