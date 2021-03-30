# git fetch trac u/yzh/hybrid_backend, git merge FETCH_HEAD then make build
# where HEAD is at 71ae94d57b (trac/u/yzh/hybrid_backend, t/18735/hybrid_backend),
# and trac is git://trac.sagemath.org/sage.git (fetch)
# https://github.com/mkoeppe/cutgeneratingfunctionology/commits/param  [branch param, commit 2ddc73711]


#import cutgeneratingfunctionology.igp as igp; from cutgeneratingfunctionology.igp import *
## import importlib; importlib.reload(igp); from cutgeneratingfunctionology.igp import *


# **********************************************************
#  LP for portfolio optimization: max (mu * reword - risk),
#  with risk measure = mean absolute deviation from the mean
# **********************************************************

def setup_portfolio_lp(hist_data, mu, solver=None):
    n = len(hist_data)
    T = len(hist_data[0])
    if mu == +Infinity or mu == -Infinity or not hasattr(mu, 'parent'):
        base_ring = None
    else:
        base_ring = mu.parent()
    lp = MixedIntegerLinearProgram(solver=solver, maximization=True, base_ring=base_ring)
    x = lp.new_variable(nonnegative=True)
    lp.add_constraint(sum(x[i] for i in range(n)) == 1)
    y = lp.new_variable(nonnegative=True)
    exp_return = [sum(r) / T for r in hist_data]
    for t in range(T):
        dev = sum(x[j] * (hist_data[j][t] - exp_return[j]) for j in range(n))
        lp.add_constraint(dev <= y[t])
        lp.add_constraint(dev >= -y[t])
    reward = sum([x[j] * exp_return[j] for j in range(n)])
    risk = sum([y[t] for t in range(T)]) / T
    if mu == +Infinity:
        obj = reward
    elif mu == -Infinity:
        obj = - reward
    else:
        obj = mu * reward - risk
    lp.set_objective(obj)
    return lp

# **********************************************************
#         Solution via Parametric Simplex Method
# **********************************************************

def find_mu_intervals_and_portfolios_by_parametric_simplex_method(hist_data, verbose=False):
    lp_reward_only = setup_portfolio_lp(hist_data, +Infinity, solver="InteractiveLP")
    max_reward = lp_reward_only.solve()
    basic_indices = lp_reward_only.get_backend().dictionary().basic_indices()
    K.<mu> = ParametricRealField([10]) # var_value [10] can be replaced by any real number.
    lp = setup_portfolio_lp(hist_data, mu)
    d = lp.get_backend().interactive_lp_problem().standard_form().revised_dictionary(*basic_indices) # .dictionary(..) works too.
    intervals_and_weights = {}
    upper = +Infinity
    while upper > 0:
        solution = d.basic_solution()
        weights = tuple(QQ(solution[i].val()) for i in range(len(hist_data)))
        obj_coef_poly = [K._sym_field(p.sym()).numerator() for p in d.objective_coefficients()]
        mu_coef_become_positive = [(- l.constant_coefficient() / l.lc()) if (l.degree() > 0) else -Infinity for l in obj_coef_poly]
        lower = max([mu_value for mu_value in mu_coef_become_positive if mu_value < upper])
        intervals_and_weights[(lower, upper)] = weights
        if verbose:
            view(d)
            logging.info("Interval=(%s, %s), portfolio=%s" % (lower, upper, weights))
        index_lower = mu_coef_become_positive.index(lower)
        entering_variable = d.nonbasic_variables()[index_lower]
        d.enter(entering_variable)
        possible = d.possible_leaving()
        if possible: #if not, then lp unbounded, and lower is already -Infinity.
            d.leave(min(possible))
            d.update()
        upper = lower
    return intervals_and_weights

# **********************************************************
#           Solution via SPAM
# **********************************************************

def solve_for_optimal_portfolio(K, lp):
    opt_value = lp.solve()
    # workaround of lp.get_values(x_var) as we didn't record the lp variables.
    n_var = lp.number_of_variables() - (lp.number_of_constraints() - 1) / 2
    h = lp.get_backend()
    weights = tuple(h.get_variable_value(i).val() for i in range(n_var))
    return weights

def find_optimal_portfolios_complex(hist_data):
    bddbsa = BasicSemialgebraicSet_veronese(poly_ring=PolynomialRing(QQ, ['mu'], 1))
    bddbsa.add_linear_constraint((-1,), 0, operator.lt)
    complex = SemialgebraicComplex(setup_portfolio_lp, ['mu'], find_region_type=solve_for_optimal_portfolio, bddbsa=bddbsa, hist_data=hist_data, solver=("GLPK", "InteractiveLP"))
    complex.bfs_completion(var_value=[10]) # var_value [10] can be replaced by any real number.
    for c in complex.components:
        c.bsa.tighten_upstairs_by_mccormick(max_iter=1)
        c.interval = (c.bsa._bounds[0])
    return complex

# **********************************************************
#           Output the solutions
# **********************************************************
    
def output_portfolios_sorted_by_mu(complex, reverse=False, digits=5):
    # Result may contain overlapping intervals.
    # This function is for debugging purpose only.
    components = copy(complex.components)
    components.sort(key=lambda c: c.var_value[0], reverse=reverse)
    for i in range(len(components)):
        c = components[i]
        (a, b) = c.interval
        if not digits:
            print("Interval %s = (%s, %s),  portfolio=%s,  mu=%s" % (i, a, b, c.region_type, c.var_value[0]))
        else:
            print("Interval %s = (%s, %s),  portfolio=%s,  mu=%s" % (i, a.n(digits=digits), b.n(digits=digits), tuple(w.n(digits=digits) for w in c.region_type), c.var_value[0].n(digits=digits)))

def mu_intervals_and_portfolios_from_complex(complex):
    intervals_and_weights = {}
    for c in complex.components:
        (lower, upper) = c.interval
        intervals_and_weights[(lower, upper)] = c.region_type
    return intervals_and_weights

def combine_intervals_with_same_portfolio(intervals_and_weights):
    sorted_list = list(intervals_and_weights.items())
    sorted_list.sort(key = lambda x: x[0][0])
    combined_list = []
    cc = None
    bb = None
    for (a, b), c in sorted_list:
        assert(bb is None or bb >= a)
        if c == cc:
            if b > bb:
                bb = b
        else:
            if not cc is None:
                combined_list.append(((aa, bb), cc))
            aa = a
            bb = b
            cc = c
    combined_list.append(((aa, bb), cc))
    return combined_list
    
def output_intervals_and_weights(intervals_and_weights, combined=False, digits=None):
    if not combined:
        intervals_weights_list = list(intervals_and_weights.items())
    else:
        intervals_weights_list = combine_intervals_with_same_portfolio(intervals_and_weights)
    for i in range(len(intervals_weights_list)): 
        ((a,b), c) = intervals_weights_list[i]
        if not digits:
            print("Interval %i = (%s, %s),  portfolio=%s" % (i, a, b, c))
        else:
            print("Interval %i = (%s, %s),  portfolio=%s" % (i, a.n(digits=digits), b.n(digits=digits), tuple(w.n(digits=digits) for w in c)))
