"""Self-contained numerical replication of the synthetic studies supporting Appendices E and F.
All inputs are synthetic. Run with Python 3.12, NumPy 2.3.5,
and SciPy 1.17.0. Output is JSON printed to standard output.
"""
import json
import math
import numpy as np
from scipy.optimize import brentq, linprog, minimize_scalar, minimize
from scipy.special import betainc
from scipy.stats import beta as beta_dist
from scipy.stats import beta
from scipy.integrate import quad


BASE = {'A': 1.3, 'k': 0.8, 'b': 0.2, 'a': 0.5, 'c': 1.2, 'lambda': 0.5, 'v'
    : 0.5, 'd': 3.0, 'gamma': 0.5, 'x_max': 0.35, 'alpha': 5.0, 'beta': 4.0,
    'theta_low': 0.4, 'theta_high': 0.9, 'rho': 0.35, 'q_min': 0.35,
    'requested_mw': 100.0, 'max_acceleration_months': 24.0,
    'value_scale_musd': 100.0, 'seed': 20260907, 'rent_weight_chi': 1.0,
    'unserved_penalty': 1000000.0, 'solver_feasibility_tolerance': 1e-09}

def primitives(p: dict) -> tuple[float, float]:
    d_eff = p['gamma'] + p['c'] - p['c'] ** 2 / (p['k'] + p['c']) - p[
        'lambda'] ** 2 / p['a']
    k_eff = p['v'] + p['c'] * p['A'] / (p['k'] + p['c']) + p['lambda'] * p[
        'b'] / p['a']
    return (d_eff, k_eff)

def headroom_cdf(r: float | np.ndarray, p: dict) -> float | np.ndarray:
    z = np.clip(np.asarray(r) / p['x_max'], 0.0, 1.0)
    ans = betainc(p['alpha'], p['beta'], z)
    return float(ans) if np.ndim(r) == 0 else ans

def headroom_pdf(r: float | np.ndarray, p: dict) -> float | np.ndarray:
    z = np.clip(np.asarray(r) / p['x_max'], 1e-12, 1 - 1e-12)
    ans = beta_dist.pdf(z, p['alpha'], p['beta']) / p['x_max']
    return float(ans) if np.ndim(r) == 0 else ans

def expected_shortfall(r: float | np.ndarray, p: dict
    ) -> float | np.ndarray:
    arr = np.asarray(r, dtype=float)
    z = np.clip(arr / p['x_max'], 0.0, 1.0)
    truncated_first = p['x_max'] * p['alpha'] / (p['alpha'] + p['beta']
        ) * betainc(p['alpha'] + 1, p['beta'], z)
    ans = arr * betainc(p['alpha'], p['beta'], z) - truncated_first
    ans = np.where(arr <= 0, 0.0, ans)
    mean_x = p['x_max'] * p['alpha'] / (p['alpha'] + p['beta'])
    ans = np.where(arr >= p['x_max'], arr - mean_x, ans)
    return float(ans) if np.ndim(r) == 0 else ans

def allocation(virtual_type: float, p: dict) -> np.ndarray:
    d_eff, k_eff = primitives(p)

    def residual(r: float) -> float:
        return d_eff * r + (p['v'] + p['d']) * headroom_cdf(r, p) - (k_eff -
            virtual_type)
    if residual(0.0) >= 0:
        r = 0.0
    elif residual(p['x_max']) <= 0:
        r = p['x_max']
    else:
        r = brentq(residual, 0.0, p['x_max'], xtol=1e-13, rtol=1e-13)
    q = (p['A'] + p['c'] * r) / (p['k'] + p['c'])
    s = (p['b'] + p['lambda'] * r) / p['a']
    qmin = p.get('q_min', 0.35)
    if qmin <= q <= 1 and 0 <= s <= 1 and (r <= p['rho'] * q):
        return np.array([q, s, r], dtype=float)

    def conditional(rr):
        qq = np.clip((p['A'] + p['c'] * rr) / (p['k'] + p['c']), max(qmin,
            rr / p['rho']), 1.0)
        ss = np.clip((p['b'] + p['lambda'] * rr) / p['a'], 0.0, 1.0)
        return np.array([qq, ss, rr])
    upper = min(p['x_max'], p['rho'])
    objective = lambda rr: -social_surplus(conditional(rr), virtual_type, p)
    res = minimize_scalar(objective, bounds=(0.0, upper), method='bounded',
        options={'xatol': 1e-13})
    rr = min([0.0, upper, res.x], key=objective)
    return conditional(rr)

def social_surplus(x: np.ndarray, theta: float, p: dict) -> float:
    q, s, r = x
    return float(p['A'] * q - 0.5 * p['k'] * q ** 2 + p['b'] * s - 0.5 * p[
        'a'] * s ** 2 - 0.5 * p['c'] * (q - r) ** 2 + p['lambda'] * s * r +
        p['v'] * r - (p['v'] + p['d']) * expected_shortfall(r, p) - theta *
        r - 0.5 * p['gamma'] * r ** 2)

def applicant_common_value(x: np.ndarray, p: dict) -> float:
    q, s, r = x
    return float(p['A'] * q - 0.5 * p['k'] * q ** 2 + p['b'] * s - 0.5 * p[
        'gamma'] * r ** 2)

def reverse_rent(theta: np.ndarray, r: np.ndarray) -> np.ndarray:
    u = np.zeros_like(theta)
    for i in range(len(theta) - 2, -1, -1):
        u[i] = u[i + 1] + 0.5 * (r[i] + r[i + 1]) * (theta[i + 1] - theta[i]
            )
    return u

def mechanism_grid(n: int, p: dict) -> dict:
    theta = np.linspace(p['theta_low'], p['theta_high'], n)
    psi = theta + p['rent_weight_chi'] * (theta - p['theta_low'])
    first_best = np.stack([allocation(t, p) for t in theta])
    second_best = np.stack([allocation(z, p) for z in psi])
    welfare_fb = np.array([social_surplus(x, t, p) for x, t in zip(
        first_best, theta)])
    welfare_sb = np.array([social_surplus(x, t, p) for x, t in zip(
        second_best, theta)])
    rent = reverse_rent(theta, second_best[:, 2])
    common = np.array([applicant_common_value(x, p) for x in second_best])
    transfer_to_applicant = rent - common + theta * second_best[:, 2]
    access_charge = -transfer_to_applicant
    return {'theta': theta, 'psi': psi, 'first_best': first_best,
        'second_best': second_best, 'welfare_fb': welfare_fb, 'welfare_sb':
        welfare_sb, 'rent': rent, 'transfer': transfer_to_applicant,
        'charge': access_charge}

def mean_on_uniform(values: np.ndarray, theta: np.ndarray, p: dict
    ) -> float:
    return float(np.trapezoid(values, theta) / (p['theta_high'] - p[
        'theta_low']))

def tier_mechanism(k_tiers: int, theta: np.ndarray, p: dict) -> dict:
    edges = np.linspace(p['theta_low'], p['theta_high'], k_tiers + 1)
    allocations = []
    representative_virtual_types = []
    for j in range(k_tiers):
        midpoint = 0.5 * (edges[j] + edges[j + 1])
        z = midpoint + p['rent_weight_chi'] * (midpoint - p['theta_low'])
        representative_virtual_types.append(z)
        allocations.append(allocation(z, p))
    rows = []
    for t in theta:
        idx = min(np.searchsorted(edges[1:], t, side='right'), k_tiers - 1)
        rows.append(allocations[idx])
    x = np.stack(rows)
    support = p['theta_high'] - p['theta_low']
    exact_welfare = 0.0
    exact_rent = 0.0
    for j, xx in enumerate(allocations):
        lo, hi = (edges[j], edges[j + 1])
        midpoint = 0.5 * (lo + hi)
        weight = (hi - lo) / support
        exact_welfare += weight * social_surplus(xx, midpoint, p)
        exact_rent += xx[2] * 0.5 * ((hi - p['theta_low']) ** 2 - (lo - p[
            'theta_low']) ** 2) / support
    design_objective = exact_welfare - p['rent_weight_chi'] * exact_rent
    return {'k': k_tiers, 'edges': edges, 'representative_virtual_types': np
        .asarray(representative_virtual_types), 'allocations': np.asarray(
        allocations), 'grid_allocations': x, 'welfare': exact_welfare,
        'rent': exact_rent, 'principal': design_objective}

def solve_scheduler(q_mw: float, r_mw: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    n_jobs = 360
    release_weights = np.ones(24)
    release_weights[21:] = 0.05
    release_weights *= 1 + 2 * np.exp(-0.5 * ((hours - 16) / 3) ** 2)
    release_weights /= release_weights.sum()
    release = rng.choice(hours, n_jobs, p=release_weights)
    slack = rng.integers(3, 10, n_jobs)
    deadline = np.minimum(23, release + slack)
    energy = rng.uniform(0.4, 1.4, n_jobs)
    max_rate = rng.uniform(0.35, 0.9, n_jobs)
    for j in range(n_jobs):
        slots = deadline[j] - release[j] + 1
        max_rate[j] = max(max_rate[j], 1.05 * energy[j] / slots)
    fixed = 45 + 3.5 * np.sin(2 * np.pi * (hours - 7) / 24) + 2 * np.exp(-
        0.5 * ((hours - 18) / 3) ** 2)
    price = np.array([30, 28, 27, 26, 27, 30, 35, 42, 48, 50, 48, 45, 42, 40
        , 38, 42, 55, 80, 105, 115, 95, 65, 48, 38], dtype=float)
    event_hours = np.array([17, 18, 19, 20], dtype=int)
    delay_penalty = 25.0
    n_dispatch_variables = n_jobs * 24
    n_variables = n_dispatch_variables + n_jobs
    objective = np.zeros(n_variables)
    bounds = []
    for j in range(n_jobs):
        for t in range(24):
            objective[j * 24 + t] = price[t] + delay_penalty * (t - release[
                j])
            if release[j] <= t <= deadline[j]:
                bounds.append((0.0, max_rate[j]))
            else:
                bounds.append((0.0, 0.0))
    objective[n_dispatch_variables:] = BASE['unserved_penalty']
    bounds.extend([(0.0, None)] * n_jobs)
    a_eq = np.zeros((n_jobs, n_variables))
    for j in range(n_jobs):
        a_eq[j, j * 24:(j + 1) * 24] = 1.0
        a_eq[j, n_dispatch_variables + j] = 1.0

    def run(contract: bool, relief_override: float | None=None) -> dict:
        a_ub = np.zeros((24, n_variables))
        b_ub = np.zeros(24)
        active_relief = r_mw if relief_override is None else relief_override
        for t in range(24):
            a_ub[t, t::24] = 1.0
            a_ub[t, n_dispatch_variables:] = 0.0
            event_cap = (q_mw - active_relief
                         if contract and t in event_hours else q_mw)
            b_ub[t] = event_cap - fixed[t]
        result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=
            energy, bounds=bounds, method='highs', options={
            'primal_feasibility_tolerance': BASE[
            'solver_feasibility_tolerance'], 'dual_feasibility_tolerance':
            BASE['solver_feasibility_tolerance'], 'ipm_optimality_tolerance'
            : BASE['solver_feasibility_tolerance'], 'presolve': True})
        if not result.success:
            raise RuntimeError(result.message)
        x = result.x[:n_dispatch_variables].reshape(n_jobs, 24)
        z = result.x[n_dispatch_variables:]
        flexible = x.sum(axis=0)
        total = fixed + flexible
        weighted_delay = sum((np.dot(hours - release[j], x[j]) for j in
            range(n_jobs))) / energy.sum()
        return {'x': x, 'z': z, 'flexible': flexible, 'total': total,
            'mean_delay_hours': float(weighted_delay),
            'facility_energy_cost': float(np.dot(total, price)),
            'flexible_energy_cost': float(np.dot(flexible, price)),
            'objective': float(result.fun), 'duals': -np.asarray(result.
            ineqlin.marginals), 'capacity_slack': np.asarray(result.ineqlin.
            residual), 'energy_balance_residual': np.asarray(a_eq @ result.x
            - energy), 'status': int(result.status), 'message': result.
            message}
    standard = run(False)
    contract = run(True)
    finite_difference_step = 0.0001
    contract_tighter = run(True, r_mw + finite_difference_step)
    total_energy = float(energy.sum())
    comparator_event_energy_difference = float((standard['total'][
        event_hours] - contract['total'][event_hours]).sum())
    return {'hours': hours, 'n_jobs': n_jobs, 'release': release, 'deadline'
        : deadline, 'energy': energy, 'max_rate': max_rate, 'fixed': fixed,
        'price': price, 'event_hours': event_hours, 'delay_penalty':
        delay_penalty, 'total_job_energy': total_energy, 'standard':
        standard, 'contract': contract,
        'comparator_event_energy_difference_mwh':
        comparator_event_energy_difference,
        'nominal_entitlement_release_mwh': float(len(event_hours) * r_mw),
        'hour_20_import_difference_mw': float(contract['total'][20] -
        standard['total'][20]), 'event_dual_sum': float(contract['duals'][
        event_hours].sum()), 'event_cap_finite_difference': float((
        contract_tighter['objective'] - contract['objective']) /
        finite_difference_step), 'finite_difference_step_mw':
        finite_difference_step, 'delay_change_minutes': 60 * (contract[
        'mean_delay_hours'] - standard['mean_delay_hours']),
        'energy_cost_change': contract['facility_energy_cost'] - standard[
        'facility_energy_cost'], 'objective_change': contract['objective'] -
        standard['objective']}

def sensitivity_grid(p: dict) -> dict:
    means = np.linspace(0.12, 0.26, 8)
    concentrations = np.array([4, 6, 9, 15, 25, 40], dtype=float)
    avg_r = np.zeros((len(concentrations), len(means)))
    principal_gain = np.zeros_like(avg_r)
    theta = np.linspace(p['theta_low'], p['theta_high'], 401)
    firm = np.array([p['A'] / (p['k'] + p['c']), p['b'] / p['a'], 0.0])
    firm_value = social_surplus(firm, 0.5 * (p['theta_low'] + p['theta_high'
        ]), p)
    for i, nu in enumerate(concentrations):
        for j, mean in enumerate(means):
            pp = dict(p)
            mean_share = mean / p['x_max']
            pp['alpha'] = mean_share * nu
            pp['beta'] = (1 - mean_share) * nu
            grid = mechanism_grid(len(theta), pp)
            avg_r[i, j] = mean_on_uniform(grid['second_best'][:, 2], theta,
                pp)
            principal = mean_on_uniform(grid['welfare_sb'] - pp[
                'rent_weight_chi'] * grid['rent'], theta, pp)
            principal_gain[i, j] = principal - firm_value
    return {'means': means, 'concentrations': concentrations, 'avg_r': avg_r
        , 'principal_gain': principal_gain}

def information_sensitivity(p: dict) -> dict:
    widths = np.linspace(0.1, 0.8, 15)
    mean_theta = 0.65
    value_screening = []
    menu_gain = []
    distortions = []
    for width in widths:
        pp = dict(p)
        pp['theta_low'] = mean_theta - width / 2
        pp['theta_high'] = mean_theta + width / 2
        grid = mechanism_grid(801, pp)
        theta = grid['theta']
        continuous_principal = mean_on_uniform(grid['welfare_sb'] - pp[
            'rent_weight_chi'] * grid['rent'], theta, pp)
        pooled = tier_mechanism(1, theta, pp)
        firm = np.array([pp['A'] / (pp['k'] + pp['c']), pp['b'] / pp['a'],
            0.0])
        firm_value = social_surplus(firm, mean_theta, pp)
        value_screening.append(continuous_principal - pooled['principal'])
        menu_gain.append(continuous_principal - firm_value)
        distortions.append(mean_on_uniform(grid['first_best'][:, 2] - grid[
            'second_best'][:, 2], theta, pp))
    return {'widths': widths, 'value_screening': np.asarray(value_screening)
        , 'menu_gain': np.asarray(menu_gain), 'distortions': np.asarray(
        distortions)}

def numerical_verification(grid: dict, p: dict) -> dict:
    theta = grid['theta']
    sb = grid['second_best']
    d_eff, k_eff = primitives(p)
    foc = d_eff * sb[:, 2] + (p['v'] + p['d']) * headroom_cdf(sb[:, 2], p
        ) - (k_eff - grid['psi'])
    interior = (sb[:, 2] > 1e-10) & (sb[:, 2] < p['x_max'] - 1e-10)
    foc_residual = float(np.max(np.abs(foc[interior])))
    q_residual = np.max(np.abs((p['k'] + p['c']) * sb[:, 0] - p['c'] * sb[:,
        2] - p['A']))
    s_residual = np.max(np.abs(p['a'] * sb[:, 1] - p['lambda'] * sb[:, 2] -
        p['b']))
    max_hessian_eigenvalue = -np.inf
    for r in sb[:, 2]:
        hessian = np.array([[-(p['k'] + p['c']), 0.0, p['c']], [0.0, -p['a']
            , p['lambda']], [p['c'], p['lambda'], -(p['gamma'] + p['c']) - (
            p['v'] + p['d']) * headroom_pdf(r, p)]])
        max_hessian_eigenvalue = max(max_hessian_eigenvalue, float(np.linalg
            .eigvalsh(hessian).max()))
    idx = np.linspace(0, len(theta) - 1, 121).round().astype(int)
    th = theta[idx]
    x = sb[idx]
    r = x[:, 2]
    common = np.array([applicant_common_value(xx, p) for xx in x])
    transfer = grid['transfer'][idx]
    truthful = transfer + common - th * r
    report_utilities = transfer[None, :] + common[None, :] - th[:, None
        ] * r[None, :]
    max_ic_gain = float(np.max(report_utilities - truthful[:, None]))
    min_ir = float(np.min(truthful))
    rng = np.random.default_rng(p['seed'])
    draws = p['x_max'] * rng.beta(p['alpha'], p['beta'], 1000000)
    r_bar = mean_on_uniform(sb[:, 2], theta, p)
    losses = np.maximum(r_bar - draws, 0.0)
    mc_mean = float(losses.mean())
    mc_se = float(losses.std(ddof=1) / math.sqrt(len(losses)))
    exact = float(expected_shortfall(r_bar, p))
    return {'effective_curvature_D': d_eff, 'effective_intercept_K': k_eff,
        'max_foc_residual': foc_residual, 'max_q_foc_residual': float(
        q_residual), 'max_s_foc_residual': float(s_residual),
        'max_hessian_eigenvalue': max_hessian_eigenvalue,
        'max_discrete_ic_gain': max_ic_gain, 'minimum_discrete_utility':
        min_ir, 'mc_draws': len(draws), 'contracted_r_for_mc': r_bar,
        'exact_shortfall': exact, 'mc_shortfall': mc_mean,
        'mc_standard_error': mc_se, 'mc_95_low': mc_mean - 1.96 * mc_se,
        'mc_95_high': mc_mean + 1.96 * mc_se, 'max_flexibility_share_slack':
        float(np.min(p['rho'] * sb[:, 0] - sb[:, 2])),
        'maximum_adjacent_change': float(np.max(np.diff(sb[:, 2]))),
        'maximum_positive_monotonicity_violation': float(max(0.0, np.max(np.
        diff(sb[:, 2]))))}

def cdf(r):
    z = np.clip(np.asarray(r) / 0.35, 0, 1)
    return 56 * z ** 5 - 140 * z ** 6 + 120 * z ** 7 - 35 * z ** 8

def loss(r):
    z = np.clip(np.asarray(r) / 0.35, 0, 1)
    return 0.35 * (28 / 3 * z ** 6 - 20 * z ** 7 + 15 * z ** 8 - 35 / 9 * z
        ** 9)

def alloc(w, cap=0.35):
    h = lambda r: 0.48 * r + 3.5 * cdf(r) - 1.48 + w
    r = 0.0 if h(0) >= 0 else cap if h(cap) <= 0 else brentq(h, 0, cap, xtol
        =5e-15)
    return np.array([0.65 + 0.6 * r, 0.4 + r, r])

def val(r, w):
    return 0.4625 + (1.48 - w) * r - 0.24 * r * r - 3.5 * loss(r)

def av(f, points=None):
    return quad(f, 0.4, 0.9, points=points, epsabs=2e-12, epsrel=2e-12,
        limit=200)[0] / 0.5

def tiers(cuts):
    J = W = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        v = a + b - 0.4
        r = alloc(v)[2]
        p = (b - a) / 0.5
        J += p * val(r, v)
        W += p * val(r, (a + b) / 2)
    return [J, W, W - J]

def extension_results():
    out = {}
    J = av(lambda t: val(alloc(2 * t - 0.4)[2], 2 * t - 0.4))
    W = av(lambda t: val(alloc(2 * t - 0.4)[2], t))
    FB = av(lambda t: val(alloc(t)[2], t))
    out['quad'] = dict(J=J, W=W, FB=FB, rent=W - J)
    xx = np.linspace(0, 0.35, 1001)
    out['loss_error'] = float(max(abs(loss(xx) - expected_shortfall(xx, BASE
        ))))
    low = alloc(1.4)[2]
    high = alloc(0.4)[2]
    mu = 0.48 + 3.5 * beta.pdf(low / 0.35, 5, 4) / 0.35
    out['mu'] = mu
    out['release_range'] = [low, high]
    out['tiers'] = [dict(m=m, J=tiers(np.linspace(0.4, 0.9, m + 1))[0], loss
        =J - tiers(np.linspace(0.4, 0.9, m + 1))[0], bound=1 / (24 * mu * m
        * m)) for m in [1, 2, 3, 4, 5, 10]]
    f = lambda c: -tiers([0.4, c, 0.9])[0]
    opt = minimize_scalar(f, bounds=(0.4000001, 0.8999999), method='bounded'
        , options={'xatol': 1e-13})
    foc = lambda c: val(alloc(c)[2], 2 * c - 0.4) - val(alloc(c + 0.5)[2], 2
        * c - 0.4)
    root = brentq(foc, 0.400001, 0.899999, xtol=1e-13)
    scan = np.linspace(0.4000001, 0.8999999, 10001)
    vs = np.array([f(c) for c in scan])
    out['two'] = dict(cut=float(opt.x), root=root, residual=foc(root),
        scan_cut=float(scan[vs.argmin()]), scan_step=float(scan[1] - scan[0]
        ), metrics=tiers([0.4, root, 0.9]), x=[alloc(root).tolist(), alloc(
        root + 0.5).tolist()])
    out['chance'] = []
    for eps in [0.01, 0.05, 0.1, 0.2]:
        cap = 0.35 * beta.ppf(eps, 5, 4)
        cut = np.clip((1.48 - 0.48 * cap - 3.5 * eps + 0.4) / 2, 0.4, 0.9)
        pts = [cut] if 0.4 < cut < 0.9 else None
        j = av(lambda t: val(alloc(2 * t - 0.4, cap)[2], 2 * t - 0.4), pts)
        w = av(lambda t: val(alloc(2 * t - 0.4, cap)[2], t), pts)
        r = av(lambda t: alloc(2 * t - 0.4, cap)[2], pts)
        out['chance'].append(dict(eps=eps, cap=cap, J=j, W=w, rent=w - j,
            mean_r=r, bunch=(cut - 0.4) / 0.5, cut=cut))
    out['learning'] = [dict(n=n, delta=float(np.sqrt(np.log(40) / (2 * n))))
        for n in [100, 500, 1000, 5000, 10000]]
    E = np.array([[1.05, 0.85], [1.1, 1.05], [1.2, 1.1]])
    prob = np.array([0.3, 0.4, 0.3])
    mean = float(prob @ E.sum(axis=1))
    eta = 0.02
    KH = 1.3 / 3 + (2 * 1.2 - 0.8) * 2.3 / 9 + 0.5 + 0.2 + 2 * eta
    DH = 6 + 1.2 - 2 * 1.2 / 3 + 2 / 9 - 0.5

    def ix(w):
        r = (KH - w) / DH
        return np.array([(2.3 + r) / 3, 0.4 + r, r])

    def B(x):
        q, s, r = x
        return 1.3 * q - 0.4 * q * q + 0.2 * s - 3 * r * r - eta * (4 * mean
            - 6 * q)

    def H(x):
        q, s, r = x
        return -0.25 * s * s - 0.6 * (q - r) ** 2 + 0.5 * s * r + 0.5 * r

    def v(x, w):
        return B(x) + H(x) - w * x[2]

    def lp(q, r, e):
        z = linprog([1, 4, 4, 1], A_ub=[[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0
            , 1]], b_ub=[q, q - r, q], A_eq=[[1, 1, 0, 0], [0, 0, 1, 1]],
            b_eq=e, bounds=(0, None), method='highs')
        assert z.success, z.message
        return z

    def qp(w):

        def obj(z):
            q, s, r = z[:3]
            v = 1.3 * q - 0.4 * q * q + 0.2 * s - 0.25 * s * s - 0.6 * (q -
                r) ** 2 + 0.5 * s * r + 0.5 * r - 3 * r * r - w * r
            return -v + eta * sum((prob[i] * np.dot([1, 4, 4, 1], z[3 + 4 *
                i:7 + 4 * i]) for i in range(3)))

        def eq(z):
            return np.array([z[3 + 4 * i] + z[4 + 4 * i] - E[i, 0] for i in
                range(3)] + [z[5 + 4 * i] + z[6 + 4 * i] - E[i, 1] for i in
                range(3)])

        def ine(z):
            q, s, r = z[:3]
            v = [0.35 * q - r]
            for i in range(3):
                a, b, c, d = z[3 + 4 * i:7 + 4 * i]
                v.extend([q - a, q - r - b - c, q - d])
            return np.array(v)
        z = np.zeros(15)
        z[:3] = [0.9, 0.5, 0.1]
        for i in range(3):
            z[3 + 4 * i:7 + 4 * i] = lp(0.9, 0.1, E[i]).x
        rs = minimize(obj, z, method='SLSQP', constraints=[{'type': 'eq',
            'fun': eq}, {'type': 'ineq', 'fun': ine}], bounds=[(0.35, 1), (0
            , 1), (0, 0.35)] + [(0, None)] * 12, options={'ftol': 1e-12,
            'maxiter': 500})
        assert rs.success, rs.message
        return (rs.x[:3], -rs.fun)
    rows = []
    qe = ce = bal = caperr = 0.0
    for t in [0.4, 0.525, 0.65, 0.775, 0.9]:
        x = ix(2 * t - 0.4)
        q, s, r = x
        u = ((KH + 0.4) * (0.9 - t) - (0.81 - t * t)) / DH
        xx, vv = qp(2 * t - 0.4)
        qe = max(qe, max(abs(xx - x)), abs(vv - v(x, 2 * t - 0.4)))
        schedules = []
        for i, e in enumerate(E):
            z = lp(q, r, e)
            ce = max(ce, abs(z.fun - (4 * e.sum() - 6 * q)))
            bal = max(bal, max(abs(z.eqlin.residual)))
            caperr = max(caperr, max(np.maximum(-z.ineqlin.residual, 0)))
            schedules.append(dict(state=i + 1, x=z.x.tolist(), cost=z.fun))
        rows.append(dict(type=t, x=x.tolist(), rent=u, charge=B(x) - t * r -
            u, schedules=schedules))
    ts = np.linspace(0.4, 0.9, 301)
    rs = np.array([ix(2 * t - 0.4)[2] for t in ts])
    us = ((KH + 0.4) * (0.9 - ts) - (0.81 - ts * ts)) / DH
    dev = us[None, :] + (ts[None, :] - ts[:, None]) * rs[None, :] - us[:,
        None]
    out['integrated'] = dict(K=KH, D=DH, E=E.tolist(), prob=prob.tolist(),
        mean=mean, rows=rows, J=av(lambda t: v(ix(2 * t - 0.4), 2 * t - 0.4)
        ), W=av(lambda t: v(ix(2 * t - 0.4), t)), FB=av(lambda t: v(ix(t), t
        )), firm=v(np.array([2.3 / 3, 0.4, 0]), 0), qp_error=qe, cost_error=
        ce, balance=bal, cap_error=caperr, IC_gain=float(dev.max()))
    grid = mechanism_grid(2001, BASE)
    th = grid['theta']
    mx = np.array([mean_on_uniform(grid['second_best'][:, j], th, BASE) for
        j in range(3)])
    sc = solve_scheduler(mx[0] * 100, mx[2] * 100, BASE['seed'])
    completion = {}
    for tag in ['standard', 'contract']:
        done = np.array([np.flatnonzero(row > 1e-09)[-1] + 1 for row in sc[
            tag]['x']])
        completion[tag] = dict(mean_flow=float(np.mean(done - sc['release'])
            ), max_flow=float(max(done - sc['release'])), on_time=int(np.sum
            (done <= sc['deadline'] + 1)))
    out['completion'] = completion
    return out


def convert(z):
    if isinstance(z, np.ndarray): return z.tolist()
    if isinstance(z, (np.integer, np.floating)): return z.item()
    raise TypeError(type(z).__name__)

def reproduce():
    p = dict(BASE)
    g = mechanism_grid(2001, p)
    th = g['theta']
    avg = lambda z: mean_on_uniform(z, th, p)
    xbar = np.array([avg(g['second_best'][:,i]) for i in range(3)])
    out = {'parameters': p, 'continuous_grid': g,
        'summary': {'J': avg(g['welfare_sb'])-avg(g['rent']),
        'W': avg(g['welfare_sb']), 'FB': avg(g['welfare_fb']),
        'rent': avg(g['rent']), 'allocation': xbar},
        'tiers': [tier_mechanism(m, th, p) for m in [1,2,3,4,5,10]],
        'uncertainty_grid': sensitivity_grid(p),
        'type_dispersion': information_sensitivity(p),
        'scheduler': solve_scheduler(100*xbar[0],100*xbar[2],p['seed']),
        'verification': numerical_verification(g,p),
        'extensions': extension_results()}
    print(json.dumps(out,default=convert,indent=2))

if __name__ == '__main__':
    reproduce()
