"""Appendix H replication. Requires the unmodified Appendix G replication module, trace_compute.py."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
import trace_compute as tc

P = Path(__file__).resolve().parent
TOL = 1e-8


def allocation(theta, xs, cap=.35, tiers=None):
    if tiers is None:
        w = 2*theta-.4
    else:
        edges = np.linspace(.4, .9, tiers+1)
        j = np.searchsorted(edges, theta, side='right')-1
        j = min(max(j, 0), tiers-1)
        w = edges[j]+edges[j+1]-.4
    return tc.release(w, xs, cap)

def diagnostics(xs, physical, cap=.35, tiers=None):
    """Reference-score metrics and exact contract-capacity feasibility."""
    physical = np.asarray(physical)
    scores = np.minimum(.35, physical)
    assert np.all(physical >= 0), 'Report infeasible days, do not censor them.'
    out = tc.evaluate_menu(xs, scores, cap=cap, tiers=tiers)
    extra = np.r_[scores+TOL, (physical+TOL)/.4,
                  (.65-physical-TOL)/(1/.35-1)]
    edges = tc.type_breaks(xs, cap, extra)
    if tiers:
        edges.extend(np.linspace(.4, .9, tiers+1))
    edges = sorted(set(edges))
    def avg(f):
        return sum(quad(f, a, b, epsabs=2e-11, epsrel=2e-11)[0]
                   for a, b in zip(edges[:-1], edges[1:]))/.5
    def actual_margin(theta):
        r = allocation(theta, xs, cap, tiers)
        q, _ = tc.conditional(r)
        return physical+q-.65-r
    out['contract_capacity_failure'] = avg(
        lambda t: float(np.mean(actual_margin(t) < -TOL)))
    out['contract_capacity_shortfall'] = avg(
        lambda t: float(np.maximum(-actual_margin(t), 0).mean()))
    out['worst_type_reference_failure'] = float(np.mean(
        scores < allocation(.4, xs, cap, tiers)-TOL))
    out['policy_cap'] = cap
    return out

def summary(rows):
    h = np.array([r['physical_headroom'] for r in rows if r['feasible']])
    x = np.minimum(.35, h)
    return dict(days=len(rows),
                reference_infeasible=sum(not r['feasible'] for r in rows),
                active_workload_days=int(np.sum(h < .35-TOL)),
                policy_capped_days=int(np.sum(h >= .35-TOL)),
                distinct_scores=len(np.unique(np.round(x, 9))),
                physical_min_median_max=(np.quantile(h, [0,.5,1]).tolist()
                                         if len(h) else [None]*3),
                score_min_median_max=(np.quantile(x, [0,.5,1]).tolist()
                                      if len(x) else [None]*3))

def run():
    byday, meta = tc.load_rows()
    days = sorted(byday)
    cut = int(.7*len(days))
    train, test = days[:cut], days[cut:]
    assert train == list(range(13,55))
    assert test == list(range(55,74))
    peaks = {
        d: tc.minimum_capacity(tc.scheduling_problem(byday[d],1.,2,2))
        for d in train}
    scale = max(peaks.values())/.60
    def episode(day, split):
        if day not in peaks:
            m = tc.scheduling_problem(byday[day],1.,2,2)
            peaks[day] = tc.minimum_capacity(m)
        m = tc.scheduling_problem(byday[day],scale,2,2)
        m['event'] = np.ones(96,dtype=bool)
        h = tc.headroom(m)
        analytic = tc.QREF-peaks[day]/scale
        if h['feasible']:
            assert abs(h['physical_headroom']-analytic) < 1e-7
            err = max(h['balance'],h['cap'],h['rate_violation'])
            assert err < 1e-7
        return dict(day=day,split=split,minimum_gpu_capacity=peaks[day],
                    analytic_headroom=analytic,**h)
    tr = [episode(d,'train') for d in train]
    assert all(v['feasible'] for v in tr)
    xs = np.array([v['headroom'] for v in tr])
    xh = np.array([v['physical_headroom'] for v in tr])
    caps = {str(e):float(np.sort(xs)[int(np.floor(e*42+1e-12))])
            for e in [.10,.05,.01]}
    training = dict(
        capacity_scale=scale,training_peak=max(peaks.values()),
        training_days=train,training_scores=xs.tolist(),
        reliability_caps=caps,
        menus={str(k):diagnostics(xs,xh,tiers=k) for k in [1,2,4,None]},
        reliability={e:diagnostics(xs,xh,cap=c) for e,c in caps.items()})
    # All fitting is complete before calculating new holdout stress outcomes.
    te = [episode(d,'test') for d in test]
    out = dict(metadata=meta,capacity_scale=scale,days=tr+te,
               train_summary=summary(tr),test_summary=summary(te),
               training=training)
    if all(v['feasible'] for v in te):
        yh = np.array([v['physical_headroom'] for v in te])
        out['holdout_menus'] = {
            str(k):diagnostics(xs,yh,tiers=k) for k in [1,2,4,None]}
        out['holdout_reliability'] = {
            e:diagnostics(xs,yh,cap=c) for e,c in caps.items()}
    else:
        out['holdout_warning'] = (
            'Reference-infeasible days retained; do not censor them.')
    print(json.dumps(out,indent=2))
    return out

if __name__ == '__main__':
    run()
