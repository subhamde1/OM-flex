"""Public-trace robustness calculation for Appendix G.
Protocol is fixed in Appendix G. Run with Python, NumPy and SciPy.
The archive is downloaded only if absent.
"""
from pathlib import Path
import csv, io, tarfile, hashlib, urllib.request, json, math
from collections import Counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack
from scipy.integrate import quad

P = Path(__file__).resolve().parent
URL = ('https://aliopentrace.oss-cn-beijing.aliyuncs.com/'
       'v2020GPUTraces/pai_task_table.tar.gz')
SHA = 'cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5'
DT = 0.25
QREF = 0.65
TOL = 1e-9
OPTS = dict(primal_feasibility_tolerance=TOL,
            dual_feasibility_tolerance=TOL,
            ipm_optimality_tolerance=TOL)

def load_rows():
    f = P/'raw/pai_task_table.tar.gz'
    f.parent.mkdir(exist_ok=True)
    if not f.exists():
        with urllib.request.urlopen(URL, timeout=90) as src, f.open('wb') as out:
            while True:
                chunk = src.read(1024*1024)
                if not chunk: break
                out.write(chunk)
    assert hashlib.sha256(f.read_bytes()).hexdigest() == SHA
    with tarfile.open(f) as arc:
        raw = arc.extractfile('pai_task_table.csv').read()
    assert hashlib.sha256(raw).hexdigest() == (
        '6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de')
    counts = Counter()
    rows = []
    n = 0
    for row in csv.reader(io.StringIO(raw.decode())):
        n += 1; counts[row[0]] += 1
        try: inst, start, end, gpu = map(float, [row[2],row[4],row[5],row[8]])
        except ValueError: continue
        if (row[3] == 'Terminated' and inst == 1 and gpu == 100
                and np.isfinite(start+end) and 0 < start < end):
            rows.append((row[0],start,end,row[9]))
    before = len(rows)
    rows = [r for r in rows if counts[r[0]] == 1]
    arr = np.array([[r[1],r[2]] for r in rows])
    day = np.floor((arr[:,0]+28800)/86400).astype(int)
    first = math.floor((arr[:,0].min()+28800)/86400)+1
    last = math.floor((arr[:,1].max()+28800)/86400)-1
    days = np.arange(first,last+1)
    local_start = (arr[:,0]+28800)/3600 - 24*day
    local_end = (arr[:,1]+28800)/3600 - 24*day
    keep = ((day >= first) & (day <= last) & (local_end+4 <= 24))
    byday = {int(d):np.c_[local_start[keep & (day==d)],
                            local_end[keep & (day==d)]] for d in days
             if np.any(keep & (day==d))}
    meta = dict(raw_rows=n,successful_single_instance_one_gpu=before,
                single_task_jobs=len(rows),eligible_jobs=int(keep.sum()),
                complete_days=days.tolist(),empty_cohort_days=[int(d) for d in days if int(d) not in byday],
                gpu_types=dict(Counter(r[3] for r in rows)),
                excluded_by_day_or_max_slack=int((~keep).sum()),
                archive_sha256=SHA,source_url=URL)
    return byday,meta

def observed_peak(rows):
    # Half-open observed reservation intervals; departures precede arrivals.
    events = sorted([(s,1) for s,e in rows]+[(e,-1) for s,e in rows])
    busy=peak=0
    for t,change in events:
        busy += change; peak=max(peak,busy)
    return peak

def scheduling_problem(rows, capacity_scale, slack, event_duration):
    a = np.ceil(rows[:,0]/DT-1e-12).astype(int)
    b = np.ceil((rows[:,1]+slack)/DT-1e-12).astype(int)
    energy = (rows[:,1]-rows[:,0])/capacity_scale
    assert np.all((0 <= a) & (a < b) & (b <= 96))
    # Every job is retained individually; no invalid aggregate rate relaxation.
    j = np.repeat(np.arange(len(rows)),b-a)
    ts = np.concatenate([np.arange(x,y) for x,y in zip(a,b)])
    cols = np.arange(len(ts))
    eq = coo_matrix((np.ones(len(ts)),(j,cols)),shape=(len(rows),len(ts))).tocsr()
    ub = coo_matrix((np.ones(len(ts)),(ts,cols)),shape=(96,len(ts))).tocsr()
    event = ((np.arange(96)*DT >= 17) &
             (np.arange(96)*DT < 17+event_duration))
    return dict(eq=eq,ub=ub,energy=energy,j=j,ts=ts,event=event,
                rate=DT/capacity_scale,slots=(a,b))

def headroom(model,q=QREF,return_dispatch=False):
    n=model['eq'].shape[1]
    eq=hstack([model['eq'],coo_matrix((len(model['energy']),1))]).tocsr()
    ub=hstack([model['ub'],coo_matrix((DT*model['event']).reshape(-1,1))]).tocsr()
    res=linprog(np.r_[np.zeros(n),-1.],A_ub=ub,b_ub=np.full(96,DT*q),
                A_eq=eq,b_eq=model['energy'],
                bounds=[(0,model['rate'])]*n+[(0,q)],
                method='highs',options=OPTS)
    if not res.success:
        return dict(feasible=False,status=int(res.status),message=res.message)
    x=res.x[:-1];r=float(res.x[-1])
    balance=float(np.max(np.abs(model['eq']@x-model['energy'])))
    cap=float(max(0,np.max(model['ub']@x+DT*r*model['event']-DT*q)))
    upper=float(max(0,np.max(x-model['rate'])))
    out=dict(feasible=True,headroom=min(.35,r),physical_headroom=r,
             balance=balance,cap=cap,rate_violation=upper)
    if return_dispatch:out['hourly_reservation']=np.asarray(model['ub']@x).reshape(24,4).sum(1).tolist()
    return out

def minimum_capacity(model):
    n=model['eq'].shape[1]
    eq=hstack([model['eq'],coo_matrix((len(model['energy']),1))]).tocsr()
    ub=hstack([model['ub'],coo_matrix(np.full((96,1),-DT))]).tocsr()
    res=linprog(np.r_[np.zeros(n),1.],A_ub=ub,b_ub=np.zeros(96),
                A_eq=eq,b_eq=model['energy'],
                bounds=[(0,model['rate'])]*n+[(0,None)],method='highs',options=OPTS)
    assert res.success,res.message
    return float(res.x[-1])

def conditional(r):
    return max(.65+.6*r,r/.35), .4+r

def reduced_coefficients(r):
    alpha,beta=(.65,.6) if r<=.2275/.79 else (0,1/.35)
    intercept=1.3*beta-.8*alpha*beta-1.2*alpha*(beta-1)+.7
    curvature=.8*beta**2+1.2*(beta-1)**2
    return intercept,curvature

def release(w,scores,cap=0.35):
    top=min(cap,0.35)
    y=np.sort(scores); n=len(y)
    switch=.2275/.79
    breaks=np.unique(np.r_[0,y[(y>0)&(y<top)],min(top,switch),top])
    mids=(breaks[:-1]+breaks[1:])/2
    counts=np.searchsorted(y,mids,side='right')
    coeff=np.array([reduced_coefficients(x) for x in mids])
    roots=np.clip((coeff[:,0]-w-3.5*counts/n)/coeff[:,1],breaks[:-1],breaks[1:])
    candidates=np.unique(np.r_[breaks,roots])
    val=np.array([value(x,w,y) for x in candidates])
    return float(candidates[np.argmax(val)])

def value(r,w,scores):
    q,s=conditional(r)
    g=1.3*q-.4*q*q+.2*s-.25*s*s-.6*(q-r)**2+.5*s*r+.5*r-.25*r*r
    return g-w*r-3.5*np.maximum(r-scores,0).mean()

def type_breaks(scores,cap=0.35,extra=()):
    edges=[.4,.9]
    knots=np.unique(np.r_[0,scores,extra,min(cap,.2275/.79),cap])
    for x in knots:
        if 0<=x<=cap:
            for side in ['left','right']:
                f=np.searchsorted(np.sort(scores),x,side=side)/len(scores)
                K,D=reduced_coefficients(x)
                t=(K-D*x-3.5*f+.4)/2
                if .4<t<.9:edges.append(float(t))
    return sorted(set(edges))

def envelope_rent(t,scores,cap=0.35):
    # The continuous allocation is piecewise affine in type. Split at every
    # CDF, policy-cap and coupled-bound transition before integrating.
    edges=sorted(set([t,.9]+[z for z in type_breaks(scores,cap) if t<z<.9]))
    return sum((b-a)*(release(2*a-.4,scores,cap)+
                      release(2*b-.4,scores,cap))/2
               for a,b in zip(edges[:-1],edges[1:]))

def evaluate_menu(scores,test,cap=0.35,tiers=None):
    def alloc(t):
        if tiers is None:return release(2*t-.4,scores,cap)
        edges=np.linspace(.4,.9,tiers+1)
        idx=min(np.searchsorted(edges,t,side='right')-1,tiers-1)
        return release(edges[idx]+edges[idx+1]-.4,scores,cap)
    # All possible CDF-kink and quadratic-root transitions in type space.
    edges=type_breaks(scores,cap,test+1e-8)
    if tiers:edges.extend(np.linspace(.4,.9,tiers+1))
    edges=sorted(set(edges))
    def avg(f):return sum(quad(f,a,b,epsabs=2e-11,epsrel=2e-11)[0]
                              for a,b in zip(edges[:-1],edges[1:]))/.5
    J=avg(lambda t:value(alloc(t),2*t-.4,scores))
    out=dict(J=J,gain_percent=100*(J/.4625-1),
             mean_release=avg(alloc),mean_capacity=avg(lambda t:conditional(alloc(t))[0]),
             mean_speed=avg(lambda t:conditional(alloc(t))[1]),
             heldout_shortfall=avg(lambda t:float(np.maximum(alloc(t)-test,0).mean())),
             # Scores near an LP boundary use a declared 1e-8 tolerance.
             heldout_failure=avg(lambda t:float(np.mean(test<alloc(t)-1e-8))))
    out['allocations']=[dict(type=float(t),r=alloc(t),q=conditional(alloc(t))[0],
                             s=conditional(alloc(t))[1],fb_release=release(t,scores,cap))
                         for t in np.linspace(.4,.9,5)]
    return out

def run():
    (P/'computed').mkdir(exist_ok=True)
    byday,meta=load_rows()
    days=sorted(byday); cut=int(.7*len(days)); train=days[:cut];test=days[cut:]
    scale=max(observed_peak(byday[d]) for d in train)/.60
    assert scale>0
    meta.update(train_days=train,test_days=test,normalization_gpu_equivalents=scale,
                observed_train_peak=.60*scale,protocol='trace_protocol.md')
    lean_peaks=[minimum_capacity(scheduling_problem(byday[d],1.,2,2)) for d in train]
    lean_scale=max(lean_peaks)/.60
    meta['minimum_training_capacities_gpu']=lean_peaks
    meta['lean_capacity_scale']=lean_scale
    cases=[(2,2,.60),(1,2,.60),(4,2,.60),(2,1,.60),(2,4,.60),
           (2,2,.75),(2,2,.90),(2,2,'lean')]
    out=dict(metadata=meta,cases=[])
    for slack,duration,load_fraction in cases:
        case_scale=lean_scale if load_fraction=='lean' else .60*scale/load_fraction
        rows=[]
        for d in days:
            m=scheduling_problem(byday[d],case_scale,slack,duration)
            h=headroom(m,return_dispatch=(d==test[0]))
            rows.append(dict(day=d,split='train' if d in train else 'test',
                             jobs=len(byday[d]),work_gpu_hours=float(np.sum(np.diff(byday[d],axis=1))),
                             **h))
        xs=np.array([z['headroom'] for z in rows if z['split']=='train' and z['feasible']])
        ys=np.array([z['headroom'] for z in rows if z['split']=='test' and z['feasible']])
        assert len(xs) and len(ys)
        menus={str(k):evaluate_menu(xs,ys,tiers=k) for k in [1,2,4,None]}
        reliability=[]
        for eps in [.10,.05,.01]:
            # Maximum empirical-safe cap: strict shortfall X<r.
            allowed=int(np.floor(eps*len(xs)+1e-12))
            cap=float(np.sort(xs)[allowed])
            reliability.append(dict(epsilon=eps,cap=cap,**evaluate_menu(xs,ys,cap=cap)))
        result=dict(slack=slack,duration=duration,load_fraction=load_fraction,
                    capacity_scale=case_scale,days=rows,menus=menus,
                    reliability=reliability,train_scores=xs.tolist(),test_scores=ys.tolist())
        out['cases'].append(result)
        print(json.dumps(dict(slack=slack,duration=duration,load_fraction=load_fraction,days=len(rows),
                              eligible_train=len(xs),eligible_test=len(ys),
                              J=menus['None']['J'],test_failure=menus['None']['heldout_failure'])),flush=True)
        (P/'computed/trace_results.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(meta),flush=True)
    return out

if __name__=='__main__':run()
