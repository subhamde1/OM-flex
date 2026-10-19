"""Exact synthetic replication for the workload-distortion results (Appendix I).

No company data or network connection is used. Run from any directory with
Python, NumPy, SciPy and Matplotlib installed. All model and figure inputs are
written to workload_distortion_results.json beside this file.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize, linprog, LinearConstraint, Bounds

P = Path(__file__).resolve().parent
A, K, B, AA, C, LAM, V, GAM = map(F, ['1.3','.8','.2','.5','1.2','.5','.2','.5'])
QMIN, RHO, RMAX, LOW, HIGH = map(F, ['.4','.35','.35','1','1.1'])
D0 = GAM+C-C*C/(K+C)-LAM*LAM/AA
BETA0 = C/(K+C)

@dataclass
class Model:
    name: str
    slots: int = 0
    event_slots: int = 0
    work: F = F(0)
    eta: F = F(0)

    def __post_init__(self):
        self.top = min(RMAX,RHO)
        if self.event_slots:
            self.top = min(self.top,(F(self.slots)-self.work)/self.event_slots)
        self.lines = [(QMIN,F(0)),(F(0),1/RHO),(A/(K+C),BETA0),(F(1),F(0))]
        if self.slots:
            self.lines.append((self.work/self.slots,F(self.event_slots,self.slots)))
        if self.eta:
            self.lines += [(A/(K+C)+self.eta/(K+C),BETA0),(F('0.4'),F(2))]
        edges = {F(0),self.top}
        for a,b in self.lines:
            for c,d in self.lines:
                if b != d:
                    z=(c-a)/(b-d)
                    if 0 < z < self.top:edges.add(z)
        edges=sorted(edges)
        self.regimes=[]
        for left,right in zip(edges[:-1],edges[1:]):
            middle=(left+right)/2
            q=self.q(middle)
            candidates=[z for z in self.lines if z[0]+z[1]*middle==q]
            assert candidates
            alpha,beta=candidates[0]
            active_cost=bool(self.eta and F('.4')-q+2*middle>0)
            cost0,costq,costr=(F('.4'),F(-1),F(2)) if active_cost else (F(0),F(0),F(0))
            zeta,sigma=B/AA,LAM/AA
            g=A*alpha-(K+C)*alpha*alpha/2+B*zeta-AA*zeta*zeta/2-self.eta*(cost0+costq*alpha)
            nu=A*beta-K*alpha*beta+B*sigma-AA*zeta*sigma-C*alpha*(beta-1)+LAM*zeta+V-self.eta*(costq*beta+costr)
            kap=K*beta*beta+AA*sigma*sigma+C*(beta-1)**2-2*LAM*sigma+GAM
            assert kap==D0+(K+C)*(beta-BETA0)**2+AA*(sigma-LAM/AA)**2
            item=dict(left=left,right=right,alpha=alpha,beta=beta,g=g,nu=nu,kappa=kap,cost_active=active_cost)
            if self.regimes and all(self.regimes[-1][k]==item[k] for k in ['alpha','beta','g','nu','kappa','cost_active']):
                self.regimes[-1]['right']=right
            else:self.regimes.append(item)
        for a,b in zip(self.regimes[:-1],self.regimes[1:]):
            r=a['right'];assert r==b['left']
            assert self.poly(a,r)==self.poly(b,r)
            assert a['nu']-a['kappa']*r >= b['nu']-b['kappa']*r
        assert all(x['kappa']>=D0 for x in self.regimes)
        self.edges=sorted({r[k] for r in self.regimes for k in ['left','right']})
        self.type_edges={LOW,HIGH}
        for r in self.regimes:
            for z in [r['left'],r['right']]:
                t=(r['nu']-r['kappa']*z+LOW)/2
                if LOW<t<HIGH:self.type_edges.add(t)
        self.type_edges=sorted(self.type_edges)

    def q(self,r):
        q0=(A+C*r)/(K+C)
        if self.eta:
            q1=q0+self.eta/(K+C)
            q0=min(q1,max(q0,F('.4')+2*r))
        lower=max(QMIN,r/RHO)
        if self.slots:lower=max(lower,(self.work+self.event_slots*r)/self.slots)
        return min(F(1),max(lower,q0))

    def cost(self,q,r):return max(F(0),F('.4')-q+2*r) if self.eta else F(0)

    @staticmethod
    def poly(reg,r):return reg['g']+reg['nu']*r-reg['kappa']*r*r/2

    def value(self,r,w):
        q=self.q(r);s=(B+LAM*r)/AA
        return A*q-K*q*q/2+B*s-AA*s*s/2-C*(q-r)**2/2+LAM*s*r+V*r-GAM*r*r/2-w*r-self.eta*self.cost(q,r)

    def release(self,w):
        candidates=set(self.edges)
        for reg in self.regimes:
            candidates.add(min(reg['right'],max(reg['left'],(reg['nu']-w)/reg['kappa'])))
        return max(sorted(candidates),key=lambda r:self.value(r,w))

    def allocation(self,w):
        r=self.release(w)
        return (self.q(r),(B+LAM*r)/AA,r)

    def rent(self,t):
        edges=[t]+[x for x in self.type_edges if t<x<HIGH]+[HIGH]
        return sum((v-u)*(self.release(2*u-LOW)+self.release(2*v-LOW))/2 for u,v in zip(edges[:-1],edges[1:]))

    def distortion(self,t):
        fb=self.allocation(t);ic=self.allocation(2*t-LOW)
        dr=fb[2]-ic[2]
        parts=[max(F(0),min(fb[2],z['right'])-max(ic[2],z['left'])) for z in self.regimes]
        assert sum(parts)==dr
        dq=sum(z['beta']*v for z,v in zip(self.regimes,parts))
        assert dq==fb[0]-ic[0]
        assert (LAM/AA)*dr==fb[1]-ic[1]
        return fb,ic,(dq,(LAM/AA)*dr,dr)

    def qp(self,w):
        n=4 if self.eta else 3
        def fun(x):
            q,s,r=x[:3]
            out=-(float(A)*q-float(K)*q*q/2+float(B)*s-float(AA)*s*s/2-float(C)*(q-r)**2/2+float(LAM)*s*r+float(V)*r-float(GAM)*r*r/2-float(w)*r)
            return out+float(self.eta)*x[3] if n==4 else out
        def jac(x):
            q,s,r=x[:3]
            z=[-float(A)+float(K)*q+float(C)*(q-r),-float(B)+float(AA)*s-float(LAM)*r,-float(C)*(q-r)-float(LAM)*s-float(V)+float(GAM)*r+float(w)]
            return np.r_[z,float(self.eta)] if n==4 else np.array(z)
        mat=[[-float(RHO),0,1]+([0] if n==4 else [])];rhs=[0.]
        if self.slots:mat.append([-self.slots,0,self.event_slots]+([0] if n==4 else []));rhs.append(-float(self.work))
        if n==4:mat.append([-1,0,2,-1]);rhs.append(-.4)
        res=minimize(fun,np.r_[1.,.4,0.,0.] if n==4 else np.array([1.,.4,0.]),jac=jac,
                     constraints=[LinearConstraint(np.array(mat),-np.inf,np.array(rhs))],
                     bounds=Bounds([float(QMIN),0,0]+([0] if n==4 else []),[1,1,float(RMAX)]+([np.inf] if n==4 else [])),
                     method='SLSQP',options={'ftol':1e-13,'maxiter':300})
        assert res.success,res.message
        return res.x[:3],res.fun

    def feasibility_lp(self,q,r,energy):
        event=np.r_[np.ones(self.event_slots),np.zeros(self.slots-self.event_slots)]
        rhs=float(q)-float(r)*event
        res=linprog(np.zeros(self.slots),A_eq=np.ones((1,self.slots)),b_eq=[float(energy)],bounds=[(0,float(b)) for b in rhs],method='highs')
        assert res.success,res.message
        return max(abs(res.x.sum()-float(energy)),float(max(0,np.max(res.x-rhs))),float(max(0,-res.x.min())))

def encoded(x):
    if isinstance(x,F):return {'exact':str(x),'value':float(x)}
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.floating,np.integer)):return x.item()
    raise TypeError(type(x).__name__)

def run():
    models=[Model('Workload-relaxed benchmark'),Model('One event slot in three',3,1,F('2.19')),
            Model('Four event slots in five',5,4,F('3.3')),Model('Piecewise-affine operating cost',eta=F('.2'))]
    out={'scope':'Fully synthetic; exact rational regime solver, independent constrained QPs, scheduling LPs, and finite-grid IC checks. No company data.','parameters':dict(A=A,k=K,b=B,a=AA,c=C,lambda_=LAM,v=V,gamma_H=GAM,q_min=QMIN,rho=RHO,bar_x=RMAX,theta_low=LOW,theta_high=HIGH,D0=D0,beta0=BETA0),'cases':[]}
    max_qp=0.;max_obj=0.;max_lp=0.;qp_count=0;lp_count=0
    theta=[LOW+(HIGH-LOW)*i/1000 for i in range(1001)]
    for model in models:
        fb=[];ic=[];gaps=[];rents=[];charges=[]
        for t in theta:
            a,b,g=model.distortion(t);u=model.rent(t)
            common=A*b[0]-K*b[0]**2/2+B*b[1]-GAM*b[2]**2/2-model.eta*model.cost(b[0],b[2])
            fb.append(list(map(float,a)));ic.append(list(map(float,b)));gaps.append(list(map(float,g)));rents.append(float(u));charges.append(float(common-t*b[2]-u))
        ts=np.array(list(map(float,theta)));ic=np.array(ic);rents=np.array(rents)
        gain=rents[None,:]+(ts[None,:]-ts[:,None])*ic[None,:,2]-rents[:,None]
        max_ic=float(np.max(gain));assert max_ic<2e-12
        assert np.max(np.diff(ic[:,2]))<1e-12
        assert rents.min()>=-1e-12
        own_qp=own_obj=own_lp=0.;nqp=nlp=0
        for t in theta[::20]:
            for w in [t,2*t-LOW]:
                exact=np.array(list(map(float,model.allocation(w))))
                numeric,obj=model.qp(w)
                own_qp=max(own_qp,float(np.max(abs(exact-numeric))))
                own_obj=max(own_obj,abs(float(model.value(F(str(exact[2])),w))+obj))
                nqp+=1
                if model.slots:
                    for energy in [model.work*F('0.9'),model.work]:
                        own_lp=max(own_lp,model.feasibility_lp(exact[0],exact[2],energy));nlp+=1
        assert own_qp<3e-6,(model.name,own_qp)
        assert own_obj<2e-10,(model.name,own_obj)
        selected=F('1.05');af,ai,diff=model.distortion(selected)
        item={'name':model.name,'slots':model.slots,'event_slots':model.event_slots,'max_work':model.work,'eta':model.eta,
              'regimes':model.regimes,'theta':ts,'first_best':fb,'second_best':ic,'distortions':gaps,'rent':rents,'charge':charges,
              'selected_type':{'theta':selected,'virtual_cost':2*selected-LOW,'first_best':af,'second_best':ai,'gaps':diff},
              'checks':{'qp_comparisons':nqp,'max_allocation_discrepancy':own_qp,'max_objective_discrepancy':own_obj,'scheduling_lp_checks':nlp,'max_lp_residual':own_lp,'max_pairwise_IC_gain':max_ic,'exact_gap_decomposition_checks':len(theta)}}
        out['cases'].append(item)
        max_qp=max(max_qp,own_qp);max_obj=max(max_obj,own_obj);max_lp=max(max_lp,own_lp);qp_count+=nqp;lp_count+=nlp
    low=models[1];rstar=F('0.73')/(1/RHO-F(1,3));qstar=rstar/RHO
    left=next(z for z in low.regimes if z['right']==rstar);right=next(z for z in low.regimes if z['left']==rstar)
    wminus=left['nu']-left['kappa']*rstar;wplus=right['nu']-right['kappa']*rstar
    shadow=(K+C)*qstar-A-C*rstar
    assert wminus-wplus==shadow*(right['beta']-left['beta'])>0
    for w in [wplus,(wplus+wminus)/2,wminus]:assert low.release(w)==rstar
    t0=(wplus+LOW)/2;t1=(wminus+LOW)/2
    ch=[]
    for t in [t0,(t0+t1)/2,t1]:
        q,s,r=low.allocation(2*t-LOW)
        common=A*q-K*q*q/2+B*s-GAM*r*r/2
        ch.append(common-t*r-low.rent(t))
    assert ch[0]==ch[1]==ch[2]
    out['pooling']={'r_star':rstar,'q_star':qstar,'shadow_cost':shadow,'beta_minus':left['beta'],'beta_plus':right['beta'],'w_plus':wplus,'w_minus':wminus,'virtual_cost_width':wminus-wplus,'IC_type_interval':[t0,t1],'type_probability':(t1-t0)/(HIGH-LOW),'constant_charge':ch[0]}
    out['verification']={'exact_regime_and_gap_checks':4004,'independent_QPs':qp_count,'max_QP_allocation_error':max_qp,'max_QP_objective_error':max_obj,'independent_scheduling_LPs':lp_count,'max_scheduling_residual':max_lp,'IC_grids':'1001 reports x 1001 types in each of four cases','all_assertions_passed':True}
    target=P/'workload_distortion_results.json';target.write_text(json.dumps(out,default=encoded,indent=2))
    print(json.dumps({'selected':[x['selected_type'] for x in out['cases']],'pooling':out['pooling'],'verification':out['verification']},default=encoded,indent=2))
    return out

if __name__=='__main__':run()
