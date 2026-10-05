import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import spearmanr


ROOT = Path('/tmp/hal-rank1-o59-calibration')
OUTPUT = Path('outputs/o59-rank1-return-calibration.json')


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), q * weights.sum(), side='left')])


def weighted_pearson(x, y, weights):
    weights = weights / weights.sum()
    mx, my = np.sum(weights*x), np.sum(weights*y)
    covariance = np.sum(weights*(x-mx)*(y-my))
    return float(covariance / np.sqrt(np.sum(weights*(x-mx)**2)*np.sum(weights*(y-my)**2)))


def quantile_fit(x, y, weights, tau, kind):
    z = x / 50
    if kind == 'affine':
        design = np.column_stack([np.ones(len(x)), z])
    elif kind == 'hinge':
        design = np.column_stack([np.ones(len(x)), z, np.maximum(z, 0), np.maximum(z - .8, 0)])
    else:
        raise ValueError(kind)
    n, p = design.shape
    constraints = sparse.hstack([sparse.csc_matrix(design), sparse.eye(n, format='csc'), -sparse.eye(n, format='csc')], format='csc')
    costs = np.r_[np.zeros(p), tau*weights, (1-tau)*weights]
    result = linprog(costs, A_eq=constraints, b_eq=y, bounds=[(None, None)]*p + [(0, None)]*(2*n), method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:p]


def predict(x, coefficients, kind):
    z = x / 50
    if kind == 'affine':
        return coefficients[0] + coefficients[1]*z
    return coefficients[0] + coefficients[1]*z + coefficients[2]*np.maximum(z,0) + coefficients[3]*np.maximum(z-.8,0)


def rows_for(games, subset):
    selected=[]
    for game in games:
        base=[row for row in game['rows'] if row['base']]
        if len(base)!=16:
            raise ValueError(f"game {game['id']} has {len(base)} base frames")
        if game['set'] != subset:
            continue
        weight=game['eligible_frames']/16
        for row in base:
            selected.append((game['id'],row['value'],row['return60'],weight))
    return np.asarray(selected,dtype=float)


def main():
    games=[json.loads((ROOT/'rows'/f'{i:03d}.json').read_text()) for i in range(200)]
    fit=rows_for(games,'fit');test=rows_for(games,'test')
    all_base=np.vstack([fit,test])
    records={}
    for name,data in [('fit',fit),('test',test),('all',all_base)]:
        x,y,w=data[:,1],data[:,2],data[:,3]
        records[name]={
            'games':int(len(np.unique(data[:,0]))),'frames':len(data),
            'pearson':float(np.corrcoef(x,y)[0,1]),
            'frame_weighted_pearson':weighted_pearson(x,y,w),
            'frame_weighted_ols_slope_intercept':np.polyfit(x,y,1,w=np.sqrt(w)).tolist(),
            'spearman':float(spearmanr(x,y).statistic),
            'p_return_above_100':float(np.average(y>100,weights=w)),
            'value_quantiles':np.quantile(x,[.05,.25,.5,.75,.95]).tolist(),
            'return_quantiles':np.quantile(y,[.05,.25,.5,.75,.8,.9,.95,.99]).tolist(),
        }
    rng=np.random.default_rng(9925)
    game_ids=np.unique(test[:,0])
    grouped=[test[test[:,0]==g] for g in game_ids]
    boot=[]
    for _ in range(2000):
        sample=np.vstack([grouped[i] for i in rng.integers(len(grouped),size=len(grouped))])
        boot.append(weighted_pearson(sample[:,1],sample[:,2],sample[:,3]))
    records['test']['pearson_game_bootstrap_95pct']=np.quantile(boot,[.025,.975]).tolist()

    x,y,w=fit[:,1],fit[:,2],fit[:,3]
    weights=w/w.mean()
    offset=weighted_quantile(y-x,w,.8)
    coefficients={kind:quantile_fit(x,y,weights,.8,kind).tolist() for kind in ('affine','hinge')}
    models={'offset':{'offset':offset},**{kind:{'coefficients':coef} for kind,coef in coefficients.items()}}
    for name,model in models.items():
        model['test']={}
        for label,data in [('all',test),('near_zero',test[(test[:,1]>=0)&(test[:,1]<20)]),('tail',test[test[:,2]>100])]:
            if not len(data):
                model['test'][label]={'n':0};continue
            vx,vy,vw=data[:,1],data[:,2],data[:,3]
            estimate=vx+offset if name=='offset' else predict(vx,np.array(model['coefficients']),name)
            model['test'][label]={
                'n':len(data), 'coverage':float(np.average(vy<=estimate,weights=vw)),
                'pinball_loss':float(np.average(np.maximum(.8*(vy-estimate),-.2*(vy-estimate)),weights=vw)),
                'estimate_quantiles':np.quantile(estimate,[.05,.5,.95]).tolist(),
            }
            if label in ('all','near_zero'):
                groups=[data[data[:,0]==game] for game in np.unique(data[:,0])]
                coverage_boot=[]
                for _ in range(1000):
                    sampled=np.vstack([groups[i] for i in rng.integers(len(groups),size=len(groups))])
                    sx,sy,sw=sampled[:,1],sampled[:,2],sampled[:,3]
                    sq=sx+offset if name=='offset' else predict(sx,np.array(model['coefficients']),name)
                    coverage_boot.append(float(np.average(sy<=sq,weights=sw)))
                model['test'][label]['coverage_game_bootstrap_95pct']=np.quantile(coverage_boot,[.025,.975]).tolist()
        if name!='offset':
            model['at_values']={str(v):float(predict(np.array([v]),np.array(model['coefficients']),name)[0]) for v in (-40,0,10,20,40,80)}

    bins=[]
    for low,high in [(-np.inf,-40),(-40,-20),(-20,0),(0,10),(10,20),(20,40),(40,80),(80,np.inf)]:
        data=test[(test[:,1]>=low)&(test[:,1]<high)]
        if not len(data):continue
        vx,vy,vw=data[:,1],data[:,2],data[:,3]
        bins.append({'low':None if not np.isfinite(low) else low,'high':None if not np.isfinite(high) else high,'n':len(data),'games':len(np.unique(data[:,0])),
            'value_median':float(np.median(vx)),'return_q80':weighted_quantile(vy,vw,.8),
            'return_q90':weighted_quantile(vy,vw,.9),'p_return_above_100':float(np.average(vy>100,weights=vw))})
    tail_rows=[]
    high_clusters=0
    for game in games:
        high_clusters+=len(game['high_clusters'])
        for row in game['rows']:
            if row['tail']:
                tail_rows.append({'game':game['id'],'set':game['set'],'value':row['value'],'return60':row['return60'],'cluster':row['cluster'],'base':row['base']})
    near=[row for row in tail_rows if 0<=row['value']<20]
    tail={'clusters':high_clusters,'sampled_rows':len(tail_rows),'sampled_games':len({r['game'] for r in tail_rows}),
        'near_zero_value_rows':len(near),'near_zero_value_games':len({r['game'] for r in near}),
        'near_zero_value_clusters':len({(r['game'],r['cluster']) for r in near}),
        'near_zero_value_examples':sorted(near,key=lambda r:r['return60'],reverse=True)[:20]}

    result={'selection':json.loads((ROOT/'selection.json').read_text()),'summary':records,'models':models,'test_bins':bins,'tail':tail,
        'analysis':'Natural frame sample: 16 stratified random frames per held-out game, one random player view. Additional >100-return frames are only for tail diagnostics and are excluded from fitting/correlation.'}
    OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary':records,'models':models,'test_bins':bins,'tail':{k:v for k,v in tail.items() if k!='near_zero_value_examples'}},indent=2,allow_nan=False))
    print('OUTPUT',OUTPUT.resolve())


if __name__=='__main__':
    main()
