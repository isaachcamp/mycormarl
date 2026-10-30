"""Retain public allocation references, fixed-command refinement and short timings."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

with redirect_stdout(sys.stderr):
    import jax
    import jax.numpy as jnp
    from mycormarl.environments.base_mycor import BaseMycorMarl, PLANT, FUNGUS
    from mycormarl.fungus.traits import FungusTraits
    from mycormarl.params import EnvConfig, SpeciesParams
    from mycormarl.plant.traits import PlantTraits
    from mycormarl.static_rollout import evaluate_static_allocation
    from scripts.phosphate_qualification import benchmark_environment, qualification_species


# Independent DOP853 illustrative reference retained in the agreed case studies.
# Columns: plant growth, output, final C, final P, C transfer. D changes limiter.
CASES = (
    ('A',1.,1.,.25,1.,0.,(.4740904191,.1580301397,.3678794412,.3678794412,0.)),
    ('B',2.,.5,.25,1.,1.,(.2370452096,.0790150699,.5518191618,.1839397206,1.1321205588)),
    ('C',1.,1.,.25,1.,1.,(.3242492688,.1080830896,.1353352832,.5676676416,.4323323584)),
    ('D',2.,.2,.25,1.,1.,(.4133337997,.1377779332,.3818521706,.7159243635,1.0670360965)),
    ('E',1.,1.,.75,1.,1.,(.1080830896,.3242492688,.1353352832,.5676676416,.4323323584)),
    ('F',1.,1.,.25,2.,1.,(.3242492688,.0540415448,.1353352832,.5676676416,.4323323584)),
)


def environment(c, p, scale=1., *, reciprocal=False, rate_step=.02, dt=1.):
    return BaseMycorMarl(
        EnvConfig(dt=dt,max_steps=1,allocation_rate_step=rate_step,
                  soil_radius_cm=.2,soil_depth_cm=.2,radial_interval_cm=.1,
                  depth_interval_cm=.1,initial_solution_p_um=0.),
        SpeciesParams(
            PlantTraits(initial_biomass=1.,initial_c_pool=c,initial_p_pool=p,
                        gamma_c=1.,gamma_p=1.,kappa_c=0.,kappa_p=0.,kleaf=0.,jmax=0.,
                        reproduction_cost_scale=scale),
            FungusTraits(initial_biomass=1.,initial_c_pool=.2 if reciprocal else 0.,
                         initial_p_pool=2. if reciprocal else 0.,gamma_c=1.,gamma_p=1.,
                         kappa_c=0.,kappa_p=0.,jmax=0.),
        ),
    )


def observe(env, actions, repeats=5):
    _,initial = env.reset(jax.random.PRNGKey(0))
    execute = jax.jit(lambda state: env.step_env(jax.random.PRNGKey(1),state,actions))
    started = time.perf_counter()
    result = execute(initial)
    jax.block_until_ready(result)
    first_seconds = time.perf_counter()-started
    started = time.perf_counter()
    for _ in range(repeats):
        result = execute(initial)
    jax.block_until_ready(result)
    warm_seconds = (time.perf_counter()-started)/repeats
    _,final,rewards,_,info = result
    residuals,organisms = {},{}
    for agent in (PLANT,FUNGUS):
        if not bool(info[agent]['allocation_converged']):
            raise RuntimeError('allocation integration failed: exhausted budget or non-finite result')
        traits = getattr(env.species,agent)
        organisms[agent] = dict(growth=float(info[agent]['growth'][0]),
            output=float(rewards[agent]),trade_out=float(info[agent]['trade_out'][0]),
            trade_in=float(info[agent]['trade_in'][0]))
        for element in ('c','p'):
            before = float(getattr(initial,f'{agent}_{element}_pool')[0])
            after = float(getattr(final,f'{agent}_{element}_pool')[0])
            if after < 0.:
                raise AssertionError('negative resource stock')
            incoming = organisms[agent]['trade_in'] if (agent,element) in ((PLANT,'p'),(FUNGUS,'c')) else 0.
            outgoing = organisms[agent]['trade_out'] if (agent,element) in ((PLANT,'c'),(FUNGUS,'p')) else 0.
            used = float(info[agent][f'growth_{element}_used'][0]+info[agent][f'reproduction_{element}'][0])
            residuals[f'{agent}_{element}'] = before-after-used-outgoing+incoming
            if abs(residuals[f'{agent}_{element}']) > 4e-6:
                raise AssertionError(residuals)
            organisms[agent][f'final_{element}'] = after
            organisms[agent][f'reproduction_{element}'] = float(info[agent][f'reproduction_{element}'][0])
            cost = getattr(traits,f'gamma_{element}')*traits.reproduction_cost_scale
            if abs(organisms[agent][f'reproduction_{element}']-cost*organisms[agent]['output']) > 4e-6:
                raise AssertionError('output/export mismatch')
    if bool(jnp.any(info[PLANT]['biomass_guard_contact'])):
        raise AssertionError('qualification contacted numerical biomass guard')
    for a,b in ((PLANT,FUNGUS),(FUNGUS,PLANT)):
        if abs(organisms[a]['trade_out']-organisms[b]['trade_in']) > 1e-7:
            raise AssertionError('unpaired trade')
    plant = organisms[PLANT]
    return dict(environment=asdict(env.config),species=asdict(env.species),
        actions={a:[float(x) for x in actions[a]] for a in (PLANT,FUNGUS)},
        plant_vector=[plant[k] for k in ('growth','output','final_c','final_p','trade_out')],
        organisms=organisms,budget_residuals=residuals,
        allocation_substeps=int(info[PLANT]['allocation_substeps']),
        timing=dict(compile_and_first_seconds=first_seconds,warm_seconds_per_step=warm_seconds,
                    repeats=repeats,protocol='repeat one public step from identical initial state'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/qualification/three-rate-allocation.json'))
    args = parser.parse_args()
    records = []
    for name,c,p,share,scale,trade,reference in CASES:
        reciprocal = name == 'D'
        env = environment(c,p,scale,reciprocal=reciprocal)
        actions = {PLANT:jnp.array([trade,1-share,share]),
                   FUNGUS:jnp.array([1.,.75,.25]) if reciprocal else jnp.zeros(3)}
        record = observe(env,actions)
        errors = [abs(actual-expected) for actual,expected in zip(record['plant_vector'],reference,strict=True)]
        if any(error > max(2e-5,2e-5*abs(expected)) for error,expected in zip(errors,reference,strict=True)):
            raise AssertionError((name,errors))
        records.append(dict(name=name,reference=list(reference),absolute_errors=errors,**record))
    refinement = []
    for rate_step in (.1,.02,.005):
        env = environment(2.,.2,reciprocal=True,rate_step=rate_step)
        record = observe(env,{PLANT:jnp.array([1.,.75,.25]),FUNGUS:jnp.array([1.,.75,.25])})
        refinement.append(dict(rate_step=rate_step,plant_c_error=abs(record['plant_vector'][2]-.3818521706),**record))
    if not (refinement[-1]['plant_c_error'] < refinement[0]['plant_c_error']
            and refinement[-1]['plant_c_error'] < 2e-6):
        raise AssertionError('fixed-command limiting-switch refinement failed')
    fast_env = environment(1.,1.,dt=.05)
    static = evaluate_static_allocation(fast_env,interval_days=.05,
        trade_shares={PLANT:.05,FUNGUS:0.},reproductive_shares={PLANT:0.,FUNGUS:0.},seed=7)
    if static['status'] != 'completed':
        raise AssertionError(static['rejection_reasons'])
    fast = observe(fast_env,{a:jnp.array(static['commanded_rate_actions'][a]) for a in (PLANT,FUNGUS)})
    if abs(fast['plant_vector'][0]-.94905)>3e-6 or abs(fast['plant_vector'][2]-.001)>3e-6:
        raise AssertionError('fast static reference failed')
    coupled_config = EnvConfig(dt=.05,max_steps=20,soil_radius_cm=1.,soil_depth_cm=2.,
                              radial_interval_cm=.5,depth_interval_cm=.5)
    benchmark = benchmark_environment(coupled_config,repeats=5)
    payload = dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version(),platform=platform.platform(),
                     packages={p:version(p) for p in ('jax','jaxlib','numpy','flax','optax','pytest')},
                     devices=[str(d) for d in jax.devices()],jax_enable_x64=jax.config.jax_enable_x64),
        tolerances=dict(reference_absolute=2e-5,reference_relative=2e-5,budget_absolute=4e-6),
        reference_origin='Independent DOP853 illustrative values retained in test_stoichiometric_actions.py; no second biological engine in this runner',
        cases=records,refinement=refinement,fast_static=dict(declaration=static,step=fast),
        coupled_benchmark=dict(environment=asdict(coupled_config),species=asdict(qualification_species(coupled=True)),
                               measurements={k:v for k,v in benchmark.items() if not k.endswith('_annual_projection')}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(f'Qualified six reference cases, fixed-command refinement and fast static allocation: {args.output}')


if __name__ == '__main__':
    main()
