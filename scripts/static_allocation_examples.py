"""Print a compact reproducible three-rate static allocation screen as JSON."""

from dataclasses import asdict
from contextlib import redirect_stdout
import json
import sys

# Optional JaxMARL backend imports print diagnostics; keep stdout valid JSON.
with redirect_stdout(sys.stderr):
    from mycormarl.environments.base_mycor import BaseMycorMarl
    from mycormarl.fungus.traits import FungusTraits
    from mycormarl.params import EnvConfig, SpeciesParams
    from mycormarl.plant.traits import PlantTraits
    from mycormarl.static_rollout import evaluate_static_allocation


def main():
    cases = []
    for name, trade, reproduction, scale in (
        ('growth', 0., 0., 1.),
        ('reproduction', 0., 1., 1.),
        ('simultaneous_trade_growth', .05, 0., 1.),
        ('simultaneous_trade_scaled_reproduction', .05, .25, 2.),
    ):
        env = BaseMycorMarl(
            EnvConfig(dt=.05, max_steps=1, soil_radius_cm=.2, soil_depth_cm=.2,
                      radial_interval_cm=.1, depth_interval_cm=.1,
                      initial_solution_p_um=0.),
            SpeciesParams(
                PlantTraits(initial_biomass=1., initial_c_pool=1., initial_p_pool=1.,
                            gamma_c=1., gamma_p=1., kappa_c=0., kappa_p=0.,
                            kleaf=0., jmax=0., reproduction_cost_scale=scale),
                FungusTraits(initial_biomass=1., initial_c_pool=0., initial_p_pool=0.,
                             gamma_c=1., gamma_p=1., kappa_c=0., kappa_p=0., jmax=0.),
            ),
        )
        result = evaluate_static_allocation(
            env, interval_days=.05, trade_shares={'plant':trade,'fungus':0.},
            reproductive_shares={'plant':reproduction,'fungus':0.}, seed=7,
        )
        if result['status'] != 'completed':
            raise RuntimeError(result['rejection_reasons'])
        cases.append(dict(name=name, environment=asdict(env.config),
                          species=asdict(env.species), result=result))
    print(json.dumps(dict(protocol='rate-static-allocation-v1', cases=cases),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
