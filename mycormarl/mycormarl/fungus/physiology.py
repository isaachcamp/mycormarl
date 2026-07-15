from __future__ import annotations


def fungal_maintenance_demand(biomass, kappa_c, kappa_p, dt):
    return kappa_c * biomass * dt, kappa_p * biomass * dt
