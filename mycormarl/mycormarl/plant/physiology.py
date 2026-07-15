from __future__ import annotations


def photosynthesis(biomass, kleaf, amass, dt):
    return kleaf * biomass * amass * dt

def plant_maintenance_demand(biomass, kappa_c, kappa_p, dt):
    return kappa_c * biomass * dt, kappa_p * biomass * dt
