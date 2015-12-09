from shared import *
import fields
import distributions
import animations
import particles

Electrons = particles.ParticleSpecies(N, distributions.NonRandomUniform, distributions.RandomUniformVel, 1, -1)
Positrons = particles.ParticleSpecies(N, distributions.NonRandomUniform, distributions.RandomUniformVel, 1, -1)
Species = [Positrons, Electrons]
potential_grid = np.ones_like(grid)
for i, t in enumerate(timegrid):
    charge_grid[:]=0
    for species in Species:
        charge_grid += species.CalculateChargeDensity()
    charge_history[i] = charge_grid
    potential_grid, iterations, l2_diff = fields.PoissonSolver(charge_grid, potential_grid)
    potential_history[i] = potential_grid
    electric_field_grid = fields.FieldCalculation(potential_grid)
    electric_field_history[i] = electric_field_grid
    iterations_history[i] = iterations
    l2_diff_history[i] = l2_diff

    for species in Species:
        species.Update(dt, fields.InterpolateElectricField, species.SimpleStep, electric_field_grid)
print("Finished loop...")

animations.AnimatedPhasePlotDiagnostics(Species)
