from shared import *
import fields
import distributions
import animations
import particles

Electrons = particles.ParticleSpecies(N, distributions.NonRandomUniform, np.ones, 1, -1)
Positrons = particles.ParticleSpecies(N, distributions.NonRandomUniform, distributions.negative_ones, 1, -1)
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

###Saving data:
#particle position history
#particle velocity history
# charge_history = np.empty((NT, NX))
# electric_field_history = np.empty((NT, NX))
# potential_history = np.empty((NT, NX))
# iterations_history = np.empty(NT)
# l2_diff_history = np.empty((NT, 20000))
