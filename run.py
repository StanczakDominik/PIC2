from shared import *
import fields
import distributions
import animations
import particles
import interpolation

parser = argparse.ArgumentParser()
parser.add_argument("--save", help="save animation", action="store_true")
args=parser.parse_args()
save_animation=args.save

Electrons = particles.ParticleSpecies(N, distributions.NonRandomUniform, np.ones, 1, -1)
Positrons = particles.ParticleSpecies(N, distributions.NonRandomUniform, distributions.negative_ones, 1, -1)
Species = [Positrons, Electrons]
potential_grid = np.ones_like(grid)
for i, t in enumerate(timegrid):
    charge_grid[:]=0
    for species in Species:
        charge_grid += species.CalculateChargeDensity()
    charge_history[i] = charge_grid
    potential_grid, iterations, l2_diff = fields.TunedSORPoissonSolver(charge_grid, potential_grid)
    potential_history[i] = potential_grid
    electric_field_grid = fields.FieldCalculation(potential_grid)
    electric_field_history[i] = electric_field_grid
    iterations_history[i] = iterations
    l2_diff_history[i] = l2_diff

    for species in Species:
        species.Update(dt, interpolation.InterpolateElectricField, species.SimpleStep, electric_field_grid)
    sys.stdout.write("\rSimulation progress %d%%" % (100*i/len(timegrid)))
    sys.stdout.flush()
print("\rFinished loop...                  ")

animations.AnimatedPhasePlotDiagnostics(Species, save_animation)
