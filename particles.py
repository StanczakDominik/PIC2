from shared import *
import distributions

class ParticleSpecies:
    def __init__(self, N, PositionDistribution, VelocityDistribution, Mass, Charge):
        self.iteration=0
        self.N=N
        self.r = PositionDistribution(N)
        self.CleanupPositions()

        self.v = VelocityDistribution(N)
        self.m = np.ones(N)*Mass
        self.q = np.ones(N)*Charge

        self.r_history = np.zeros((NT+1,N))
        self.v_history = np.zeros((NT+1,N))
        self.r_history[0] = self.r
        self.v_history[0] = self.v

    def SimpleStep(self, dt, ElectricField, ElectricFieldGrid=False):
        return self.r + self.v*dt, self.v+dt*ElectricField(self.r, ElectricFieldGrid)*self.q/self.m

    def CleanupPositions(self):
        self.r = self.r % L

    def CalculateParticleDensity(self):
        """
        A simple counter for particle density per cell.
        """
        density_grid = np.zeros_like(grid)
        indices = (self.r//dx).astype(int)
        # grid_indices = {}
        for particle_index, grid_index in enumerate(indices):
            # if(grid_index not in grid_indices.keys()):
            #     grid_indices[grid_index]=[]
            # grid_indices[grid_index].append(self.r[particle_index])
            density_grid[grid_index]+=1
        # for i in grid_indices:
        #     print (i, grid_indices[i])
        return density_grid

    def CalculateChargeDensity(self):
        """
        A simple model for charge density deposition. Returns a grid-like array
        With the sum of charges in them.
        """
        charge_grid = np.zeros_like(grid)
        indices = (self.r//dx).astype(int)
        # full_positions = indices*dx
        # relative_positions = self.r-full_positions
        #TODO: how to do this via arrays?
        if((indices<0).any()):
            print("DEBUG")
            print(indices<0)
            print(self.r[indices<0])
        for particle_index, grid_index in enumerate(indices):
            charge_grid[grid_index]+=self.q[particle_index]
        return charge_grid


    def Update(self, dt, ElectricField, Step, ElectricFieldGrid=False):
        self.r, self.v = Step(dt, ElectricField, ElectricFieldGrid)
        self.CleanupPositions()
        self.iteration+=1
        self.r_history[self.iteration]=self.r
        self.v_history[self.iteration]=self.v

    def CalculateKineticEnergy(self):
        return 0.5 * self.m * self.v**2
    def TextDiagnostics(self):
        print (self.r)
        print (self.v)
        print (self.m)
        print (self.q)

    def PositionPlotDiagnostics(self):
        plt.hist(self.r, bins=100)
        plt.grid()
        plt.xlabel('x')
        plt.show()

    def PhasePlotDiagnostics(self):
        plt.plot(self.r_history[:self.iteration+1],self.v_history[:self.iteration+1], '-')
        plt.xlim(0,L)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel(r'v_x')
        plt.show()



def test_uniform_particle_distribution_density():
    particles = ParticleSpecies(N, distributions.NonRandomUniform, np.ones, 1, -1)
    density = particles.CalculateParticleDensity()
    average_density = N/L/NX
    average_density_from_grid = np.sum(density)/NX
    print(average_density, average_density_from_grid)

    condition1 = average_density == average_density_from_grid

    goal = average_density*np.ones_like(grid)

    condition2 = (density == goal).all()

    plt.plot(grid, density)
    plt.plot(particles.r, average_density*np.ones_like(particles.r), "g.")
    plt.plot(grid, goal, "ro--")
    plt.show()

    assert condition1, "average density is incorrect"
    assert condition2, "some grid values are off (boundaries?)"

# def test_sin_particle_distribution_density():
#     particles = ParticleSpecies(N, lambda x: L*np.sin(2*np.pi*distributions.NonRandomUniform(x)/L), np.ones, 1, -1)
#     density = particles.CalculateParticleDensity()
#     average_density = N/L/NX
#     average_density_from_grid = np.sum(density)/NX
#     print(average_density, average_density_from_grid)
#
#     condition1 = average_density == average_density_from_grid
#
#     goal = average_density*np.ones_like(grid)
#
#     condition2 = (density == goal).all()
#
#     plt.plot(grid, density)
#     plt.plot(particles.r, average_density*np.ones_like(particles.r), "g.")
#     plt.plot(grid, goal, "ro--")
#     plt.show()
#
#     assert condition1, "average density is incorrect"
#     assert condition2, "some grid values are off (boundaries?)"


def test_uniform_particle_distribution_charge_density():
    particles = ParticleSpecies(N, distributions.NonRandomUniform, np.ones, 1, -1)
    charge = particles.CalculateChargeDensity()
    average_charge = np.sum(particles.q)/L/NX
    average_charge_from_grid = np.sum(charge)/NX

    print(average_charge, average_charge_from_grid)

    condition1 = average_charge == average_charge_from_grid

    goal = average_charge*np.ones_like(grid)

    condition2 = (charge == goal).all()

    plt.plot(grid, charge)
    plt.plot(particles.r, average_charge*np.ones_like(particles.r), "g.")
    plt.plot(grid, goal, "ro--")
    plt.show()

    assert condition1, "average charge is incorrect"
    assert condition2, "some grid values are off (boundaries?)"
