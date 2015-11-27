import numpy as np
import matplotlib.pyplot as plt
L=1
NX = 32
grid, dx = np.linspace(0,L,NX, retstep=True, endpoint=False)

T = 5
NT = 1001
timegrid, dt = np.linspace(0,T,NT, retstep=True)

charge_grid = np.zeros_like(grid)
charge_history = np.empty((NT, NX))
electric_field_grid = np.zeros_like(grid)
electric_field_history = np.empty((NT, NX))



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
    def Acceleration(self, ElectricField):
        return ElectricField(self.r)*self.q/self.m

    def SimpleStep(self, dt, ElectricField):
        return self.r + self.v*dt, self.v+self.Acceleration(ElectricField)*dt

    def CleanupPositions(self):
        self.r = self.r % L

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
        for particle_index, grid_index in enumerate(indices):
            charge_grid[grid_index]+=self.q[particle_index]
        charge_grid/=dx
        return charge_grid


    def Update(self, dt, ElectricField, Step):
        self.r, self.v = Step(dt, ElectricField)
        self.CleanupPositions()
        self.iteration+=1
        self.r_history[self.iteration]=self.r
        self.v_history[self.iteration]=self.v
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

def RandomGaussian(N):
    return np.random.normal(0, 1, N)

def NoElectricField(r):
    return np.zeros_like(r)

def UniformElectricField(r):
    """
    Takes particle positions and returns their E acceleration
    Simple model assumes acceleration by a constant factor
    """
    return np.ones_like(r)

def InterpolateElectricField(r):
    field_array = np.zeros_like(r)

    indices = (r//dx).astype(int)
    full_positions = indices*dx
    relative_positions = r-full_positions

    #TODO: arrayize this
    for particle_index, grid_index in indices(grid):
        if grid_index==0:
            pass
        elif grid_index==31:
            pass
        else:
            field_array[particle_index] = relative_positions[particle_index] *\
                                    electric_field_grid[grid_index] +\
                                    (relative_positions[particle_index]-dx)*\
                                    electric_field_grid[grid_index+1]
    return field_array

def RampElectricField(r):
    return -(r-L/2)

def SinElectricField(r):
    return -np.sin((r-L/2))

N=200
Electrons = ParticleSpecies(N, RandomGaussian, RandomGaussian, 1, -1)
Positrons = ParticleSpecies(N, RandomGaussian, RandomGaussian, 1, 1)
Species = [Positrons, Electrons]
for i, t in enumerate(timegrid):
    charge_grid[:]=0
    for species in Species:
        charge_grid += species.CalculateChargeDensity()
        species.Update(dt, SinElectricField, species.SimpleStep)
    charge_history[i] = charge_grid
for species in Species:
    species.PhasePlotDiagnostics()

for index, chargesnapshot in enumerate(charge_history[:]):
    plt.plot(grid, chargesnapshot)
plt.show()
