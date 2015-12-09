import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba
from numba import jit

L=1
NX = 32
grid, dx = np.linspace(0,L,NX, retstep=True, endpoint=False)

SOR_omega = 2/(1+np.pi/NX)
SOR_L2_target = 1e-8

T = 0.1
NT = 1001
timegrid, dt = np.linspace(0,T,NT, retstep=True)

field_scale=5000

charge_grid = np.zeros_like(grid)
charge_history = np.empty((NT, NX))
electric_field_grid = np.zeros_like(grid)
electric_field_history = np.empty((NT, NX))
potential_history = np.empty((NT, NX))
iterations_history = np.empty(NT)
l2_diff_history = np.empty((NT, 20000))

# plt.ion()

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
        if((indices<0).any()):
            print("DEBUG")
            print(indices<0)
            print(self.r[indices<0])
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

def RandomUniform(N):
    return np.random.random(N)

def RandomUniformVel(N):
    return RandomUniform(N)-0.5

def NoElectricField(r):
    return np.zeros_like(r)

def UniformElectricField(r):
    """
    Takes particle positions and returns their E acceleration
    Simple model assumes acceleration by a constant factor
    """
    return np.ones_like(r)

@jit(nopython=True)
def PoissonSolver(charge_grid, current_potential):
    """
    Takes the previous potential as candidate for calculating the new ones_like
    """
    iterations=0
    iteration_difference = SOR_L2_target + 1
    denominator = 0.0
    l2_diff = np.zeros(20000)

    while iteration_difference > SOR_L2_target:
        old_potential = current_potential.copy()
        current_potential[0] = (1-SOR_omega)*current_potential[0] + SOR_omega*0.5*(charge_grid[0]*dx*dx + charge_grid[-1] + charge_grid[1])
        for i in range(1,NX-1):
            current_potential[i] = (1-SOR_omega)*current_potential[i] + SOR_omega*0.5*(charge_grid[i]*dx*dx + charge_grid[i-1] + charge_grid[i+1])
        current_potential[-1] = (1-SOR_omega)*current_potential[-1] + SOR_omega*0.5*(charge_grid[-1]*dx*dx + charge_grid[-2] + charge_grid[0])

        iteration_difference = 0.0
        denominator = 0.0

        for i in range(NX):
            iteration_difference += (current_potential[i]-old_potential[i])**2
            denominator += old_potential[i]**2

        iteration_difference /= denominator
        iteration_difference = iteration_difference**2
        l2_diff[iterations] = iteration_difference
        iterations += 1

    return current_potential, iterations, l2_diff

def FieldCalculation(potential_grid):
    field = -np.gradient(potential_grid, dx, edge_order=2)
    return field

def RampElectricField(r):
    return -(r-L/2)

def SinElectricField(r):
    return -np.sin((r-L/2))

def InterpolateElectricField(r):
    field_array = np.zeros_like(r)

    indices = (r//dx).astype(int)
    full_positions = indices*dx
    relative_positions = r-full_positions

    #TODO: arrayize this
    for particle_index, grid_index in enumerate(indices):
        if grid_index==0:
            pass
            # field_array[particle_index] = relative_positions[particle_index] *\
            #                         electric_field_grid[-1]/dx +\
            #                         (relative_positions[particle_index]-dx)/dx*\
            #                         electric_field_grid[grid_index+1]
        elif grid_index==31:
            pass
            # field_array[particle_index] = relative_positions[particle_index] *\
            #                         electric_field_grid[grid_index]/dx +\
            #                         (relative_positions[particle_index]-dx)/dx*\
            #                         electric_field_grid[grid_index+1]
        else:
            #TODO: this is probably incorrect somehow
            field_array[particle_index] = relative_positions[particle_index] *\
                                    electric_field_grid[grid_index]/dx +\
                                    (relative_positions[particle_index]-dx)/dx*\
                                    electric_field_grid[grid_index+1]
    return field_array #TODO: minus tutaj?

def NonRandomUniform(N):
    return np.linspace(0,1,N)



def negative_ones(N):
    return -np.ones(N)

def AnimatedPhasePlotDiagnostics(species):
    fig, ax = plt.subplots()
    points = [0]*len(species)
    for i, specie in enumerate(species):
        points[i], = ax.plot(specie.r_history[0], specie.v_history[0], "o", label=i)
    field, = ax.plot(grid, electric_field_history[0], "co-", label="Electric field")
    potential, = ax.plot(grid, potential_history[0], "yo-", label="Potential")
    charge, = ax.plot(grid, charge_history[0], "go-", label="Charge density")
    plt.xlim(0,L)
    plt.ylim(-10,10)
    plt.grid()
    gridpoints, = ax.plot(grid, np.zeros_like(grid), "ro", label="Grid")
    def animate(i):
        for j, specie in enumerate(species):
            points[j].set_xdata(specie.r_history[i])
            points[j].set_ydata(specie.v_history[i])
        field.set_ydata(electric_field_history[i]/field_scale)
        charge.set_ydata(charge_history[i]/field_scale)
        potential.set_ydata(potential_history[i]/field_scale)
        return points, field, charge, potential,
    def init():
        for j, specie in enumerate(species):
            points[j].set_xdata(specie.r_history[0])
            points[j].set_ydata(specie.v_history[0])
        field.set_ydata(electric_field_history[0]/field_scale)
        charge.set_ydata(charge_history[0]/field_scale)
        potential.set_ydata(potential_history[0]/field_scale)
        return points, field, charge, potential,
    plt.legend()
    ani = animation.FuncAnimation(fig, animate, np.arange(1,NT), init_func=init, interval=25, blit=False)
    plt.show()

def AnimatedFieldDiagnostics():
    fig, ax = plt.subplots()
    field, = ax.plot(grid, electric_field_history[0], "co-", label="Electric field")
    potential, = ax.plot(grid, potential_history[0], "yo-", label="Potential")
    charge, = ax.plot(grid, charge_history[0], "go-", label="Charge density")
    plt.xlim(0,L)
    plt.ylim(-10,10)
    plt.grid()
    gridpoints, = ax.plot(grid, np.zeros_like(grid), "ro", label="Grid")
    def animate(i):
        field.set_ydata(electric_field_history[i]/field_scale)
        charge.set_ydata(charge_history[i]/field_scale)
        potential.set_ydata(potential_history[i]/field_scale)
        return field, charge, potential,
    def init():
        field.set_ydata(electric_field_history[0]/field_scale)
        charge.set_ydata(charge_history[0]/field_scale)
        potential.set_ydata(potential_history[0]/field_scale)
        return field, charge, potential
    plt.legend()
    ani = animation.FuncAnimation(fig, animate, np.arange(1,NT), init_func=init, interval=25, blit=False)
    plt.show()


N=3200
Electrons = ParticleSpecies(N, NonRandomUniform, RandomUniformVel, 1, -1)
Positrons = ParticleSpecies(N, NonRandomUniform, RandomUniformVel, 1, -1)
Species = [Positrons, Electrons]


potential_grid = np.ones_like(grid)

for i, t in enumerate(timegrid):
    charge_grid[:]=0
    for species in Species:
        charge_grid += species.CalculateChargeDensity()
    charge_history[i] = charge_grid
    potential_grid, iterations, l2_diff = PoissonSolver(charge_grid, potential_grid)
    potential_history[i] = potential_grid
    electric_field_grid = FieldCalculation(potential_grid)
    electric_field_history[i] = electric_field_grid
    iterations_history[i] = iterations
    l2_diff_history[i] = l2_diff

    for species in Species:
        species.Update(dt, SinElectricField , species.SimpleStep)



# AnimatedFieldDiagnostics()
AnimatedPhasePlotDiagnostics(Species)
