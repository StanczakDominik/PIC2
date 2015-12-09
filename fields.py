from shared import *

def NoElectricField(r, electric_field_grid=False):
    return np.zeros_like(r)

def UniformElectricField(r, electric_field_grid=False):
    """
    Takes particle positions and returns their E acceleration
    Simple model assumes acceleration by a constant factor
    """
    return np.ones_like(r)

def RampElectricField(r, electric_field_grid=False):
    return -(r-L/2)

def SinElectricField(r, electric_field_grid=False):
    return -np.sin((r-L/2))

@jit(nopython=True)
def InterpolateElectricField(r, electric_field_grid):
    """Takes the position array of particles
    Interpolates
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
    """
    r = r % 1
    N = len(r)
    field_array=np.zeros(N)
    for i in range(N):
        ind = int(r[i]//dx)
        if ind == NX-1:
            field_array[i] += (r[i]-grid[ind]) * electric_field_grid[ind]
            field_array[i] += (L-r[i]) * electric_field_grid[0]
        else:
            field_array[i] += (r[i]-grid[ind]) * electric_field_grid[ind]
            field_array[i] += (grid[ind+1]-r[i]) * electric_field_grid[ind+1]
    return field_array/dx

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



TestElectricField = RampElectricField
def test_at_regular_gridpoint_InterpolateElectricField():
    electric_field_grid = TestElectricField(grid)
    r = np.asarray([dx])
    test_field = InterpolateElectricField(r, electric_field_grid)
    target = electric_field_grid[1]
    print(r, test_field, target)
    assert test_field==target

def test_at_half_interval_InterpolateElectricField():
    electric_field_grid = TestElectricField(grid)
    r = np.asarray([1.5*dx])
    test_field = InterpolateElectricField(r, electric_field_grid)
    target = (electric_field_grid[1]+electric_field_grid[2])/2
    print(r, test_field, target)
    assert test_field == target

def test_at_175_interval_InterpolateElectricField():
    electric_field_grid = TestElectricField(grid)
    r = np.asarray([1.75*dx])
    test_field = InterpolateElectricField(r, electric_field_grid)
    target = 0.25*electric_field_grid[1]+0.75*electric_field_grid[2]
    print(r, test_field, target)
    assert test_field == target
def test_at_zero_InterpolateElectricField():
    electric_field_grid = TestElectricField(grid)
    r = np.asarray([0])
    test_field = InterpolateElectricField(r, electric_field_grid)
    target = electric_field_grid[0]
    print(r, test_field, target)
    assert test_field == target
def test_at_last_InterpolateElectricField():
    electric_field_grid = TestElectricField(grid)
    r = np.asarray([L])
    test_field = InterpolateElectricField(r, electric_field_grid)
    target = electric_field_grid[0]
    print(r, test_field, target)
    assert test_field == target
