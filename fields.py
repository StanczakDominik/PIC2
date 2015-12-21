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
def TunedSORPoissonSolver(charge_grid, current_potential):
    """
    Solves the Poisson equation via relaxing the previous potential via tuned SOR
    (see Practical Numerical Methods part 5)
    Takes the previous potential as candidate for calculating the new ones_like
    """
    iterations=0
    iteration_difference = SOR_L2_target + 1
    denominator = 0.0
    l2_diff = np.zeros(20000)

    while iteration_difference > SOR_L2_target:
        old_potential = current_potential.copy()
        # current_potential[0] = (1-SOR_omega)*current_potential[0] + SOR_omega*0.5*(charge_grid[0]*dx*dx + charge_grid[-1] + charge_grid[1])
        for i in range(0,NX-1):
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

def L2_rel_error(p, pn):
    ''' Compute the relative L2 norm of the difference
    Parameters:
    ----------
    p : array of float
        array 1
    pn: array of float
        array 2
    Returns:
    -------
    Relative L2 norm of the difference
    '''
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))

# @jit(nopython=True)
def ConjugateGradientPoissonSolver(charge_grid, current_potential):
    r = np.zeros(NX) # residual
    Ad = np.zeros(NX)

    L2_norm = 1
    iterations = 0
    L2_conv = []

    r[0] = charge_grid[0]*dx*dx + 2*current_potential[0] - current_potential[1] - current_potential[-1]
    r[1:-1] = charge_grid[1:-1]*dx*dx + 2*current_potential[1:-1] - current_potential[2:] - current_potential[:-2]
    r[-1] = charge_grid[-1]*dx*dx + 2*current_potential[-1] - current_potential[-2] - current_potential[0]
    d = r.copy()
    rho = np.sum(r*r)
    Ad[0] = -2*d[0]+d[1]+d[-1]
    Ad[1:-1] = -2*d[1:-1]+d[2:]+d[:-2]
    Ad[-1] = -2*d[-1]+d[0]+d[-2]

    sigma = np.sum(d*Ad)

    while L2_norm > SOR_L2_target:
        # not sure this is helpful
        pk = current_potential.copy()
        rk = r.copy()
        dk = d.copy()

        alpha = rho/sigma

        current_potential += alpha*dk
        # current_potential = pk + alpha*dk
        # r = rk - alpha*Ad
        r -= alpha*Ad

        rhop1 = np.sum(r*r)
        beta = rhop1/rho
        rho = rhop1

        d = r + beta*dk
        Ad[0] = -2*d[0] + d[2] + d[-2]
        Ad[1:-1] = -2*d[1:-1] + d[2:] + d[:-2]
        Ad[-1] = -2*d[-1] + d[2] + d[-2]
        sigma = np.sum(d*Ad)

        L2_norm=L2_rel_error(pk, current_potential)
        iterations += 1
        L2_conv.append(L2_norm)
        # print(iterations, L2_norm)
    return current_potential, iterations, L2_conv

if __name__=="__main__":
    charge = np.sin(grid/np.max(grid)*np.pi)
    zeroes = np.ones_like(charge)
    TSp, TSiters, TSconv = TunedSORPoissonSolver(charge,zeroes)
    print(TSp, TSiters)
    CGp, CGiters, CGconv = ConjugateGradientPoissonSolver(charge, zeroes)
    print(CGp, CGiters)

    Difference = CGp-TSp
    print(L2_rel_error(CGp, TSp))
    plt.figure(1)
    plt.plot(grid, charge, label="Charge")
    plt.plot(grid, TSp, label="Tuned SOR")
    plt.plot(grid, CGp, label="Conjugate Gradient")
    plt.plot(grid, Difference, label="Difference")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(TSconv, label="Tuned SOR convergence")
    plt.plot(CGconv, label="CG convergence")
    plt.legend()
    plt.show()



@jit(nopython=True)
def FieldCalculation(potential_grid):
    field = np.zeros_like(potential_grid)
    for i in range(0, NX-1):
        field[i] = (potential_grid[i+1]-potential_grid[i-1])/(2*dx)

    field[NX-1]=(potential_grid[0]-potential_grid[NX-2])/(2*dx)
    # field = -np.gradient(potential_grid, dx, edge_order=2)
    return -field



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
