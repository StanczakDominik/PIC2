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
        for i in range(0,NX):
            current_potential[i] = (1-SOR_omega)*current_potential[i] + SOR_omega*0.5*(charge_grid[i]*dx*dx + current_potential[i-1] + current_potential[i+1])

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

#TODO: this is broken somehow
# @jit(nopython=True)
def ConjugateGradientPoissonSolver(charge_grid, current_potential):
    r = np.zeros(NX) # residual
    Ad = np.zeros(NX)

    L2_norm = 1
    iterations = 0
    L2_conv = []

    # r[0] = charge_grid[0]*dx*dx + 2*current_potential[0] - current_potential[1] - current_potential[-1]
    r[1:-1] = charge_grid[1:-1]*dx*dx + 2*current_potential[1:-1] - current_potential[2:] - current_potential[:-2]
    # r[-1] = charge_grid[-1]*dx*dx + 2*current_potential[-1] - current_potential[-2] - current_potential[0]
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
        # Ad[0] = -2*d[0] + d[2] + d[-2]
        Ad[1:-1] = -2*d[1:-1] + d[2:] + d[:-2]
        # Ad[-1] = -2*d[-1] + d[2] + d[-2]
        sigma = np.sum(d*Ad)

        L2_norm=L2_rel_error(pk, current_potential)
        iterations += 1
        L2_conv.append(L2_norm)
        # print(iterations, L2_norm)
    return current_potential, iterations, L2_conv


# @jit(nopython=True)
def FieldCalculation(potential_grid):
    field = np.zeros_like(potential_grid)
    for i in range(0, NX-1):
        field[i] = (-potential_grid[i+1]+potential_grid[i-1])/(2*dx)
    field[NX-1]=(-potential_grid[0]+potential_grid[NX-2])/(2*dx)
    return field

if __name__=="__main__":
    charge = np.sin(grid*2*np.pi/L)
    zeroes = np.ones_like(charge)
    TSp, TSiters, TSconv = TunedSORPoissonSolver(charge,zeroes)
    print(TSp, TSiters)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
    ax1.plot(grid, np.zeros_like(grid), "ro")
    ax1.plot(grid, charge, label="Charge")
    ax2.plot(grid, TSp, label="Tuned SOR")
    ax2.plot(grid, np.zeros_like(grid), "ro")
    ax3.plot(grid, FieldCalculation(TSp), label="TSP field")
    ax3.plot(grid, np.zeros_like(grid), "ro")
    ax2.legend()
    ax3.legend()
    plt.show()

def test_gradient_ramp_potential():
    #this test fails at the boundary
    # a ramp potential would require a x**3 charge distribution
    potential_grid = grid-0.5*L
    electric_field = FieldCalculation(potential_grid)
    goal = -np.ones_like(grid)

    condition = (electric_field[1:-1]==goal[1:-1]).all()
    if not condition:
        plt.title("ramp potential electric field")
        plt.plot(grid, potential_grid, "go-", label="potential")
        plt.plot(grid, electric_field, "bo-", label="field")
        plt.plot(grid, goal, "ro--", label="goal")
        plt.legend()
        plt.show()
    assert condition, "not all fields are proportional to spatial gradient"

def test_gradient_constant_potential():
    potential_grid = np.ones_like(grid)
    electric_field = FieldCalculation(potential_grid)

    condition = (electric_field==0).all()
    if not condition:
        plt.title("constant potential electric field")
        plt.plot(grid, electric_field)
        plt.show()
    assert condition, "not all fields are zero"

def test_gradient_sin_potential():
    potential_grid = np.sin(grid*2*np.pi/L)
    electric_field = FieldCalculation(potential_grid)
    goal = -2*np.pi/L*np.cos(grid*2*np.pi/L)

    error = L2_rel_error(goal, electric_field)
    print(error)
    condition = error < 0.05
    if not condition:
        plt.title("ramp potential electric field")
        plt.plot(grid, potential_grid, "go-", label="potential")
        plt.plot(grid, electric_field, "bo-", label="field")
        plt.plot(grid, goal, "ro--", label="goal")
        plt.legend()
        plt.show()
    assert condition, "not all fields are proportional to spatial gradient"
    return error
