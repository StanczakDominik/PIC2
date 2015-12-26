from shared import *
import fields
import distributions
from scipy.optimize import curve_fit


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
    return field_array#/dx

if __name__=="__main__":
    def line(x, a, b):
        return a*x+b
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    field=grid
    r = distributions.NonRandomUniform(10)
    interpolated_field = InterpolateElectricField(r, field)
    print(interpolated_field)
    ax1.plot(grid, np.zeros_like(grid), "r.")
    ax1.plot(r, np.zeros_like(r), "bo")
    ax1.quiver(r, np.zeros_like(r), np.zeros_like(r), interpolated_field)
    ax2.plot(grid, np.zeros_like(grid), "r.")
    ax2.plot(grid, field, "g-")
    ax2.plot(r, interpolated_field, "bo")
    ax1.set_xlim(0,L)
    ax1.grid(True)
    ax2.grid(True)
    params, covs = curve_fit(line, r, interpolated_field)
    a, b = params
    field_line = line(r, a, b)
    ax2.plot(r, field_line, "bo--")
    print(a, dx)
    plt.show()
    #fit straightline
    #find a
    #compare
