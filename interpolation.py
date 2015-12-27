from shared import *
import fields
import distributions
from scipy.optimize import curve_fit


# @jit(nopython=True)
def InterpolateElectricField(r, electric_field_grid):
    # r = r % L
    N = len(r)
    field_array=np.zeros(N)
    # print(grid, dx)
    for i in range(N):
        ind = int(r[i]//dx)
        # print(r[i], dx, r[i]/dx, r[i]//dx)
        if ind == NX-1:
            # print(ind, i, r[i], grid[ind], grid[ind])
            # TODO: tu jest problem!!!!!
            field_array[i] += (r[i]-grid[ind]) * electric_field_grid[ind]
            field_array[i] += (L-r[i]) * electric_field_grid[0]
        else:
            field_array[i] += (r[i]-grid[ind]) * electric_field_grid[ind]
            field_array[i] += (grid[ind+1]-r[i]) * electric_field_grid[ind+1]
    return field_array/dx

def test_sin_field_interpolation():
    def sin(x, a, b,c):
        return a*np.sin(b*x+c)
    field=np.sin(2*np.pi*grid/L)
    r = distributions.NonRandomUniform(1000)
    interpolated_field = InterpolateElectricField(r, field)
    params, covs = curve_fit(sin, r, interpolated_field)
    a, b, c = params
    da, db, dc = np.sqrt(np.diag(covs**2))
    total_rel_error = np.sqrt((da/a)**2+(db/b)**2+(dc/c)**2)
    print(total_rel_error)
    condition = total_rel_error < 0.001
    if not condition:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
        ax1.plot(grid, np.zeros_like(grid), "r.")
        ax1.plot(r, np.zeros_like(r), "bo")
        ax1.quiver(r, np.zeros_like(r), np.zeros_like(r), interpolated_field, interpolated_field)
        ax2.plot(grid, np.zeros_like(grid), "r.")
        ax2.plot(grid, field, "g-")
        ax2.plot(r, interpolated_field, "bo--")
        ax1.set_xlim(0,L)
        ax1.grid(True)
        ax2.grid(True)
        field_sin = sin(r, a, b, c)
        ax2.plot(r, field_sin, "g--")
        print(a, b, c)
        print(da, db, dc)
        plt.show()
    assert condition

def test_line_field_interpolation():
    def line(x, a, b):
        return a*x+b
    field=grid-L/2
    r = distributions.NonRandomUniform(1000)
    interpolated_field = InterpolateElectricField(r, field)
    params, covs = curve_fit(line, r, interpolated_field)
    a, b = params
    da, db = np.sqrt(np.diag(covs**2))
    total_rel_error = np.sqrt((da/a)**2+(db/b)**2)
    print(total_rel_error)
    condition = total_rel_error < 0.0000001
    if not condition:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
        ax1.plot(grid, np.zeros_like(grid), "r.")
        ax1.plot(r, np.zeros_like(r), "bo")
        ax1.quiver(r, np.zeros_like(r), np.zeros_like(r), interpolated_field, interpolated_field)
        ax2.plot(grid, np.zeros_like(grid), "r.")
        ax2.plot(grid, field, "g-")
        ax2.plot(r, interpolated_field, "bo--")
        ax1.set_xlim(0,L)
        ax1.grid(True)
        ax2.grid(True)
        field_line = line(r, a, b)
        ax2.plot(r, field_line, "g--")
        print(a, b)
        print(da, db)
        plt.show()
    assert condition
if __name__=="__main__":
    pass
