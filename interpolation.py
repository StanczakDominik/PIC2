from shared import *
import fields

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


TestElectricField = fields.RampElectricField
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
