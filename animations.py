from shared import *

def AnimatedPhasePlotDiagnostics(species):
    fig, ax = plt.subplots()
    points = [0]*len(species)
    for i, specie in enumerate(species):
        points[i], = ax.plot(specie.r_history[0], specie.v_history[0], "o", label=i, alpha=0.5)
    field, = ax.plot(grid, electric_field_history[0], "co-", label="Electric field")
    potential, = ax.plot(grid, potential_history[0], "yo-", label="Potential")
    charge, = ax.plot(grid, charge_history[0], "go-", label="Charge density")
    plt.xlim(0,L)
    plt.ylim(-250,250)
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
    print("Ready to start animation")
    ani = animation.FuncAnimation(fig, animate, np.arange(1,NT), init_func=init, interval=25, blit=False, repeat=True)
    ani.save("./Videos/" + time.strftime("%y-%m-%d_%H-%M-%S") + ".mp4", writer='mencoder', fps=30)
    print("Saved animation")
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
    ani = animation.FuncAnimation(fig, animate, np.arange(1,NT), init_func=init, interval=25, blit=False, repeat=True)
    ani.save('test.mp4')
    plt.show()
