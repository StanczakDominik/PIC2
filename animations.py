from shared import *

def AnimatedPhasePlotDiagnostics(species):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    points = [0]*len(species)
    for i, specie in enumerate(species):
        points[i], = ax1.plot(specie.r_history[0], specie.v_history[0], "o", label=i, alpha=0.5)
    field, = ax2.plot(grid, electric_field_history[0], "co-", label="Electric field")
    potential, = ax2.plot(grid, potential_history[0], "yo-", label="Potential")
    charge, = ax2.plot(grid, charge_history[0], "go-", label="Charge density")
    ax1.set_xlim(0,L)
    ax1.set_ylim(-250*L,250*L)
    ax1.grid(True)
    ax2.grid(True)

    ax1.set_xlabel("Position (along grid)")
    ax1.set_ylabel("Particle velocity")
    ax2.set_ylabel("Grid property magnitude")
    ax1.set_title("Particles")
    ax2.set_title("Fields")

    # gridpoints1, = ax1.plot(grid, np.zeros_like(grid), "ro", label="Grid")
    # gridpoints2, = ax2.plot(grid, np.zeros_like(grid), "ro", label="Grid")
    def animate(i):
        for j, specie in enumerate(species):
            points[j].set_xdata(specie.r_history[i])
            points[j].set_ydata(specie.v_history[i])
        field.set_ydata(electric_field_history[i])
        charge.set_ydata(charge_history[i])
        potential.set_ydata(potential_history[i])
        # for ax in (ax1, ax2):
        #     ax.relim()
        #     ax.autoscale_view()
        return points, field, charge, potential, ax1, ax2
    def init():
        for j, specie in enumerate(species):
            points[j].set_xdata(specie.r_history[0])
            points[j].set_ydata(specie.v_history[0])
        field.set_ydata(electric_field_history[0])
        charge.set_ydata(charge_history[0])
        potential.set_ydata(potential_history[0])
        # for ax in (ax1, ax2):
        #     ax.relim()
        #     ax.autoscale_view()
        return points, field, charge, potential, ax1, ax2
    plt.legend()
    print("Ready to start animation")
    ani = animation.FuncAnimation(fig, animate, np.arange(1,NT), init_func=init, interval=25, blit=False, repeat=True)
    ani.save("./Videos/" + timestamp + ".mp4", writer='mencoder', fps=30)
    print("Saved animation")
    plt.show()
