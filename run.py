import numpy as np

L=1
NX = 32
grid, dx = np.linspace(0,L,NX, retstep=True)

T = 10
NT=101
timegrid, dt = np.linspace(0,T,NT, retstep=True)



class ParticleSpecies:
    def __init__(self, N, PositionDistribution, VelocityDistribution, Mass, Charge):
        self.N=N
        self.r = PositionDistribution(N)
        self.v = VelocityDistribution(N)
        self.m = np.ones(N)*Mass
        self.q = np.ones(N)*Charge

    def Acceleration(self, ElectricField):
        return ElectricField(self.r)*self.q/self.m

    def Simple_step(self, dt, ElectricField):
        return self.r + self.v*dt, self.v+self.Acceleration(ElectricField)*dt

    def Update(self, dt, ElectricField, Step):
        self.r, self.v = Step(dt, ElectricField)

    def TextDiagnostics(self):
        print (self.r)
        print (self.v)
        print (self.m)
        print (self.q)
def UniformElectricField(r):
    """
    Takes particle positions and returns their E acceleration
    Simple model assumes acceleration by a constant factor
    """
    return np.ones_like(r)

Electrons = ParticleSpecies(10, np.ones, np.ones, 1, -1)
Positrons = ParticleSpecies(10, np.ones, np.ones, 1, 1)
Species = [Electrons, Positrons]
for species in Species:
    species.TextDiagnostics()
    species.Update(dt, UniformElectricField, species.Simple_step)
    species.TextDiagnostics()
