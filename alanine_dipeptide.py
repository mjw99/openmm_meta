from openmm import *
from openmm.app import *
from simtk.unit import *

import matplotlib.pyplot as plot
import numpy as np


pdb = PDBFile('alanine-dipeptide.pdb')
# Use AMBER ff14SB
forceField = ForceField('amber14-all.xml','./amber14/tip3p.xml')

modeller =Modeller(pdb.topology, pdb.positions)

# Add TIP3P solvent
modeller.addSolvent(forceField, model='tip3p', padding=10*angstrom)

PDBFile.writeFile(modeller.topology,
                      modeller.positions,
                      open('alanine-dipeptide-explicit.pdb', 'w'),
                      keepIds=True)


# Create a System for alanine dipeptide in water.
# https://github.com/openmm/openmm/issues/4048

pdb = PDBFile('alanine-dipeptide-explicit.pdb')

#for atom in pdb.topology.atoms():
#    print(atom)

forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds)

# Define collective variables for phi and psi.

# https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
cv1 = CustomTorsionForce('theta')
cv1.addTorsion(1, 6, 8, 14)
phi = BiasVariable(cv1, -np.pi, np.pi, 0.5, True)

cv2 = CustomTorsionForce('theta')
cv2.addTorsion(6, 8, 14, 16)
psi = BiasVariable(cv2, -np.pi, np.pi, 0.5, True)

# Set up the simulation.

#meta = Metadynamics(system, [phi, psi], 300.0*kelvin, 1000.0*kelvin, 1.0*kilojoules_per_mole, 100)

# https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.Metadynamics.html
meta = Metadynamics(system=system,
                        variables=[phi, psi],
                        temperature=300.0,
                        biasFactor= 5.0,
                        height=1 * kilocalories_per_mole,
                        frequency=100,
                        saveFrequency=100,
                        biasDir='./')



integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

# Run the simulation and plot the free energy landscape.

meta.step(simulation, 50000)
#meta.step(simulation, 500000)
#meta.step(simulation, 5000000)

print(meta.getFreeEnergy())

c = plot.imshow(meta.getFreeEnergy())
plot.colorbar(c)
plot.savefig('alanine.png')

