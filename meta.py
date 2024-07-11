import openmm.app as app
import openmm as mm
import simtk.unit as unit

import numpy as np

import mdtraj.reporters
import mdtraj as md

import matplotlib.pyplot as plot

platform = mm.Platform.getPlatformByName("CUDA")
platformProperties = {}


pdb = app.PDBFile("production_final.pdb")
forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')


# Get the index of protein atoms only,
# to enable stripping of water and counter ions in the production trajectory
traj = mdtraj.load('production_final.pdb')
protein_indices = traj.top.select('protein')



# Prepare the system
system = forcefield.createSystem(pdb.topology,
                                 nonbondedMethod=app.PME, 
                                 nonbondedCutoff=1*unit.nanometer,
                                 removeCMMotion=False)

# Create the CV: V340A–I458B
# See https://www.nature.com/articles/s41598-022-05875-8/tables/1

#V340A
val_index = [atom.index for atom in pdb.topology.atoms() if (atom.residue.id in ['340'] and atom.residue.chain.id == 'A' and atom.name in ['N','CA','C','O'])]
#I458B
iso_index = [atom.index for atom in pdb.topology.atoms() if (atom.residue.id in ['458'] and atom.residue.chain.id == 'B' and atom.name in ['N','CA','C','O'])]

# http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCentroidBondForce.html
# A Gaussian energy impulse centered at a running value of the CV is periodically added to the total potential energy of the system
force = mm.CustomCentroidBondForce(2, "distance(g1, g2)")

force.addGroup(val_index)
force.addGroup(iso_index)

force.addBond([0, 1])
force.setUsesPeriodicBoundaryConditions(True)

# http://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.BiasVariable.html

# The Gaussian width and height are set to 0.05 Å and 0.01 kcal/mol, respectively.
V340AI458B = app.BiasVariable(force=force,
                              minValue=6 * unit.angstrom, 
                              maxValue=30 * unit.angstrom,
                              biasWidth=0.05 * unit.angstrom, 
                              periodic=False)
#http://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.Metadynamics.html
#
# The time interval between which a Gaussian is injected is set to 0.09 picoseconds (ps)
meta = app.Metadynamics(system=system,
                        variables=[V340AI458B],
                        temperature=310.0,
                        biasFactor=5.0,
                        height=0.01 * unit.kilocalories_per_mole,
                        frequency=45)


# Set up the integrator and simulation
integrator = mm.LangevinIntegrator(310*unit.kelvin,
                                   1/unit.picosecond,
                                   0.002*unit.picoseconds)

simulation = app.Simulation(pdb.topology,
                            system,
                            integrator,
                            platform)

simulation.context.setPositions(pdb.positions)


# Hack to clean up after restart
mm.LocalEnergyMinimizer.minimize(simulation.context)

simulation.reporters.append(mdtraj.reporters.HDF5Reporter('meta.h5', 10000, atomSubset=protein_indices))

# Total steps
# 20,000,000 steps @ 2fs == 40 ns ~3 h on a v100
total_steps=20000000

simulation.reporters.append(app.StateDataReporter("meta.log", 10000, step=True,
        potentialEnergy=True, temperature=True, progress=True, remainingTime=True, totalSteps=total_steps, separator='\t'))

meta.step(simulation, total_steps)

# Close the HDF5 file
simulation.reporters[0].close()


# The result is returned as a N-dimensional NumPy array, where N is the number of collective
# variables.  The values are in kJ/mole.  The i'th position along an axis corresponds to
# minValue + i*(maxValue-minValue)/gridWidth.
print(meta.getFreeEnergy())

plot.plot(meta.getFreeEnergy())
plot.savefig('meta.png')


print(meta.getCollectiveVariables(simulation))


# Distance:

traj = md.load("meta.h5")

V340A = traj.top.select('resSeq 340 and name CA')[0]
I458B = traj.top.select('resSeq 458 and name CA')[0]

dist = md.compute_distances(traj, [[V340A,I458B]], periodic=True, opt=True)

plot, ax = plot.subplots()
ax.plot(dist, alpha=0.5)

ax.set_xlabel("time/ps")
ax.set_ylabel("distance / nm")
ax.set_title("V340A CA / I458B CA distance")
ax.legend()

plot.savefig('dist.png')

