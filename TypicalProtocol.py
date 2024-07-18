#!/usr/bin/env python

# Typical AMBER MD protocol
# 1 ns, HMR, 5 fs time step


# See: http://docs.openmm.org/8.1.0/api-python/index.html
import openmm.app as app
import openmm as mm
import simtk.unit as unit

import numpy as np

import mdtraj.reporters
import mdtraj as md
from pdbfixer import PDBFixer

platform = mm.Platform.getPlatformByName("CUDA")
platformProperties = {}
# CUDA precision
platformProperties['CudaPrecision'] = 'mixed'
platformProperties['CudaDeviceIndex'] = '0'


# Obtain the original PDB
# wget https://files.rcsb.org/download/3F1P.pdb


fixer = PDBFixer(filename='3F1P.pdb')

#Remove waters
fixer.removeHeterogens(False)

fixer.findMissingResidues()
#(0, 0): ['GLY', 'GLU'], 
#(0, 114): ['ASN'], 
#(1, 0): ['GLY', 'GLU', 'PHE', 'LYS', 'GLY', 'LEU', 'ASN'],
#(1, 111): ['SER', 'GLN', 'GLU']}

#Truncate ends
del fixer.missingResidues[(0,0)]
del fixer.missingResidues[(0,114)]
del fixer.missingResidues[(1,0)]
del fixer.missingResidues[(1,111)]

fixer.findMissingAtoms()
fixer.addMissingAtoms()


fixer.addMissingHydrogens(7.4)

app.PDBFile.writeFile(fixer.topology,
                      fixer.positions,
                      open('3F1P_fixed.pdb', 'w'),
                      keepIds=True)



###############################
# 1) Build and solvate system #
###############################
pdb = app.PDBFile('3F1P_fixed.pdb')
# Use AMBER ff14SB
forceField = app.ForceField('amber14-all.xml','./amber14/tip3p.xml')

modeller = app.Modeller(pdb.topology, pdb.positions)

# Add TIP3P solvent
modeller.addSolvent(forceField, model='tip3p', padding=50*unit.angstrom)

app.PDBFile.writeFile(modeller.topology,
                      modeller.positions,
                      open('3F1P_fixed_solvated.pdb', 'w'),
                      keepIds=True)
print("Number of atoms ", len(modeller.positions))


######################
# 2) Minimisation    #
######################
print("Minimising system")

system = forceField.createSystem(
    modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=15*unit.angstrom)
integrator = mm.VerletIntegrator(1*unit.femtosecond)
simulation = app.Simulation(
    modeller.topology, system, integrator, platform, platformProperties)

print("Platform: ", (simulation.context.getPlatform().getName()))

simulation.context.setPositions(modeller.positions)

mm.LocalEnergyMinimizer.minimize(simulation.context)

# Saving minimised positions
positions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(simulation.topology,
                      positions,
                      open('minimisation.pdb', 'w'),
                      keepIds=True)


#######################
#######################
## PMEMD like stages ##
#######################
#######################

friction = 1*(1/unit.picosecond)
dt = 2*unit.femtoseconds
constraints = app.HBonds

################################
# 3) Thermalisation under NVT  #
################################
print("Heating system under NVT")
integrator = mm.LangevinIntegrator(300*unit.kelvin, friction, dt)

# Note, new system, with SHAKE
system = forceField.createSystem(simulation.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=8*unit.angstrom, constraints=constraints)

# Set the COM Removal to something sensible
for i in range(system.getNumForces()):
    if (type(system.getForce(i)) == mm.CMMotionRemover):
        system.getForce(i).setFrequency(1000)


simulation = app.Simulation(
    modeller.topology, system, integrator, platform, platformProperties)
simulation.context.setPositions(positions)


simulation.reporters.append(app.StateDataReporter("heating.csv", 1000,  time=True,
                                                  potentialEnergy=True,
                                                  temperature=True,
                                                  density=True,
                                                  remainingTime=True,
                                                  speed=True,
                                                  totalSteps=35000))
#simulation.reporters.append(app.PDBReporter('heating.pdb', 1000))
simulation.step(35000)  # i.e.

# Save the positions and velocities
state = simulation.context.getState(getPositions=True, getVelocities=True)

app.PDBFile.writeFile(simulation.topology,
                      state.getPositions(),
                      open('heating_final.pdb', 'w'),
                      keepIds=True)

# clear reporters
simulation.reporters = []


####################################
# 4) Density correction under NPT  #
####################################
print("Density correction under NPT")

system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))
integrator = mm.LangevinIntegrator(300*unit.kelvin, friction, dt)

simulation = app.Simulation(
    modeller.topology, system, integrator, platform, platformProperties)
simulation.context.setState(state)


simulation.reporters.append(app.StateDataReporter("density.csv", 1000, 	time=True,
                                                  potentialEnergy=True,
                                                  temperature=True,
                                                  density=True,
                                                  remainingTime=True,
                                                  speed=True,
                                                  totalSteps=35000))
#simulation.reporters.append(app.PDBReporter('density.pdb', 1000))

# 70 ps
simulation.step(35000)

# Save the positions and velocities
state = simulation.context.getState(getPositions=True, getVelocities=True)

app.PDBFile.writeFile(simulation.topology,
                      state.getPositions(),
                      open('density_final.pdb', 'w'),
                      keepIds=True)

# clear reporters
simulation.reporters = []


# Get the index of protein atoms only,
# to enable stripping of water and counter ions in the production trajectory
protein_indices=[atom.index for atom in modeller.topology.atoms() if (not atom.residue.name in "HOH")]


####################################
# 5) Production under NPT          #
####################################

# HMR
print("100 ns Production under NPT, HMR, 5fs ")
friction = 1*(1/unit.picosecond)
dt = 5*unit.femtoseconds
constraints = app.AllBonds
hydrogenMass = 4*unit.amu

# Note, new system, with HMR
system = forceField.createSystem(simulation.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=8*unit.angstrom, constraints=constraints, hydrogenMass=hydrogenMass)

system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))
integrator = mm.LangevinIntegrator(300*unit.kelvin, friction, dt)


simulation = app.Simulation(
    simulation.topology, system, integrator, platform, platformProperties)

simulation.context.setState(state)

# Report every  5,000 steps
simulation.reporters.append(app.StateDataReporter("production.csv", 5000,
                                                  time=True,
                                                  potentialEnergy=True,
                                                  temperature=True,
                                                  density=True,
                                                  remainingTime=True,
                                                  speed=True,
                                                  totalSteps=2000000))
# Structure every 5,000 steps
simulation.reporters.append(mdtraj.reporters.HDF5Reporter(
    'production.h5', 5000, atomSubset=protein_indices))

# Post process with
# mdconvert -f -o production.pdb production.h5


# 2,000,000 steps @ 5fs == 1ns
simulation.step(2000000)

# Save the positions
positions = simulation.context.getState(getPositions=True).getPositions()


app.PDBFile.writeFile(simulation.topology, positions,
                      open('production_final.pdb', 'w'),
                      keepIds=True)

