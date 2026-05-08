"""
OpenMM molecular dynamics simulation.
Run protein stability checks and ligand binding free energy calculations.
SDKs: OpenMM, OpenMMForceFields, PDBFixer, MDAnalysis
"""
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from pdbfixer import PDBFixer
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("Warning: OpenMM not available. Install: conda install -c conda-forge openmm pdbfixer")

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, align
    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False


class OpenMMSimulator:
    """
    GPU-accelerated molecular dynamics using OpenMM.
    Supports protein stability runs, energy minimization, NPT/NVT ensembles.
    """

    def __init__(self, platform: str = "CUDA"):
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM required. Install: conda install -c conda-forge openmm")
        self.platform_name = platform
        try:
            self.platform = mm.Platform.getPlatformByName(platform)
            print(f"[OpenMM] Using {platform} platform")
        except Exception:
            self.platform = mm.Platform.getPlatformByName("CPU")
            print("[OpenMM] Falling back to CPU platform")

    def fix_pdb(self, pdb_path: str, output_path: Optional[str] = None) -> str:
        """
        Use PDBFixer to add missing residues, atoms, and hydrogens.
        Returns path to fixed PDB.
        """
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        output_path = output_path or pdb_path.replace(".pdb", "_fixed.pdb")
        with open(output_path, "w") as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
        print(f"[PDBFixer] Fixed PDB saved to {output_path}")
        return output_path

    def build_system(self, pdb_path: str, forcefield: str = "amber14-all.xml"):
        """Load PDB and build OpenMM system with forcefield."""
        pdb = app.PDBFile(pdb_path)
        ff = app.ForceField(forcefield, "implicit/gbn2.xml")
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            hydrogenMass=1.5 * unit.amu,
        )
        return pdb, system

    def minimize_energy(self, pdb_path: str, output_path: Optional[str] = None) -> str:
        """
        Energy minimization to relax structure before MD.
        Returns path to minimized PDB.
        """
        pdb, system = self.build_system(pdb_path)
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
        )
        simulation = app.Simulation(pdb.topology, system, integrator, self.platform)
        simulation.context.setPositions(pdb.positions)

        print(f"[OpenMM] Initial energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")
        simulation.minimizeEnergy(tolerance=10 * unit.kilojoule_per_mole / unit.nanometer)
        print(f"[OpenMM] Minimized energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")

        output_path = output_path or pdb_path.replace(".pdb", "_minimized.pdb")
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(output_path, "w") as f:
            app.PDBFile.writeFile(pdb.topology, positions, f)
        print(f"[OpenMM] Minimized structure saved to {output_path}")
        return output_path

    def run_nvt(
        self,
        pdb_path: str,
        n_steps: int = 50_000,
        temperature: float = 300.0,
        report_interval: int = 1000,
        output_dir: str = "./sim_output",
    ) -> Dict[str, Any]:
        """
        NVT (constant volume + temperature) MD simulation.
        n_steps=50000 at 2fs timestep = 100ps
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pdb, system = self.build_system(pdb_path)

        # Add thermostat
        integrator = mm.LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            1 / unit.picosecond,
            0.002 * unit.picoseconds,
        )

        simulation = app.Simulation(pdb.topology, system, integrator, self.platform)
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

        # Reporters
        traj_path = os.path.join(output_dir, "trajectory.dcd")
        log_path = os.path.join(output_dir, "md.log")
        simulation.reporters.append(app.DCDReporter(traj_path, report_interval))
        simulation.reporters.append(app.StateDataReporter(
            log_path, report_interval,
            step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, temperature=True, progress=True,
            totalSteps=n_steps,
        ))

        print(f"[OpenMM] Running NVT MD: {n_steps} steps at {temperature}K...")
        simulation.step(n_steps)
        print(f"[OpenMM] Simulation complete. Trajectory: {traj_path}")

        return {
            "trajectory": traj_path,
            "log": log_path,
            "n_steps": n_steps,
            "temperature_K": temperature,
        }


class MDAnalysisProcessor:
    """
    Analyze MD trajectories using MDAnalysis.
    RMSD, RMSF, radius of gyration, contact maps.
    """

    def __init__(self, topology: str, trajectory: str):
        if not MDA_AVAILABLE:
            raise ImportError("MDAnalysis required. Install: pip install MDAnalysis")
        self.universe = mda.Universe(topology, trajectory)
        print(f"[MDAnalysis] Loaded {self.universe.trajectory.n_frames} frames, "
              f"{len(self.universe.atoms)} atoms")

    def compute_rmsd(self, selection: str = "backbone") -> np.ndarray:
        """Compute RMSD over trajectory relative to first frame."""
        ref = self.universe.copy()
        R = rms.RMSD(self.universe, ref, select=selection)
        R.run()
        rmsd = R.results.rmsd[:, 2]
        print(f"[MDAnalysis] RMSD: mean={rmsd.mean():.2f}A, max={rmsd.max():.2f}A")
        return rmsd

    def compute_rmsf(self, selection: str = "backbone") -> np.ndarray:
        """Compute per-residue RMSF (flexibility)."""
        align.AlignTraj(self.universe, self.universe, select=selection).run()
        atoms = self.universe.select_atoms(selection)
        R = rms.RMSF(atoms)
        R.run()
        return R.results.rmsf

    def radius_of_gyration(self) -> np.ndarray:
        """Compute radius of gyration over trajectory."""
        rg = []
        protein = self.universe.select_atoms("protein")
        for ts in self.universe.trajectory:
            rg.append(protein.radius_of_gyration())
        return np.array(rg)
