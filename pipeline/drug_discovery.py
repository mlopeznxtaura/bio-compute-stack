"""
Drug discovery pipeline: target protein -> candidate molecules -> docking -> MD validation.
SDKs: RDKit, ESMFold, OpenMM, py3Dmol, BioNeMo
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem, Descriptors, Draw, FilterCatalog,
        rdMolDescriptors, rdFingerprintGenerator
    )
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install: pip install rdkit")

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False


class MoleculeFilter:
    """
    Apply medicinal chemistry filters to candidate molecules.
    Lipinski, PAINS, structural alerts, ADMET properties.
    """

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required")
        # PAINS filter
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_catalog = FilterCatalog.FilterCatalog(params)

    def lipinski_ro5(self, mol) -> Dict[str, Any]:
        """Check Lipinski Rule of Five compliance."""
        mw = Descriptors.MolWt(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        violations = sum([
            mw > 500,
            hbd > 5,
            hba > 10,
            logp > 5,
        ])

        return {
            "MW": round(mw, 2),
            "HBD": hbd,
            "HBA": hba,
            "LogP": round(logp, 2),
            "TPSA": round(tpsa, 2),
            "RotBonds": rot_bonds,
            "Ro5_violations": violations,
            "passes_Ro5": violations <= 1,
        }

    def is_pains(self, mol) -> bool:
        """Check if molecule hits PAINS (pan-assay interference) filters."""
        entry = self.pains_catalog.GetFirstMatch(mol)
        return entry is not None

    def compute_fingerprint(self, mol, radius: int = 2, n_bits: int = 2048):
        """Morgan fingerprint for similarity search."""
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return gen.GetFingerprint(mol)

    def filter_candidates(self, smiles_list: List[str]) -> List[Dict]:
        """
        Filter a list of SMILES strings through drug-likeness criteria.
        Returns list of passing molecules with their properties.
        """
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            ro5 = self.lipinski_ro5(mol)
            if not ro5["passes_Ro5"]:
                continue
            if self.is_pains(mol):
                continue
            results.append({
                "smiles": smi,
                "mol": mol,
                **ro5,
            })
        print(f"[Filter] {len(results)}/{len(smiles_list)} molecules passed drug-likeness filters")
        return results


class DrugDiscoveryPipeline:
    """
    End-to-end pipeline: protein sequence -> structure -> candidate screening -> hits.
    """

    def __init__(self, output_dir: str = "./pipeline_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filter = MoleculeFilter()

    def smiles_to_3d(self, smiles: str, out_sdf: Optional[str] = None):
        """Generate 3D coordinates for a molecule from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        if out_sdf:
            writer = Chem.SDWriter(out_sdf)
            writer.write(mol)
            writer.close()
        return mol

    def visualize_3d(self, pdb_str: str, width: int = 800, height: int = 600):
        """Render protein structure in 3D with py3Dmol."""
        if not PY3DMOL_AVAILABLE:
            print("py3Dmol not available for visualization")
            return
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_str, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.zoomTo()
        return view

    def run(
        self,
        target_sequence: str,
        candidate_smiles: List[str],
        run_name: str = "run01",
    ) -> Dict[str, Any]:
        """
        Full pipeline: predict target structure, filter candidates, prep for docking.
        """
        print(f"[Pipeline] Starting drug discovery run: {run_name}")
        print(f"[Pipeline] Target: {len(target_sequence)} residues, {len(candidate_smiles)} candidates")

        # Step 1: Predict protein structure
        from models.esm_fold import ESMFoldPredictor
        predictor = ESMFoldPredictor()
        structure = predictor.predict(
            target_sequence,
            output_pdb=str(self.output_dir / f"{run_name}_target.pdb")
        )

        # Step 2: Filter candidates
        passing = self.filter.filter_candidates(candidate_smiles)

        # Step 3: Generate 3D coords for top candidates
        mol3d_paths = []
        for i, cand in enumerate(passing[:20]):
            sdf_path = str(self.output_dir / f"{run_name}_candidate_{i:03d}.sdf")
            try:
                self.smiles_to_3d(cand["smiles"], out_sdf=sdf_path)
                mol3d_paths.append(sdf_path)
            except Exception as e:
                print(f"[Pipeline] 3D gen failed for candidate {i}: {e}")

        print(f"[Pipeline] {len(mol3d_paths)} candidates ready for docking")
        return {
            "run_name": run_name,
            "target_structure": structure,
            "passing_candidates": len(passing),
            "mol3d_paths": mol3d_paths,
        }
