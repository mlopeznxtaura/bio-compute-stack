"""
ESMFold protein structure prediction via Meta ESM-2.
Predicts 3D structure from amino acid sequence in seconds on GPU.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union
try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("Warning: ESM not available. Install: pip install fair-esm")

try:
    import biotite.structure as struc
    import biotite.structure.io as strucio
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False


class ESMFoldPredictor:
    """
    Predict protein 3D structure from sequence using ESMFold.
    No MSA required — single-sequence, fast inference.
    SDKs: ESM-2, PyTorch, Biotite
    """

    def __init__(self, device: str = "cuda"):
        if not ESM_AVAILABLE:
            raise ImportError("Install fair-esm: pip install fair-esm")
        self.device = device
        print("[ESMFold] Loading model...")
        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval().to(device)
        print(f"[ESMFold] Ready on {device}")

    @torch.no_grad()
    def predict(self, sequence: str, output_pdb: Optional[str] = None) -> dict:
        """
        Predict structure from amino acid sequence.
        Returns dict with pdb_str, pTM score, pLDDT per residue.
        """
        sequence = sequence.upper().replace(" ", "")
        print(f"[ESMFold] Predicting structure for {len(sequence)}-residue protein...")

        with torch.cuda.amp.autocast():
            output = self.model.infer_pdb(sequence)

        # Extract confidence scores
        plddt = output.get("plddt", None)
        ptm = output.get("ptm", None)

        result = {
            "sequence": sequence,
            "pdb_str": output if isinstance(output, str) else output.get("pdb_str", ""),
            "length": len(sequence),
            "pTM": float(ptm) if ptm is not None else None,
            "mean_pLDDT": float(plddt.mean()) if plddt is not None else None,
        }

        if output_pdb:
            Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
            pdb_str = result["pdb_str"]
            with open(output_pdb, "w") as f:
                f.write(pdb_str)
            print(f"[ESMFold] Saved PDB to {output_pdb}")
            result["pdb_path"] = output_pdb

        if result["pTM"]:
            print(f"[ESMFold] pTM={result['pTM']:.3f}, mean_pLDDT={result['mean_pLDDT']:.1f}")

        return result

    def batch_predict(self, sequences: list, output_dir: str = "./structures") -> list:
        """Predict structures for a batch of sequences."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        for i, seq in enumerate(sequences):
            out_pdb = f"{output_dir}/protein_{i:04d}.pdb"
            result = self.predict(seq, output_pdb=out_pdb)
            results.append(result)
            print(f"[ESMFold] {i+1}/{len(sequences)} done")
        return results

    def compute_plddt_regions(self, plddt: np.ndarray) -> dict:
        """
        Categorize residues by pLDDT confidence.
        Very high: >90, High: 70-90, Low: 50-70, Very low: <50
        """
        return {
            "very_high": int((plddt > 90).sum()),
            "high": int(((plddt > 70) & (plddt <= 90)).sum()),
            "low": int(((plddt > 50) & (plddt <= 70)).sum()),
            "very_low": int((plddt <= 50).sum()),
            "total": len(plddt),
        }
