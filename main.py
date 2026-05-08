"""
bio-compute-stack — Entry Point

GPU-accelerated computational biology: protein structure prediction,
drug candidate screening, molecular dynamics simulation, and model fine-tuning.

Usage:
  python main.py --mode predict --sequence MKTIIALSYIFCLVFA
  python main.py --mode screen --fasta targets.fasta --smiles candidates.txt
  python main.py --mode simulate --pdb structure.pdb --steps 100000
  python main.py --mode finetune --fasta proteins.fasta
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Bio-Compute Stack")
    parser.add_argument("--mode", required=True,
                        choices=["predict", "screen", "simulate", "finetune"],
                        help="Operation mode")
    parser.add_argument("--sequence", type=str, help="Single protein sequence (predict mode)")
    parser.add_argument("--fasta", type=str, help="FASTA file with sequences")
    parser.add_argument("--smiles", type=str, help="File with SMILES strings (one per line)")
    parser.add_argument("--pdb", type=str, help="PDB file for simulation")
    parser.add_argument("--steps", type=int, default=50_000, help="MD simulation steps")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def mode_predict(args):
    from models.esm_fold import ESMFoldPredictor
    predictor = ESMFoldPredictor(device=args.device)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.sequence:
        result = predictor.predict(
            args.sequence,
            output_pdb=f"{args.output}/predicted.pdb"
        )
        print(f"
Result: pTM={result['pTM']}, mean_pLDDT={result['mean_pLDDT']:.1f}")

    elif args.fasta:
        from models.bionemo_trainer import BioNeMoFineTuner
        ft = BioNeMoFineTuner()
        sequences = ft.load_sequences_from_fasta(args.fasta)
        results = predictor.batch_predict(sequences, output_dir=args.output)
        print(f"
Predicted {len(results)} structures to {args.output}")
    else:
        print("Provide --sequence or --fasta")
        sys.exit(1)


def mode_screen(args):
    if not args.fasta or not args.smiles:
        print("--fasta and --smiles required for screen mode")
        sys.exit(1)

    from models.bionemo_trainer import BioNeMoFineTuner
    ft = BioNeMoFineTuner()
    sequences = ft.load_sequences_from_fasta(args.fasta)

    with open(args.smiles) as f:
        smiles_list = [l.strip() for l in f if l.strip()]

    from pipeline.drug_discovery import DrugDiscoveryPipeline
    pipeline = DrugDiscoveryPipeline(output_dir=args.output)

    for i, seq in enumerate(sequences[:5]):
        result = pipeline.run(
            target_sequence=seq,
            candidate_smiles=smiles_list,
            run_name=args.run_name or f"screen_{i:03d}",
        )
        print(f"Target {i}: {result['passing_candidates']} candidates passed, "
              f"{len(result['mol3d_paths'])} ready for docking")


def mode_simulate(args):
    if not args.pdb:
        print("--pdb required for simulate mode")
        sys.exit(1)

    from dynamics.openmm_sim import OpenMMSimulator
    sim = OpenMMSimulator(platform="CUDA" if args.device == "cuda" else "CPU")

    fixed_pdb = sim.fix_pdb(args.pdb)
    min_pdb = sim.minimize_energy(fixed_pdb, output_path=f"{args.output}/minimized.pdb")
    result = sim.run_nvt(
        min_pdb,
        n_steps=args.steps,
        output_dir=args.output,
    )
    print(f"
Simulation done: {result['n_steps']} steps, trajectory at {result['trajectory']}")


def mode_finetune(args):
    if not args.fasta:
        print("--fasta required for finetune mode")
        sys.exit(1)

    from models.bionemo_trainer import BioNeMoFineTuner, BioNeMoTrainConfig
    config = BioNeMoTrainConfig(output_dir=args.output)
    trainer = BioNeMoFineTuner(config)
    sequences = trainer.load_sequences_from_fasta(args.fasta)
    split = int(len(sequences) * 0.9)
    trainer.train(
        train_sequences=sequences[:split],
        eval_sequences=sequences[split:],
        run_name=args.run_name or "esm-finetune",
    )


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Bio-Compute Stack")
    print(f"  Mode: {args.mode.upper()} | Device: {args.device}")
    print("=" * 55)

    dispatch = {
        "predict": mode_predict,
        "screen": mode_screen,
        "simulate": mode_simulate,
        "finetune": mode_finetune,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
