# bio-compute-stack

GPU-accelerated drug discovery and genomics pipeline, fully open-source.

**Cluster 05 of 25 — NextAura 500 SDKs / 25 Clusters**

## Overview
A research platform that chains together protein structure prediction, small molecule generation,
molecular dynamics validation, and genomics analysis — all running on GPU, all free and open-source.

Pipeline: target identification (Scanpy scRNA) → protein structure (BioNeMo) → molecule generation
(MolMIM + DiffDock) → MD validation (OpenMM). One DAG, end to end.

## 20 SDKs
BioNeMo · NVIDIA Clara Parabricks · MONAI · RDKit · BioPython · OpenMM · GROMACS · Scanpy ·
PyTorch · Hugging Face Transformers · JAX · NumPy · Polars · DuckDB · Apache Arrow ·
Weights & Biases · FastAPI · Prefect · PennyLane · SymPy

## Training Corpus
200,000 runs · 20 SDKs · 10 cycles · 1,000 runs/SDK/cycle

## Schema
```json
{"run_id": "cluster_05_bionemo_1_00001", "cluster": "cluster_05", "sdk": "bionemo", "cycle": 1,
 "model": "bionemo-esm2", "task": "folding", "status": "success",
 "latency_s": 2.341, "tokens_in": 512, "tokens_out": 256, "cost_usd": 0.012, "ts": "2026-05-01T00:01:00Z"}
```

## Repository
Part of [NextAura](https://github.com/mlopeznxtaura) — github.com/mlopeznxtaura
