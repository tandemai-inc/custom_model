#!/usr/bin/env python
"""
Quick sanity check: train a dummy ChemProp MPNN on a handful of molecules
to verify the ChemProp setup works.
"""
import os
import sys

# Dummy data: simple small molecules + random labels
DUMMY_SMILES = [
    "C",           # methane
    "CC",          # ethane
    "CCC",         # propane
    "CCO",         # ethanol
    "c1ccccc1",    # benzene
    "CC(=O)O",     # acetic acid
    "CNC",         # dimethylamine
    "CCOC",        # diethyl ether
]
DUMMY_LABELS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


def main():
    print("Testing ChemProp MPNN...")
    
    try:
        from chemprop import data, models, nn
        from lightning.pytorch import Trainer
    except ImportError as e:
        print(f"ERROR: ChemProp not available: {e}")
        print("Activate xgboost_chemprop env: conda activate xgboost_chemprop")
        sys.exit(1)
    
    print(f"  Creating datapoints: {len(DUMMY_SMILES)} molecules")
    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, [label])
        for smi, label in zip(DUMMY_SMILES, DUMMY_LABELS)
    ]
    
    print("  Building dataset and dataloader...")
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer=featurizer)
    scaler = dataset.normalize_targets()
    
    train_loader = data.build_dataloader(dataset, shuffle=True)
    val_loader = data.build_dataloader(dataset, shuffle=False)
    
    print("  Building MPNN model...")
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform)
    model = models.MPNN(
        nn.BondMessagePassing(),
        nn.MeanAggregation(),
        ffn,
    )
    
    print("  Training for 3 epochs...")
    trainer = Trainer(max_epochs=3, logger=False, enable_progress_bar=True)
    trainer.fit(model, train_loader, val_loader)
    
    output_dir = "chemprop_test_output"
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "dummy_model.ckpt")
    trainer.save_checkpoint(ckpt_path)
    
    print(f"\nSUCCESS: ChemProp MPNN trained and saved to {ckpt_path}")
    print("ChemProp is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
