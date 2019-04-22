from model.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer("bace_rotated")

    hyperparameters = {"epoch": 150, "batch": 16, "fold": 10, "units_conv": 128, "units_dense": 128, "pooling": "max",
                       "num_layers": 2, "loss": "binary_crossentropy", "monitor": "val_roc", "label": ""}

    features = {"use_atom_symbol": True, "use_degree": True, "use_hybridization": True, "use_implicit_valence": True,
                "use_partial_charge": True, "use_ring_size": True, "use_hydrogen_bonding": True,
                "use_acid_base": True, "use_aromaticity": True, "use_chirality": True, "use_num_hydrogen": True}

    # Baseline
    trainer.fit("model_3DGCN", **hyperparameters, **features)
