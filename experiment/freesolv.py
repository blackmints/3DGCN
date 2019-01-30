from model.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer("freesolv")

    trainer.fit("arxiv_model", 150, batch=4, fold=10, normalize=True, normalize_adj=True, normalize_pos=0,
                units_conv=128, units_dense=128, num_layers=2, loss="mse", monitor="val_rmse")
