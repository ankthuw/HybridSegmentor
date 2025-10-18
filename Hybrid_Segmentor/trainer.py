from dataloader import get_loaders
import pytorch_lightning as pl
import config
from model import HybridSegmentor
import torch
from callback import MyPrintingCallBack, checkpoint_callback, early_stopping
import os
from pytorch_lightning.loggers import CSVLogger



torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR, config.VAL_MASK_DIR,
        config.TEST_IMG_DIR, config.TEST_MASK_DIR,
        config.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY,
    )   
    logger = CSVLogger(save_dir="logs", name="hybrid_segmentor")
    model = HybridSegmentor(learning_rate=config.LEARNING_RATE).to(config.DEVICE)
    
    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        accelerator="gpu",
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision="16-mixed",
        callbacks=[MyPrintingCallBack(), checkpoint_callback, early_stopping],
        enable_model_summary=False,
        log_every_n_steps=10,
    )
    
    # Training
    trainer.fit(model, train_loader, val_loader)
    
    # Validation
    trainer.validate(model, val_loader)
    
    # Testing với best checkpoint
    trainer.test(model, test_loader, ckpt_path="best")
    
    # Lưu model cuối cùng
    save_path = os.path.join(os.getcwd(), 'final_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Saved final model to {save_path}")
