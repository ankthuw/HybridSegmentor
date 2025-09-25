from dataloader import get_loaders
import pytorch_lightning as pl
import config
from model import HybridSegmentor
import torch
# from callback import MyPrintingCallBack, EarlyStopping, checkpoint_callback, early_stopping
from callback import EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    # logger = TensorBoardLogger("tb_logs", name="v7_BCEDICE_0_2_final")
    train_loader, val_loader, test_loader = get_loaders(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, config.VAL_IMG_DIR, config.VAL_MASK_DIR, config.TEST_IMG_DIR,
                                                        config.TEST_MASK_DIR, config.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY,)   
    # input_size = (3, 256, 256)
    model = HybridSegmentor(learning_rate=config.LEARNING_RATE).to(config.DEVICE)
    # print(summary(model, input_size=input_size))
    
    # trainer = pl.Trainer(logger=logger, accelerator="gpu", min_epochs=1, 
    #                      max_epochs=config.NUM_EPOCHS, precision='16-mixed',
    #                      callbacks=[checkpoint_callback, early_stopping])
    trainer = pl.Trainer(logger=False, 
                         enable_checkpointing=False, 
                         accelerator="gpu",
                         min_epochs=1, 
                         max_epochs=config.NUM_EPOCHS, 
                         precision="16-mixed",
                         callbacks=[EarlyStopping(monitor="val_loss", patience=10)], 
                         enable_model_summary=False, 
                         log_every_n_steps=0  
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)
    # trainer.test(model, test_loader, ckpt_path="best")
    trainer.test(model, test_loader, ckpt_path=None)
