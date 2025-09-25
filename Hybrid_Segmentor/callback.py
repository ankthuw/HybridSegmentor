from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint
import os

class MyPrintingCallBack(Callback):
    def __init__(self):
        super(MyPrintingCallBack, self).__init__()

    def on_train_start(self, trainer, pl_module):
        print("Start Training")

    def on_train_end(self, trainer, pl_module):
        print("Training is done")
        
    def on_validation_end(self, trainer, pl_module):
        print("Validation completed")

# Tạo thư mục checkpoints nếu chưa tồn tại
checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints', 'hybrid_model_bifusion')
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='hybrid-{epoch:02d}-{val_loss:.4f}',
    verbose=True,
    save_last=True,
    save_top_k=3,  # Lưu 3 model tốt nhất
    monitor='val_loss',
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Giảm patience để dừng sớm hơn
    verbose=True,
    mode='min',
    min_delta=1e-4  # Thêm ngưỡng tối thiểu để xem xét cải thiện
)
