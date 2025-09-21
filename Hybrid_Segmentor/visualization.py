import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import cv2

# Grad-CAM cho CNN

def generate_gradcam_cnn(model, input_tensor, target_layer_name='encoder5', class_idx=None):
    """
    Sinh Grad-CAM heatmap cho nhánh CNN (ResNetEncoder)
    Args:
        model: mô hình HybridSegmentor
        input_tensor: tensor ảnh đầu vào (1, C, H, W)
        target_layer_name: tên layer trong ResNetEncoder để lấy feature map
        class_idx: class index để tính gradient (nếu segmentation thì có thể None)
    Returns:
        heatmap: numpy array (H, W)
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    cnn_encoder = model.cnn_encoder
    feature_maps = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # Lấy layer cần hook
    target_layer = getattr(cnn_encoder, target_layer_name)
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    # Forward
    input_tensor = input_tensor.requires_grad_()
    output = model(input_tensor)[0]  # segmentation output
    if class_idx is None:
        score = output.mean()  # segmentation: lấy mean toàn ảnh
    else:
        score = output[:, class_idx].mean()

    # Backward
    model.zero_grad()
    score.backward(retain_graph=True)

    # Tính Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # (C,)
    feature_maps = feature_maps[0]  # (C, H, W)
    for i in range(feature_maps.shape[0]):
        feature_maps[i, ...] *= pooled_gradients[i]
    heatmap = feature_maps.detach().cpu().numpy().mean(axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    # Resize về kích thước ảnh gốc
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.stack([heatmap]*3, axis=-1)  # RGB
    # Clean up hook
    handle_fwd.remove()
    handle_bwd.remove()
    return heatmap

# Attention rollout cho Transformer

def generate_attention_rollout_transformer(model, input_tensor, target_stage=None):
    """
    Sinh attention rollout heatmap cho nhánh Transformer (MiT)
    Args:
        model: mô hình HybridSegmentor
        input_tensor: tensor ảnh đầu vào (1, C, H, W)
        target_stage: index stage trong MiT (0-3), nếu None sẽ tự chọn stage cuối có block
    Returns:
        heatmap: numpy array (H, W)
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    transformer = model.mix_transformer
    attn_weights = []

    # Tìm stage cuối cùng có block nếu target_stage=None
    if target_stage is None:
        for i in reversed(range(len(transformer.stages))):
            stage = transformer.stages[i][1]
            if len(stage) > 0:
                target_stage = i
                break
        if target_stage is None:
            raise RuntimeError("Không tìm thấy stage nào có block trong Transformer.")

    if target_stage >= len(transformer.stages):
        raise ValueError(f"target_stage={target_stage} vượt quá số stage ({len(transformer.stages)}) trong mô hình.")
    stage = transformer.stages[target_stage][1]  # layers
    if len(stage) == 0:
        raise ValueError(f"Stage {target_stage} không có block nào.")

    # Hook để lấy attention weights từ EfficientMSA
    def get_attention_hook(module, input, output):
        if hasattr(module, 'last_attn_weights') and module.last_attn_weights is not None:
            attn = module.last_attn_weights
            if isinstance(attn, torch.Tensor) and attn.ndim >= 2:
                attn_weights.append(attn.detach().cpu())

    handles = []
    for block in stage:
        if len(block) == 0:
            continue
        msa = block[0]
        handle = msa.register_forward_hook(get_attention_hook)
        handles.append(handle)

    # Forward
    _ = model.mix_transformer(input_tensor)

    if len(attn_weights) == 0:
        for handle in handles:
            handle.remove()
        raise RuntimeError(f"Không thu được attention weights từ stage {target_stage}. Kiểm tra lại mô hình hoặc input.")

    # Rollout: multiply attention qua các layer
    rollout = torch.eye(attn_weights[0].shape[-1])
    for attn in attn_weights:
        attn = attn.squeeze(0).mean(0)  # (N, N)
        attn = attn + torch.eye(attn.shape[0])  # add identity
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout
    mask = rollout[0, 1:]  # bỏ token đầu nếu có, lấy attention tới các patch
    N = mask.shape[0]
    h = int(np.floor(np.sqrt(N)))
    w = int(np.ceil(N / h))
    if h * w != N:
        mask = np.pad(mask, (0, h * w - N), mode='constant')
    heatmap = mask.reshape(h, w).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.stack([heatmap]*3, axis=-1)  # RGB
    # Clean up hook
    for handle in handles:
        handle.remove()
    return heatmap

# So sánh hai heatmap

def compare_heatmaps(heatmap1, heatmap2, image, alpha=0.5, colormap='jet'):
    """
    Overlay hai heatmap lên ảnh gốc để so sánh
    Args:
        heatmap1: numpy array (H, W, 3), ví dụ Grad-CAM
        heatmap2: numpy array (H, W, 3), ví dụ attention rollout
        image: numpy array (H, W, 3) hoặc (C, H, W)
        alpha: độ trong suốt
        colormap: tên colormap matplotlib (mặc định 'jet')
    Returns:
        fig: matplotlib figure
    """
    import matplotlib.cm as cm
    if image.shape[0] == 3 and image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = np.uint8(255 * image)
    # Resize heatmap nếu cần
    if heatmap1.shape[:2] != image.shape[:2]:
        heatmap1 = np.array(Image.fromarray(heatmap1).resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR))
    if heatmap2.shape[:2] != image.shape[:2]:
        heatmap2 = np.array(Image.fromarray(heatmap2).resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR))
    # Convert heatmap to grayscale if needed
    if heatmap1.ndim == 3 and heatmap1.shape[2] == 3:
        heatmap1_gray = cv2.cvtColor(heatmap1, cv2.COLOR_RGB2GRAY)
    else:
        heatmap1_gray = heatmap1
    if heatmap2.ndim == 3 and heatmap2.shape[2] == 3:
        heatmap2_gray = cv2.cvtColor(heatmap2, cv2.COLOR_RGB2GRAY)
    else:
        heatmap2_gray = heatmap2
    # Apply colormap
    heatmap1_color = cm.get_cmap(colormap)(heatmap1_gray / 255.0)[..., :3]
    heatmap1_color = np.uint8(255 * heatmap1_color)
    heatmap2_color = cm.get_cmap(colormap)(heatmap2_gray / 255.0)[..., :3]
    heatmap2_color = np.uint8(255 * heatmap2_color)
    # Overlay
    overlay1 = cv2.addWeighted(image, 1-alpha, heatmap1_color, alpha, 0)
    overlay2 = cv2.addWeighted(image, 1-alpha, heatmap2_color, alpha, 0)
    # So sánh
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[1].imshow(overlay1)
    axs[1].set_title('Grad-CAM (CNN)')
    axs[2].imshow(overlay2)
    axs[2].set_title('Attention Rollout (Transformer)')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    return fig

def visualize_all_stages(model, img_tensor, image_np, alpha=0.5, colormap='jet'):
    """
    Visualize Grad-CAM và Attention Rollout cho tất cả các stage của CNN và Transformer.
    Args:
        model: mô hình HybridSegmentor
        img_tensor: tensor ảnh đầu vào (1, C, H, W)
        image_np: numpy array ảnh gốc (H, W, 3)
        alpha: độ trong suốt overlay
        colormap: tên colormap matplotlib
    Returns:
        fig: matplotlib figure
    """
    import matplotlib.cm as cm
    cnn_layers = ['encoder2', 'encoder3', 'encoder4', 'encoder5']
    n_stages = len(model.mix_transformer.stages)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, max(len(cnn_layers), n_stages), figsize=(5*max(len(cnn_layers), n_stages), 10))
    # Grad-CAM cho các layer CNN
    for i, layer in enumerate(cnn_layers):
        heatmap_cnn = generate_gradcam_cnn(model, img_tensor, target_layer_name=layer)
        # Resize heatmap nếu cần
        if heatmap_cnn.shape[:2] != image_np.shape[:2]:
            heatmap_cnn = np.array(Image.fromarray(heatmap_cnn).resize((image_np.shape[1], image_np.shape[0]), resample=Image.BILINEAR))
        # Convert to color
        if heatmap_cnn.ndim == 3 and heatmap_cnn.shape[2] == 3:
            heatmap_gray = cv2.cvtColor(heatmap_cnn, cv2.COLOR_RGB2GRAY)
        else:
            heatmap_gray = heatmap_cnn
        heatmap_color = cm.get_cmap(colormap)(heatmap_gray / 255.0)[..., :3]
        heatmap_color = np.uint8(255 * heatmap_color)
        # Ensure both are uint8
        img_uint8 = image_np.astype(np.uint8)
        heatmap_color = heatmap_color.astype(np.uint8)
        overlay = cv2.addWeighted(img_uint8, 1-alpha, heatmap_color, alpha, 0)
        axs[0, i].imshow(overlay)
        axs[0, i].set_title(f'Grad-CAM {layer}')
        axs[0, i].axis('off')
    # Attention rollout cho các stage Transformer
    for i in range(n_stages):
        heatmap_trans = generate_attention_rollout_transformer(model, img_tensor, target_stage=i)
        if heatmap_trans.shape[:2] != image_np.shape[:2]:
            heatmap_trans = np.array(Image.fromarray(heatmap_trans).resize((image_np.shape[1], image_np.shape[0]), resample=Image.BILINEAR))
        if heatmap_trans.ndim == 3 and heatmap_trans.shape[2] == 3:
            heatmap_gray = cv2.cvtColor(heatmap_trans, cv2.COLOR_RGB2GRAY)
        else:
            heatmap_gray = heatmap_trans
        heatmap_color = cm.get_cmap(colormap)(heatmap_gray / 255.0)[..., :3]
        heatmap_color = np.uint8(255 * heatmap_color)
        img_uint8 = image_np.astype(np.uint8)
        heatmap_color = heatmap_color.astype(np.uint8)
        overlay = cv2.addWeighted(img_uint8, 1-alpha, heatmap_color, alpha, 0)
        axs[1, i].imshow(overlay)
        axs[1, i].set_title(f'Attention Rollout Stage {i}')
        axs[1, i].axis('off')
    plt.tight_layout()
    return fig
