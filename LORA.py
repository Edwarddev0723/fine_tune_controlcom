import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms

def resize_with_pad(image, target_size, pad_color=(0, 0, 0)):
    h, w = image.shape[:2]
    scale = float(target_size) / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top_pad = (target_size - new_h) // 2
    bottom_pad = target_size - new_h - top_pad
    left_pad = (target_size - new_w) // 2
    right_pad = target_size - new_w - left_pad
    padded_image = cv2.copyMakeBorder(
        resized,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return padded_image

def normalize_bbox(bbox, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return [
        x1 / float(w),
        y1 / float(h),
        x2 / float(w),
        y2 / float(h)
    ]

# ============================
# 自定義資料集 (與之前相同)
# ============================
class CustomObjectDataset(Dataset):
    def __init__(self, root_dir, target_size=512, transform=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform
        self.samples = []
        categories = os.listdir(root_dir)
        for cat in categories:
            cat_path = os.path.join(root_dir, cat)
            if not os.path.isdir(cat_path):
                continue
            bg_path = os.path.join(cat_path, 'bg')
            if not os.path.exists(bg_path):
                continue
            bg_files = glob.glob(os.path.join(bg_path, '*.jpg')) + \
                       glob.glob(os.path.join(bg_path, '*.png'))
            fg_folders = [d for d in os.listdir(cat_path) if d.startswith('fg')]
            for fg_folder in fg_folders:
                fg_folder_path = os.path.join(cat_path, fg_folder)
                if not os.path.isdir(fg_folder_path):
                    continue
                jpg_files = sorted(glob.glob(os.path.join(fg_folder_path, '*.jpg')))
                png_files = sorted(glob.glob(os.path.join(fg_folder_path, '*.png')))
                for jf, pf in zip(jpg_files, png_files):
                    for bgf in bg_files:
                        bbox = [0.3, 0.3, 0.7, 0.7]  # 固定示範
                        self.samples.append((bgf, jf, pf, bbox))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bg_path, fg_path, mask_path, bbox = self.samples[idx]
        bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        fg_img = cv2.imread(fg_path, cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        bg_resized = resize_with_pad(bg_img, self.target_size)
        fg_resized = resize_with_pad(fg_img, self.target_size)
        mask_resized = resize_with_pad(mask_img, self.target_size)
        norm_bbox = normalize_bbox(bbox, bg_resized.shape)
        bg_tensor = torch.from_numpy(bg_resized).float().permute(2, 0, 1) / 255.0
        fg_tensor = torch.from_numpy(fg_resized).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
        return bg_tensor, fg_tensor, mask_tensor, norm_bbox

# ============================
# 替換為 LatentDiffusion 的 LoRA 注入
# ============================
from lora_diffusion import inject_trainable_lora
from libcom.objectstitch import Mure_ObjectStitchModel

class LoRA_ObjectStitch(Mure_ObjectStitchModel):
    def __init__(self, device='cuda:0', lora_rank=4, **kwargs):
        super().__init__(device=device, **kwargs)
        self.num_timesteps = 1000
        # 如果檢查後發現 U-Net 其實存放在 self.model.model
        self.diffusion_model = self.model.model  # <--- 假設實際名稱是這個
        inject_trainable_lora(self.diffusion_model, r=lora_rank)
        self._freeze_base_model()

    def _freeze_base_model(self):
        for param in self.diffusion_model.parameters():
            param.requires_grad = False
        for module in self.diffusion_model.modules():
            if hasattr(module, "lora_layer"):
                for param in module.lora_layer.parameters():
                    param.requires_grad = True

    def compute_training_loss(self, background_image, foreground_image, foreground_mask, bbox):
        # 假設 self.num_timesteps 已在 __init__ 中定義, 如 self.num_timesteps=1000
        batch_size = foreground_image.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=foreground_image.device).long()

        # 將單通道遮罩複製成 3 通道
        mask_3ch = foreground_mask.repeat(1, 3, 1, 1)  # shape=(B,3,H,W)

        # 將背景(3ch)、前景(3ch)及複製後遮罩(3ch)串接
        x = torch.cat([background_image, foreground_image, mask_3ch], dim=1)  # shape=(B,9,H,W)

        # 通道滿足9後再 forward
        pred_noise = self.diffusion_model(x, t)

        loss = ((pred_noise - foreground_image) ** 2).mean()
        return loss

def train_lora_objectstitch(
    root_dir,
    batch_size=1,
    epochs=10,
    lr=1e-4,
    lora_rank=8,
    sampler='ddim',
    device='cuda:0'
):
    dataset = CustomObjectDataset(root_dir=root_dir, target_size=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 初始化含 LoRA 的模型
    model = LoRA_ObjectStitch(device=device, lora_rank=lora_rank, sampler=sampler)
    # 這裡的 optimizer 要抓的是 diffusion_model
    optimizer = AdamW(model.diffusion_model.parameters(), lr=lr)
    model.model.to(device)

    for epoch in range(epochs):
        for i, (bg, fg, mask, bbox) in enumerate(dataloader):
            bg = bg.to(device)
            fg = fg.to(device)
            mask = mask.to(device)
            loss = model.compute_training_loss(
                background_image=bg,
                foreground_image=fg,
                foreground_mask=mask,
                bbox=bbox
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{epochs}, Loss = {loss.item()}")
    return model

def inference_lora_objectstitch(
    model_path,
    background_path,
    foreground_paths,
    mask_paths,
    bbox,
    output_dir="output",
    num_samples=3,
    sample_steps=25,
    device='cuda:0'
):
    model = LoRA_ObjectStitch(device=device, lora_rank=8)
    # 載入 LoRA 權重到 diffusion_model
    lora_weights = torch.load(model_path, map_location=device)
    model.diffusion_model.load_state_dict(lora_weights, strict=False)

    bg_img = cv2.imread(background_path, cv2.IMREAD_COLOR)
    fg_imgs = [cv2.imread(fg, cv2.IMREAD_COLOR) for fg in foreground_paths]
    mask_imgs = [cv2.imread(m, cv2.IMREAD_GRAYSCALE) for m in mask_paths]

    # 使用 model(...) 執行推理，實際需依 Mure_ObjectStitchModel 流程
    composite_images = model(
        background_image=bg_img,
        foreground_image=fg_imgs,
        foreground_mask=mask_imgs,
        bbox=bbox,
        num_samples=num_samples,
        sample_steps=sample_steps
    )

    os.makedirs(output_dir, exist_ok=True)
    for i, cimg in enumerate(composite_images):
        out_path = os.path.join(output_dir, f"composite_{i}.png")
        cv2.imwrite(out_path, cimg)
    return composite_images

if __name__ == "__main__":
    trained_model = train_lora_objectstitch(
        root_dir="MureCom",
        batch_size=1,
        epochs=5,
        lr=1e-4,
        lora_rank=8,
        sampler='ddim',
        device='cuda:0'
    )
    # 假設完成後儲存
    torch.save(trained_model.diffusion_model.state_dict(), "lora_weights.bin")

    inference_lora_objectstitch(
        model_path="lora_weights.bin",
        background_path="MureCom/Airplane/bg/airplane_bg1.jpg",
        foreground_paths=["MureCom/Airplane/fg1/0.jpg", "MureCom/Airplane/fg1/1.jpg"],
        mask_paths=["MureCom/Airplane/fg1/0.png", "MureCom/Airplane/fg1/1.png"],
        bbox=[0.3, 0.3, 0.7, 0.7],
        output_dir="output",
        num_samples=3,
        sample_steps=25,
        device='cuda:0'
    )
