import torch
import torch.nn as nn
from imagen_pytorch import Unet, Imagen
from PIL import Image
import os
# unet for imagen


class IMAGEN(nn.Module):
    def __init__(self):
        super().__init__()

        self.current_epoch = 0

        unet1 = Unet(
            dim = 32,
            cond_dim = 512,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = 3,
            layer_attns = (False, False, True, True),
            layer_cross_attns = (False, False, True, True)
        )

        unet2 = Unet(
            dim = 32,
            cond_dim = 512,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (1, 2, 4, 8),
            layer_attns = (False, False, False, True),
            layer_cross_attns = (False, False, False, True)
        )

        # imagen, which contains the unets above (base unet and super resoluting ones)
        self.imagen = Imagen(
            unets = (unet1, unet2),
            image_sizes = (64, 256),
            timesteps = 200,
            cond_drop_prob = 0.1
        )

        self.text_embedding = nn.Embedding(50002, 768)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.8)
        return optim, scheduler

    def forward(self, batch, batch_idx, optimizer_idx=None, func='train'):
        if func == 'train':
            return self.training_step(batch, batch_idx, optimizer_idx)
        elif func == 'val':
            return self.validation_step(batch, batch_idx)
        elif func == 'test':
            return self.test_step(batch, batch_idx)

    def training_step(self, batch, batch_idx, optimizer_idx):
        id, text, image = batch
        mask = torch.ones_like(text, dtype=torch.bool)
        mask[text == 50001] = False
        if optimizer_idx == 0:
            loss = self.imagen(image, text_embeds=self.text_embedding(text), text_masks=mask, unet_number=1)
        elif optimizer_idx == 1:
            loss = self.imagen(image, text_embeds=self.text_embedding(text), text_masks=mask, unet_number=2)
        return loss

    def validation_step(self, batch, batch_idx):
        id, text, image = batch
        mask = torch.ones_like(text, dtype=torch.bool)
        mask[text == 50001] = False
        # loss_1 = self.imagen(image, text_embeds=self.text_embedding(text), text_masks=mask, unet_number=1)
        # loss_2 = self.imagen(image, text_embeds=self.text_embedding(text), text_masks=mask, unet_number=2)
        # self.log('val_loss_step1', loss_1, prog_bar=True, sync_dist=True)
        # self.log('val_loss_step2', loss_2, prog_bar=True, sync_dist=True)

        images = self.imagen.sample(text_embeds=self.text_embedding(text), text_masks=mask, cond_scale = 3., batch_size=text.shape[0], use_tqdm=False)
        images = images.clamp(0.0, 1.0) # batch_size * 3 * 256 * 256
        images = (images * 255.0).to(torch.uint8).permute(0, 2, 3, 1) # batch_size * 256 * 256 * 3
        os.makedirs(os.path.join('checkpoints', 'val_samples', str(self.current_epoch)), exist_ok=True)
        for b in range(images.shape[0]):
            image = Image.fromarray(images[b].cpu().detach().numpy()).convert('RGB')
            image.save(os.path.join('checkpoints', 'val_samples', str(self.current_epoch), id[b]+'.jpg'))

        return images


    def test_step(self, batch, batch_idx):
        id, text, image = batch
        mask = torch.ones_like(text, dtype=torch.bool)
        mask[text == 50001] = False
        images = self.imagen.sample(text_embeds=self.text_embedding(text), text_masks=mask, cond_scale = 3.)
        images = images.clamp(0.0, 1.0)
        return images

