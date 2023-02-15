from asyncio import start_server
import torch
import os
from modules import IMAGEN
from data import read_dataset
from torch.utils.data import DataLoader
import accelerate
from accelerate import Accelerator
from PIL import Image

torch.backends.cudnn.benchmark = True


def main():
    DistributedDataParallelKwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs])
    device = accelerator.device

    model = IMAGEN()
    model.to(device)
    optim, scheduler = model.configure_optimizers()
    
    sample_input_npy_path = 'dataset/test.npy'
    sample_input = read_dataset(sample_input_npy_path, None)
    batch_size = 1
    sample_DataLoader = DataLoader(sample_input, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    model, optim, scheduler, sample_dataloader,  = accelerator.prepare(
        model, optim, scheduler, sample_DataLoader, 
    )
    accelerator.unwrap_model(model).load_state_dict(torch.load('checkpoints/199.pt', map_location='cpu'))
    os.makedirs(os.path.join('test'), exist_ok=True)
    for i, batch in enumerate(sample_dataloader):
        if accelerator.is_local_main_process and (i % 10 == 0):
            print('[%03d / %03d]' % (i, len(sample_dataloader)))
        images = model(batch, i, func='test')
        images.squeeze(0)
        images = images[0]
        images = (images * 255.0).to(torch.uint8).permute(1, 2, 0) # 256 * 256 * 3
        image = Image.fromarray(images.cpu().detach().numpy()).convert('RGB')
        image.save(os.path.join('test', batch[0][0] + '.jpg'))
    


if __name__ == '__main__':
    main()
