from asyncio import start_server
import torch
import os
from modules import IMAGEN
from data import read_dataset
from torch.utils.data import DataLoader
import accelerate
from accelerate import Accelerator


torch.backends.cudnn.benchmark = True


def main():
    DistributedDataParallelKwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs])
    device = accelerator.device

    model = IMAGEN()
    model.to(device)
    optim, scheduler = model.configure_optimizers()
    

    train_input_npy_path = 'dataset/train.npy'
    train_input_img_path = 'dataset/train'
    val_input_npy_path = 'dataset/val12.npy'
    val_input_img_path = 'dataset/val'
    train_input = read_dataset(train_input_npy_path, train_input_img_path)
    val_input = read_dataset(val_input_npy_path, val_input_img_path)
    batch_size = 4
    train_DataLoader = DataLoader(train_input, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    val_DataLoader = DataLoader(val_input, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    model, optim, scheduler, training_dataloader, val_dataLoader = accelerator.prepare(
        model, optim, scheduler, train_DataLoader, val_DataLoader
    )
    # accelerator.unwrap_model(model).load_state_dict(torch.load('checkpoints/49.pt', map_location='cpu'))

    start_epochs = 0
    epochs = 250
    step_count = 0
    for ep in range(start_epochs, epochs):
        if hasattr(model, 'module'):
            model.module.current_epoch = ep
        else:
            model.current_epoch = ep

        model.train()
        for i, batch in enumerate(training_dataloader):
            optim.zero_grad()
            loss_stage1 = model(batch, i, 0, func='train')
            accelerator.backward(loss_stage1)
            optim.step()

            optim.zero_grad()
            loss_stage2 = model(batch, i, 1, func='train')
            accelerator.backward(loss_stage2)
            optim.step()

            step_count += 1
            if accelerator.is_local_main_process and (i % 10 == 0):
                print('[%03d / %03d] step: %06d, loss_stage1: %.3f, loss_stage2: %.3f' % (ep+1, epochs, step_count, loss_stage1.item(), loss_stage2.item()))

        scheduler.step()

        accelerator.save(accelerator.get_state_dict(model), os.path.join('checkpoints', '%d.pt' % (ep)))
        accelerator.wait_for_everyone()

        if accelerator.is_local_main_process:
            print('Start validation')
        model.eval()
        for i, batch in enumerate(val_dataLoader):
            images = model(batch, i, func='val')


if __name__ == '__main__':
    main()
