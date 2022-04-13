import time

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import trange

from average_meter import AverageMeter
from frames_dataset import DatasetRepeater
from logger import Logger
from modules.model import DiscriminatorFullModel, GeneratorFullModel
from sync_batchnorm import DataParallelWithCallback


def train(config, generator, discriminator, kp_detector, he_estimator, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_he_estimator = torch.optim.Adam(he_estimator.parameters(), lr=train_params['lr_he_estimator'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector, he_estimator,
                                      optimizer_generator, optimizer_discriminator, optimizer_kp_detector, optimizer_he_estimator)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_he_estimator = MultiStepLR(optimizer_he_estimator, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, he_estimator, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'])
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
    g_batch_time = AverageMeter()
    g_back_time = AverageMeter()
    d_batch_time = AverageMeter()
    d_back_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for i, x in enumerate(dataloader):
                data_time.update(time.time() - end)
                end = time.time()
                losses_generator, generated = generator_full(x)
                g_batch_time.update(time.time() - end)
                end = time.time()
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                optimizer_he_estimator.step()
                optimizer_he_estimator.zero_grad()
                g_back_time.update(time.time() - end)
                end = time.time()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    d_batch_time.update(time.time() - end)
                    end = time.time()

                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()

                    d_back_time.update(time.time() - end)
                    end = time.time()
                else:
                    d_batch_time.update(0)
                    d_back_time.update(0)
                    losses_discriminator = {}
                if i % 10 == 0:
                    print(f"Iter [{i}/{len(dataloader)}] done g batch {g_batch_time.val:.6f} g back {g_back_time.val:.6f} d batch {d_batch_time.val:.6f} d back {d_back_time.val:.6f}")
                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            scheduler_he_estimator.step()

            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'he_estimator': he_estimator,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_he_estimator': optimizer_he_estimator}, inp=x, out=generated)
            print(f"epoch {epoch} done!")
