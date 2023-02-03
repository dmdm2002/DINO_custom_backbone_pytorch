import os
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from Utils.Option import Param


class CkpHandler(Param):
    def __init__(self):
        super(CkpHandler, self).__init__()

    def weight_init(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

    def load_ckp(self, model, imagenet=False):
        if imagenet:
            print(f'Pretrained [IMAGE NET] Loading..')
            return model, 0

        else:
            if self.CKP_LOAD:
                print(f'Check Point [{self.LOAD_CKP_EPCOH}] Loading...')
                ckp = torch.load(f'{self.OUTPUT_CKP}/{self.LOAD_CKP_EPCOH}.pth')
                model.load_state_dict(ckp['model_state_dict'])
                epoch = ckp['epoch'] + 1

            else:
                print(f'Initialize Model Weight...')
                model.apply(self.weight_init)
                epoch = 0

            return model, epoch

    def save_ckp(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
            },
            os.path.join(f'{self.OUTPUT_CKP}', f'{epoch}.pth')
        )


class TransformBuilder(Param):
    def __init__(self):
        super(TransformBuilder, self).__init__()

    def set_train_transform(self):
        assert type(self.AUG) is bool, 'Only boolean type is available for self.AUG.'

        if self.AUG:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(self.INPUT_SIZE),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(self.INPUT_SIZE),
            ])

        return transform

    def set_valid_test_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
        ])

        return transform


class TensorboardHandler(Param):
    def __init__(self):
        super(TensorboardHandler, self).__init__()
        self.summary = SummaryWriter(self.OUTPUT_LOG)

    def global_logging(self, score_list, ep):
        self.summary.add_scalar("train/loss", score_list[0], ep)
        self.summary.add_scalar("valid/best_acc", score_list[1], ep)
        self.summary.add_scalar("valid/bset_loss", score_list[2], ep)

    def valid_local_logging(self, score_list, ep):
        path = f'{self.OUTPUT_LOG}/valid_log'
        os.makedirs(path, exist_ok=True)
        df = pd.DataFrame(score_list, columns=['Accuracy', 'Loss'])
        df.to_csv(f'{path}/{ep}.csv', index=False)
