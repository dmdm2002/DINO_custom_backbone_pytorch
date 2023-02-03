import os
import tqdm
import pandas as pd

import torch
import torchsummary
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from Model.Backbone.VIGA.PyramidViGA import Pyramid_ViGA
from Model.DINO.DINO import DINO_model
from Model.Classifier.eval_linear import LinearClassifier

from Utils.Custom_Loader import Loader
from Utils.Option import Param
from Utils.Modules import CkpHandler, TransformBuilder, TensorboardHandler
from Utils.Displayer import LossAccDisplayer

from lightly.data import DINOCollateFunction
from lightly.loss import DINOLoss
from lightly.models.utils import update_momentum


class Trainer(Param):
    def __init__(self):
        super(Trainer, self).__init__()
        self.ckp_handler = CkpHandler()
        self.transform_builder = TransformBuilder()
        self.tensorboard_handler = TensorboardHandler()

        os.makedirs(self.OUTPUT_CKP, exist_ok=True)
        os.makedirs(self.OUTPUT_LOG, exist_ok=True)

        _error_path = f'{self.OUTPUT_LOG}/Error_Log.csv'

        if not os.path.isfile(_error_path):
            df = pd.DataFrame(columns=['Folder', 'ImageName', 'Class'])
            df.to_csv(_error_path, index=False)

    def run(self):
        print('--------------------------------------')
        print(f'[RunType] : Training!!')
        print(f'[Device] : {self.DEVICE}!!')
        print('--------------------------------------')

        backbone = Pyramid_ViGA(num_class=10, k=9)
        backbone, epoch = self.ckp_handler.load_ckp(backbone, imagenet=False)
        backbone = nn.Sequential(*list(backbone.children())[:-1])

        model = DINO_model(backbone, 768)
        model.to(self.DEVICE)
        torchsummary.summary(model, (3, 224, 224), device='cpu')

        transform = self.transform_builder.set_train_transform()
        tr_dataset = Loader(self.DATASET_PATH, run_type='train', transform=transform)

        collate_fn = DINOCollateFunction()
        criterion = DINOLoss(
            output_dim=2048,
            warmup_teacher_temp_epochs=5,
        )

        criterion = criterion.to(self.DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=self.LR)

        train_disp = LossAccDisplayer(['loss'])

        for ep in range(epoch, self.EPOCH):
            tr_loader = DataLoader(dataset=tr_dataset, batch_size=self.BATCHSZ, collate_fn=collate_fn, shuffle=True)

            model.train()
            for idx, (views, _, name) in enumerate(tqdm.tqdm(tr_loader, desc=f'NOW EPOCH [{ep}/{self.EPOCH}]')):
                update_momentum(model.student_backbone, model.teacher_backbone, m=0.99)
                update_momentum(model.student_head, model.teacher_head, m=0.99)
                views = [view.to(self.DEVICE) for view in views]

                global_views = views[:2]

                teacher_output = [model.forward_teacher(view) for view in global_views]
                student_output = [model.forward(view) for view in views]

                loss = criterion(teacher_output, student_output, epoch=ep)
                train_disp.record([loss.item()])

                loss.backward()
                model.student_head.cancel_last_layer_gradients(current_epoch=ep)
                optimizer.step()
                optimizer.zero_grad()

            score_list = train_disp.get_avg_losses()
            print(f'==> TRAIN EPOCH[{ep}/{self.EPOCH}] || Loss : {score_list[0]} ||')

            model.eval()
            best_socre = self.run_valid_classifier(model, ep)

            score_list = score_list + best_socre

            self.tensorboard_handler.global_logging(score_list, ep)

    def run_valid_classifier(self, dino_model, train_epoch):
        print('--------------------------------------')
        print(f'[RunType] : Valid Classifier!!')
        print('--------------------------------------')
        transform = self.transform_builder.set_valid_test_transform()
        val_dataset = Loader(self.DATASET_PATH, run_type='test', transform=transform)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.BATCHSZ, shuffle=False)

        classifier = LinearClassifier(1).to(self.DEVICE)
        criterion = nn.BCELoss()

        optimizer = optim.Adam(classifier.parameters(), lr=self.LR)

        valid_disp = LossAccDisplayer(['acc', 'loss'])

        best_score = [0, 0]
        full_score_list = []

        for ep in range(self.VALID_EPOCH):
            for idx, (item, label, name) in enumerate(
                    tqdm.tqdm(val_loader, desc=f'NOW VALIDATION EPOCH [{ep}/{self.VALID_EPOCH}]')):
                item = item.to(self.DEVICE)
                label = label.to(self.DEVICE)

                dino_feature_map = dino_model.forward_teacher(item)
                probs = classifier(dino_feature_map)

                loss = criterion(probs.type(torch.FloatTensor), label.type(torch.FloatTensor))

                probs = probs.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                preds = probs > 0.5
                batch_acc = (label == preds).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                valid_disp.record([batch_acc, loss.item()])

            score_list = valid_disp.get_avg_losses()
            full_score_list.append(score_list)
            if best_score[0] <= score_list[0]:
                best_score = score_list

            print(f'==> VALID EPOCH[{ep}/{self.VALID_EPOCH}] || Accuracy : {score_list[0]} | Loss : {score_list[1]} ||]')
            valid_disp.reset()
        self.tensorboard_handler.valid_local_logging(full_score_list, train_epoch)
        return best_score
