import torch


class Param(object):
    def __init__(self):
        super(Param, self).__init__()
        # Path
        self.ROOT = 'C:/Users/rlawj'
        self.DATASET_PATH = f'{self.ROOT}/sample_DB'
        self.OUTPUT_CKP = f'{self.ROOT}/backup/ckp'
        self.OUTPUT_LOG = f'{self.ROOT}/backup/log'
        self.CKP_LOAD = False
        self.LOAD_CKP_EPCOH = 0

        # Data
        self.INPUT_SIZE = 256
        self.AUG = False

        # Train or Test
        self.EPOCH = 300
        self.VALID_EPOCH = 10
        self.LR = 1e-4
        self.BATCHSZ = 1
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.PRINT_DISPLAY = True

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 0