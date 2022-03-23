import cv2
import utils 
import torch
import wandb
import argparse
import collections
import numpy as np
import data_loader.data_loaders_crop as Custom_loader
import data_loader.transforms as Custom_transforms
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

from parse_config import ConfigParser
from trainer import Trainer


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # config_parser는 말 그대로 dict type이 맞음 []로 접근할 수 있음.
    def _main(config):
        logger = config.get_logger('train')
        if config["is_Preprocessing"]:
            utils.make_necessary_dir(config['Prepared_Data_Path'], config['Model_Save_Path'])
            utils.raw_image_json = utils.get_image_json_path(config["Raw_Data_Path"])
            utils.get_Resized_Image_Dataset(config)
            utils.Set_Dataset_CSV(config)

        # Training Process with Config Parser
        # TODO : logger가 어떤 transform이 붙었는지 확인해야하지 않나?
        train_transform, default_transform = config.init_ftn(
            "transform_name", Custom_transforms
        )()
        # TODO config.json에 trainsform_name란을 만들면 된다(type과 args를 추가)
        Dataloader = config.init_obj(
            "Data_Loader",
            Custom_loader,
            transform=train_transform,
            default_transform=default_transform,
            config=config,
        )

        train_data_loader, valid_data_loader = Dataloader.split_validation()

        model = config.init_obj('arch', module_arch)
        # TODO : logger가 현재 어떤 역할을 하는지 알아보기. -> 해결됨
        # TODO : model에 대한 정보도 log로 찍고 싶긴 함.
        logger.info(model) 

        device, _ = utils.prepare_device(config['n_gpu'])
        model = model.to(device)
                                                                                               
        # TODO : 다양한 loss function이 어떻게 만들어지고 정의되는지, Custom loss function에 대해 공부해보기
        criterion = getattr(module_loss, config['loss']['type'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(model, criterion, metrics, optimizer,
                    config = config,
                    device=device,
                    train_data_loader=train_data_loader,
                    valid_data_loader=valid_data_loader,
                    lr_scheduler=lr_scheduler)

        trainer.train()
    
    if config['wandb']['use']:
        with wandb.init(name=config['name'], project=config['wandb']['args']['project'],
                        entity=config['wandb']['args']['entity'], config=config):
            wandb_config = wandb.config
            _main(config)
    else:
        _main(config)
        

# TODO :  Preprocess 절차를 생략할지 말지 정하는 config 와 argparser가 있으면 좋을 듯 하다.
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    # wandb.login()
    main(config)
