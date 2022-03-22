import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils import inf_loop, MetricTracker
import wandb
# TODO : 이거 from base import BaseTrainer가 원본이다. 왜 그런지 이유 찾기
# TODO : 내 프로젝트 고유의 것은 없으려나...? 없어보이기는 함.

class Trainer(BaseTrainer):
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        self.device = device
        if len_epoch is None:
            # 뭔가 assert 문으로 처리해도 될 듯 함.
            self.len_epoch = len(self.train_data_loader)
        else:
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = len(self.train_data_loader)//10
        # log를 batch_size의 제곱근마다 해라?
        # TODO : log_step 국룰 찾아보기.

        # TODO : MetricTracker에 대한 정확한 이해
        # 현재의는 loss + self.metric_ftns(acc, f1)들의 평균 total을 tracking하는 class
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Args:
            epoch (int): Length of epoch of the training process currently in progress
        """
        self.model.train()
        self.train_metrics.reset()
        output_list = []
        target_list = []
        # TODO : 해당 for문을 돌았을 때 data를 얼마나 iteration을 하는지? 그리고 해당 과정은 몇개의 step이라고 말할 수 있는지
        # TODO : 
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            # TODO : Data도 GPU에 보내고 있음
            data, target = data.to(self.device), target.to(self.device)
            # TODO : 얘 역할 알아내기. 예상으로는 아래 writer.set_step에 작성하는 걸로 보아 
            # logging 하는 step 단위를 적는게 아닐까...?
            example_ct = (epoch - 1) * self.len_epoch + batch_idx
            self.optimizer.zero_grad()
            
            output = self.model(data)
            # 특이한 것들은 model의 loss를 계산하기 어렵다.
            # 뭔가 특이하게 나오므로 아래와 같이 사용하면 작성했다.
            if self.model.__class__.__name__ == "SomethingSpecial":
                output = self.model.SomethingSpecial(output, target)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # self.writer.set_step(example_ct) 
            self.train_metrics.update('loss', loss.detach())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
            
            if batch_idx % self.log_step == 0:
                # TODO :현재는 logger로 만들어져 있지만 logger가 아니라 Tqdm으로 했으면 더 좋지 않았을까 싶음.
                # TODO :logger에 정확한 기능이 무엇인지 공부해야할 듯 함.
                self.logger.debug('Train Epoch : {} {} Loss :{:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.detach()
                ))
                if self.cfg_wandb['use']:
                    wandb.log({"loss" : loss.detach()}, step=example_ct)

            output_list.extend(torch.argmax(output, dim=1).tolist())
            target_list.extend(target.tolist())
            
            if batch_idx == self.len_epoch:
                # 이렇게 data_loader를 끝낼 수 있는 듯 하다.
                break 
        # dict 형태로 metrics를 return 해줌. 아마 config의 metrics 이름과 loss로 접근해 값을 얻을 수 있을 듯 함.
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            if self.cfg_wandb['use']:
                wandb.log({'val_'+k : v for k, v in val_log.items()}, step=example_ct)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return log, target_list, output_list

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        Args:
            epoch (int): current training epoch
        
        return : A log that contains information abuout validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                if self.model.__class__.__name__ == "SomethingSpecial":
                    output = self.model.SomethingSpecial(output, target)
                loss = self.criterion(output, target)
                
                # TODO : 아래의 코드 존재 이유 알아내기
                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.detach())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # TODO : 아래의 코드 존재 이유 알아내기 아마 tensorboard때문일 듯.
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        # TODO : logging tool은 1개로 종합하는게 좋은가? 그럼 언떤걸로 종합하는게 좋을지 고민하기. 상황과 장비 등등을 고려 한 선택

        # TODO : 아래 코드 어떻게 할 지 고민하기
        # 갑자기 tensorboard가 나온 이유는 template에서 tensorboard를 쓰도록 basetrainer에 writer를 지정해놓음.
        # 바꾸기에는 코드가 너무 좋아서 tensorboard로 진행해볼까
        # add fistogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histrogram(name, p, bins='auto')
        return self.valid_metrics.result()
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)