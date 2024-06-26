from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from source.datasets.dataset_miniscapes import DatasetMiniscapes
from source.models import *



def resolve_dataset_class(name):
    return {
        'miniscapes': DatasetMiniscapes,
    }[name]


def resolve_dataset_class(name):
    return {
        'miniscapes': DatasetMiniscapes,
    }[name]

def resolve_model_class(name):
    return {
        'deeplabv3p': ModelDeepLabV3Plus,
        'adaptive_depth': ModelAdaptiveDepth,
        'deeplabv3p_multitask': ModelDeepLabV3PlusMultiTask,
        'deeplabv3p_multitask_task5': ModelDeepLabV3PlusMultiTask_task5,
    }[name]


def resolve_optimizer(cfg, params):
    if cfg.optimizer == 'sgd':
        return SGD(
            params,
            lr=cfg.optimizer_lr,
            momentum=cfg.optimizer_momentum,
            weight_decay=cfg.optimizer_weight_decay,
        )
    elif cfg.optimizer == 'adam':
        return Adam(
            params,
            lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_weight_decay,
        )
    else:
        raise NotImplementedError


def resolve_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'poly':
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / cfg.num_epochs) ** cfg.lr_scheduler_power)
        )
    else:
        raise NotImplementedError

