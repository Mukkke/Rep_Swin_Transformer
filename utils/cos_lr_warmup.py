import math

def cosine_warmup_schedule(epoch, config):
    warmup_epochs = config['TRAIN']['WARMUP_EPOCHS']
    max_epochs = config['TRAIN']['EPOCHS']
    warmup_lr = float(config['TRAIN']['WARMUP_LR'])
    min_lr = float(config['TRAIN']['MIN_LR'])
    base_lr = float(config['TRAIN']['BASE_LR'])

    if epoch < warmup_epochs:
        # Linear warmup
        return warmup_lr + epoch * (base_lr - warmup_lr) / warmup_epochs
    else:
        # Cosine annealing after warmup
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos((epoch - warmup_epochs) / (max_epochs - warmup_epochs) * math.pi))

