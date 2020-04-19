from torchvision.models.resnet import resnet18
from optimizer import Ranger
from optimizer.lr_scheduler import DelayerScheduler
import torch


if __name__ == '__main__':
    model = resnet18()
    # Let's have different LRs for weight and biases for instance
    bias_params, weight_params = [], []
    for n, p in model.named_parameters():
        if n.endswith('.bias'):
            bias_params.append(p)
        else:
            weight_params.append(p)
    # We pass the parameters to the optimizer
    optimizer = Ranger([dict(params=weight_params, lr=2e-4), dict(params=bias_params, lr=1e-4)])

    steps = 25
    base_lr = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=4)
    scheduler = DelayerScheduler(optimizer, delay_epochs=10, after_scheduler=base_lr)

    # Let's record the evolution of LR in each group
    lrs = [[], []]
    for step in range(steps):
        for idx, group in enumerate(optimizer.param_groups):
            lrs[idx].append(group['lr'])
        # Train your model and perform optimizer.step() here
        scheduler.step()

    # And plot the result
    import matplotlib.pyplot as plt
    plt.plot(lrs[0], label='Weight LR'); plt.plot(lrs[1], label='Bias LR'); plt.legend(); plt.show()