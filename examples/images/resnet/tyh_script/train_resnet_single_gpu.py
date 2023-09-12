import argparse
import time
import torch
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Optimizer
import torch.optim as optim
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator

# ==============================
# 超参数
# ==============================
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3


def train_epoch(epoch: int, model: nn.Module, optimizer: Optimizer, criterion: nn.Module, train_dataloader: DataLoader,
                booster: Booster, coordinator: DistCoordinator):
    start_time = time.time()  # 记录每个 epoch 所有 batch 的开始时间
    iterations_in_epoch = 0  # 记录每个 epoch 中所有 batch 的迭代次数和

    model.train()
    with tqdm(train_dataloader, desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}]', disable=not coordinator.is_master()) as pbar:
        for images, labels in pbar:
            iterations_in_batch = len(train_dataloader)  # 记录每个 batch 的迭代次数
            # print("当前batch大小：{}".format(iterations_in_batch))
            iterations_in_epoch += iterations_in_batch

            images = images.cuda()
            labels = labels.cuda()
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播 优化器
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # Print log info
            pbar.set_postfix({'loss': loss.item()})

    time_in_epoch = time.time() - start_time  # 记录每个 epoch 所有 batch 的结束时间

    average_iterations_per_second_in_epoch = iterations_in_epoch / time_in_epoch  # 单 GPU 每个 epoch 的平均迭代速度

    print("【单GPU】当前epoch耗时：{}，每秒迭代速度：{}"
          .format(time_in_epoch, average_iterations_per_second_in_epoch))

    a_i_p_s_i_e_tensor = torch.tensor(average_iterations_per_second_in_epoch, dtype=torch.float32, device='cuda')

    dist.all_reduce(a_i_p_s_i_e_tensor, op=dist.ReduceOp.SUM)  # 所有 GPU 的每秒迭代速度求和

    average_iterations_per_second_in_epoch = a_i_p_s_i_e_tensor.item()

    all_gpu_average_iterations_per_second_in_epoch = \
        average_iterations_per_second_in_epoch / dist.get_world_size()  # 所有 GPU 的每秒迭代速度均值

    if coordinator.is_master():
        print("【多GPU合并后】当前GPU数：{}，所有GPU每秒迭代次数：{}，每个gpu平均迭代速度：{}"
              .format(dist.get_world_size(), average_iterations_per_second_in_epoch,
                      all_gpu_average_iterations_per_second_in_epoch))

    return average_iterations_per_second_in_epoch


def main():
    # ==============================
    # 准备参数
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--plugin',
                        type=str,
                        default='torch_ddp',
                        choices=['torch_ddp', 'torch_ddp_fp16', 'low_level_zero'],
                        help="plugin to use")
    parser.add_argument('-r', '--resume', type=int, default=-1, help="resume from the epoch's checkpoint")
    parser.add_argument('-c', '--checkpoint', type=str, default='./colossalai_model/resnet50',
                        help="checkpoint directory")
    parser.add_argument('-i', '--interval', type=int, default=333, help="interval of saving checkpoint")
    parser.add_argument('--target_acc',
                        type=float,
                        default=None,
                        help="target accuracy. Raise exception if not reached")
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()

    NUM_EPOCHS = args.epoch

    # ==============================
    # 指定轮数保存模型
    # ==============================
    if args.interval > 0:
        Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    # ==============================
    # 启动分布式环境
    # ==============================
    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    # 更新学习率
    # old_gpu_num / old_lr = new_gpu_num / new_lr
    global LEARNING_RATE
    LEARNING_RATE *= coordinator.world_size

    # ==============================
    # 实例化Plugin和Booster
    # ==============================
    booster_kwargs = {}
    if args.plugin == 'torch_ddp_fp16':
        booster_kwargs['mixed_precision'] = 'fp16'
    if args.plugin.startswith('torch_ddp'):
        plugin = TorchDDPPlugin()
    elif args.plugin == 'gemini':
        plugin = GeminiPlugin(placement_policy='cuda', strict_ddp_mode=True, initial_scale=2 ** 5)
    elif args.plugin == 'low_level_zero':
        plugin = LowLevelZeroPlugin(initial_scale=2 ** 5)

    booster = Booster(plugin=plugin, **booster_kwargs)

    # ==============================
    # 准备数据
    # ==============================

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet的输入大小为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差
    ])
    '''

    transform = transforms.Compose(
        [transforms.Pad(4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32),
         transforms.ToTensor()]
    )
   '''
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # ====================================
    # 准备模型、优化器、准则
    # ====================================
    if args.model == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)

    # 将最后一层全连接层替换为适应CIFAR10数据集的输出类别数（10类）
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

    # ==============================
    # 启动 ColossalAI
    # ==============================
    model, optimizer, criterion, _, lr_scheduler = booster.boost(model,
                                                                 optimizer,
                                                                 criterion=criterion,
                                                                 lr_scheduler=lr_scheduler)

    # ==============================
    # 训练模型
    # ==============================
    start_epoch = args.resume if args.resume >= 0 else 0
    iterations_per_second_sum = 0  # 用于记录所有 epoch 的平均迭代次数和

    for epoch in range(start_epoch, NUM_EPOCHS):

        average_iterations_per_second_in_epoch = train_epoch(epoch, model, optimizer, criterion
                                                             , train_loader, booster, coordinator)
        lr_scheduler.step()

        iterations_per_second_sum += average_iterations_per_second_in_epoch

        # 保存模型
        if args.interval > 0 and (epoch + 1) % args.interval == 0:
            booster.save_model(model, f'{args.checkpoint}/model_{epoch + 1}.pth')
            booster.save_optimizer(optimizer, f'{args.checkpoint}/optimizer_{epoch + 1}.pth')
            booster.save_lr_scheduler(lr_scheduler, f'{args.checkpoint}/lr_scheduler_{epoch + 1}.pth')

    # 计算整个训练过程的平均每秒迭代次数
    average_iterations_per_second = iterations_per_second_sum / NUM_EPOCHS

    if coordinator.is_master():
        print(f"训练完成. 整个训练过程每秒迭代次数: {average_iterations_per_second:.2f}")


if __name__ == '__main__':
    main()
