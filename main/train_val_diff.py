import argparse
import logging
import os
import pprint
import shutil
import time
import yaml
from easydict import EasyDict
import sys

sys.path.append(os.path.split(sys.path[0])[0])

import torch
import torch.distributed as dist
import torch.optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from lib.datasets.data_builder import build_dataloader
from lib.models.model_helper import ModelHelper
from lib.utils.criterion_helper import build_criterion
from lib.utils.dist_helper import setup_distributed
from lib.utils.eval_helper import dump, log_metrics, merge_together, performances
from lib.utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
)
from lib.utils.optimizer_helper import get_optimizer, get_scheduler
from lib.utils.vis_helper import visualize_compound, visualize_single

parser = argparse.ArgumentParser(description="Training Diffusion AD model.")

parser.add_argument("-c", "--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("-g", "--gpu", default="0", help="indices of gpus")
parser.add_argument('--local_rank', default=-1, help='local rank for dist',type = int)
parser.add_argument('--class_name', default=None, help='special if train single cls',type = str)

parser.add_argument('--random_seed', default=None, help='for changing cfg',type=int)


def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    rank, world_size = setup_distributed(port=config.port)

    current_time = get_current_time()

    config = update_config(config)

    config.exp_path = os.path.dirname(args.config)
    if args.class_name is not None:
        config.dataset.train.meta_file = \
            config.dataset.train.meta_file.replace("{class_name}", args.class_name)
        config.dataset.test.meta_file = \
            config.dataset.test.meta_file.replace("{class_name}", args.class_name)
        config.exp_path = os.path.join(config.exp_path, args.class_name)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, current_time + config.evaluator.save_dir)

    save_every_epoch = config.saver.save_step
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    # random_seed = config.get("random_seed", None)
    random_seed = config.get("random_seed", None) if args.random_seed is None else args.random_seed
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    model = ModelHelper(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    parameters = [
        {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)

    train_loader, val_loader = build_dataloader(config.dataset, distributed=True)

    if args.evaluate:
        validate(val_loader, model)
        return

    criterion = build_criterion(config.criterion)

    for epoch in range(last_epoch, config.trainer.max_epoch):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
        )
        lr_scheduler.step(epoch)

        #########################################################
        if rank == 0 and (
                ((epoch + 1) % save_every_epoch == 0) or
                ((epoch + 1) == config.trainer.max_epoch) or
                epoch == 0):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": config.net,
                    "state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "optimizer": optimizer.state_dict(),
                },
                False,
                config,
            )
        #########################################################

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            ret_metrics = validate(val_loader, model)
            if rank == 0:
                ret_key_metric = ret_metrics[key_metric]
                is_best = ret_key_metric >= best_metric
                best_metric = max(ret_key_metric, best_metric)
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": config.net,
                        "state_dict": model.state_dict(),
                        "best_metric": best_metric,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    config,
                )


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
):

    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()

    for layer in frozen_layers:
        module = getattr(model.module, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        data_time.update(time.time() - end)

        outputs = model(input)
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        optimizer.zero_grad()
        loss.backward()

        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        model.module.ema_update()

        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()


def validate(val_loader, model):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    dist.barrier()

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            outputs = model(input)
            dump(config.evaluator.eval_dir, outputs)

            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

    dist.barrier()
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    dist.all_reduce(total_num, async_op=True)
    dist.all_reduce(loss_sum, async_op=True)
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {}
    if rank == 0:
        logger.info("Gathering final results ...")
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        fileinfos, preds, masks = merge_together(config.evaluator.eval_dir, use_gauss=True)
        shutil.rmtree(config.evaluator.eval_dir)

        ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)
        if config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                preds,
                masks,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
                config.exp_path,
            )
        if config.evaluator.get("vis_single", None):
            visualize_single(
                fileinfos,
                preds,
                config.evaluator.vis_single,
                config.dataset.image_reader,
                config.exp_path,
            )
    model.train()
    return ret_metrics


if __name__ == "__main__":
    main()
