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

import faiss
import numpy as np
from einops import rearrange
from sklearn.random_projection import SparseRandomProjection

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
    set_random_seed,
    update_config,
)
from lib.utils.vis_helper import visualize_compound, visualize_single
from lib.tools.kcenter_greedy import kCenterGreedy

parser = argparse.ArgumentParser(description="Training for or test with memory bank.")

parser.add_argument("-c", "--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("-g", "--gpu", default="0", help="indices of gpus")
parser.add_argument("--coreset_sampling_ratio", default=0.01, help="save dir of membank", type=float)
parser.add_argument('--n_neighbors', default=3, help='k of knn in validate', type=int)
parser.add_argument('--only_membank', action="store_true", help='if not use reconstruction')
parser.add_argument('--local_rank', default=-1, help='local rank for dist', type=int)
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
        config.membank_dir = \
            config.membank_dir.replace("{class_name}", args.class_name)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, current_time + config.evaluator.save_dir)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))

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

    key_metric = config.evaluator["key_metric"]
    best_metric = 0

    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)

    membank_folder_dir = config.membank_dir
    if not os.path.exists(membank_folder_dir):
        os.makedirs(membank_folder_dir)

    membank_save_dir = os.path.join(membank_folder_dir, 'membank{:d}.faiss'
            .format(int(args.coreset_sampling_ratio * 100)))
    if os.path.exists(membank_save_dir) and (not args.evaluate):
        swi = input('A memory bank already exists, do you want to retrain it? [y/N]: ')
        if swi not in ['y', 'Y', 'Yes', 'yes', 'YES']:
            raise ValueError('A memory bank already exists, stop training.')
    if args.evaluate and os.path.exists(membank_save_dir):
        membank_model = faiss.read_index(membank_save_dir)
    else:
        membank_model = faiss.IndexFlatL2(model.module.neck.get_outplanes()[0])

    train_loader, val_loader = build_dataloader(config.dataset, distributed=True)

    if args.evaluate:
        if not os.path.exists(membank_save_dir):
            train_one_epoch(train_loader, model, membank_model, membank_save_dir)
        validate(val_loader, model, membank_model)
        return

    train_loader.sampler.set_epoch(1)
    train_one_epoch(train_loader, model, membank_model, membank_save_dir)


def train_one_epoch(train_loader, model, membank_model, membank_save_dir=None, neck_stop=True):
    batch_time = AverageMeter(0)

    model.eval()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    end = time.time()

    dist.barrier()

    embedding_list = []
    with torch.no_grad():
        for i, input in enumerate(train_loader):
            outputs = model(input, neck_stop=neck_stop)
            embedding_list.append(outputs['feature_align'].cpu())

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
                logger.info(
                    "Train menbank: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(train_loader), batch_time=batch_time
                    )
                )

    dist.barrier()

    if True:
        embedding_list = torch.cat(embedding_list, dim=0)
        embedding_list = rearrange(embedding_list, 'N C H W -> (N H W) C').cpu().numpy()

    try:
        # Johnson-Lindenstrauss lemma
        randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)
        logger.info('=> Fiting Johnson-Lindenstrauss lemma model...')
        randomprojector.fit(embedding_list)
    except:
        randomprojector = None
        logger.info('Features are too large, drop random linear step.')

    if abs(args.coreset_sampling_ratio - 1) > 1e-6:
        logger.info('=> Coreset subsampling...')
        selector = kCenterGreedy(embedding_list, 0, 0)
        selected_idx = selector.select_batch(model=randomprojector, already_selected=[],
                N=int(embedding_list.shape[0] * args.coreset_sampling_ratio))
        embedding_coreset = embedding_list[selected_idx]
    else:
        embedding_coreset = embedding_list
    logger.info('initial embedding size : ' + str(embedding_list.shape))
    logger.info('final embedding size : ' + str(embedding_coreset.shape))

    logger.info('=> Faiss adding...')
    membank_model.add(embedding_coreset)

    if membank_save_dir is not None:
        logger.info('=> Faiss saving...')
        if torch.cuda.is_available() and (args.coreset_sampling_ratio < 1):
            membank_model_save = membank_model
        else:
            membank_model_save = membank_model
        faiss.write_index(membank_model_save, membank_save_dir)

    logger.info('Finish training.')
    return embedding_coreset

def validate(val_loader, model, membank_model):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    logger.info("Test with memory bank.")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)

    dist.barrier()

    logger.info('Test embedding number : ' + str(membank_model.ntotal))

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            outputs = model(input, membank_model, n_neighbos=args.n_neighbors,
                    only_membank=args.only_membank)
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

            if args.only_membank and config.net[-1]['kwargs']['save_recon']:
                save_recon_dict = config.net[-1]['kwargs']['save_recon']
                clsnames = outputs["clsname"]
                filenames = outputs["filename"]
                for clsname, filename, feat_rec in zip(clsnames, filenames, outputs['feature_rec']):
                    filedir, filename = os.path.split(filename)
                    _, defename = os.path.split(filedir)
                    filename_, _ = os.path.splitext(filename)
                    recon_save_dir = os.path.join(save_recon_dict['save_dir'],
                                            model.module.reconstruction.__class__.__name__, clsname, defename)
                    os.makedirs(recon_save_dir, exist_ok=True)
                    feature_rec_np = feat_rec.detach().cpu().numpy()
                    np.save(os.path.join(recon_save_dir, filename_ + ".npy"), feature_rec_np)

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
