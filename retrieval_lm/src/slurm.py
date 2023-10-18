# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
import os
import sys
import torch
import socket
import signal
import subprocess


logger = getLogger()

def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    else:
        logger.warning("Not the main process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - local_rank
        - global_rank
        - world_size
    """
    is_slurm_job = 'SLURM_JOB_ID' in os.environ and not 'WORLD_SIZE' in os.environ
    has_local_rank = hasattr(params, 'local_rank')

    # SLURM job without torch.distributed.launch
    if is_slurm_job and has_local_rank:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])
        params.world_size = int(os.environ['SLURM_NTASKS'])

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.main_addr = hostnames.split()[0].decode('utf-8')
        assert 10001 <= params.main_port <= 20000 or params.world_size == 1

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.main_addr
        os.environ['MASTER_PORT'] = str(params.main_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)
        is_distributed = True


    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif has_local_rank and params.local_rank != -1:

        assert params.main_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])

        is_distributed = True

    # local job (single GPU)
    else:
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        is_distributed = False

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if is_distributed:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        #print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
            #world_size=params.world_size,
            #rank=params.global_rank,
        )