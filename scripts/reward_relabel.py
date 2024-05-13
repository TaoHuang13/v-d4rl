import hydra
from pathlib import Path
import pathlib
import h5py
import numpy as np
import torch

from diffusion_reward.models.reward_models.amp import AMP
from diffusion_reward.models.reward_models.diffusion_reward import DiffusionReward
from diffusion_reward.models.reward_models.rnd import RND
from diffusion_reward.models.reward_models.viper import VIPER


def make_rm(cfg):
    if cfg.rm_model == 'diffusion_reward':
        rm = DiffusionReward(cfg=cfg)
    elif cfg.rm_model == 'viper':
        cfg.cfg_path = str(Path(__file__).parents[3]) + cfg.cfg_path
        cfg.ckpt_path = str(Path(__file__).parents[3]) + cfg.ckpt_path
        rm = VIPER(cfg=cfg)
    elif cfg.rm_model == 'amp':
        rm = AMP(cfg)
    elif cfg.rm_model == 'rnd':
        rm = RND(cfg)
    return rm

@hydra.main(config_path='/home/lucas/eccv/diffusion_reward/diffusion_reward/configs/rl', config_name='default')
def main(cfgs):
    device = 'cuda:0'
    cfgs.reward.cfg_path = '/home/lucas/eccv/diffusion_reward/exp_local/video_models/vqdiffusion/dmc/.hydra/config.yaml'
    cfgs.reward.ckpt_path = '/home/lucas/eccv/diffusion_reward/exp_local/video_models/vqdiffusion/dmc/checkpoint/best.pth'
    cfgs.reward.obs_shape = None
    cfgs.reward.action_shape = None
    cfgs.reward.use_std = False
    rm = make_rm(cfgs.reward).to(device)

    task = 'cheetah_run'
    data_type = 'expert'
    input_dir = f'/home/lucas/eccv/v-d4rl/vd4rl_data/main/{task}/{data_type}/raw_data'
    output_dir = f'/home/lucas/eccv/v-d4rl/vd4rl_data/main/{task}/{data_type}/vqdiffusion_relabel'

    output = {}

    filenames = sorted(pathlib.Path(input_dir).glob('*.hdf5'))
    for filename in filenames:
        with h5py.File(filename, "r") as f:
            actions = f['action'][:]
            observations = f['observation'][:]
            rewards = f['reward'][:]
            discounts = f['discount'][:]
            step_types = f['step_type'][:]
            done = np.array(f['step_type'][:]) == 2

    end_idxes = np.where(done == 1)[0]
    start_idx = 0
    all_rewards = []
    for end_idx in end_idxes:
        print(start_idx, end_idx)
        # import ipdb; ipdb.set_trace()
        video_clip = observations[start_idx:end_idx].transpose(0, 2, 3, 1)
        
        video_clip = torch.Tensor(video_clip)[None] / 127.5 - 1
        print(video_clip.max(), video_clip.min())
        pred_rewards = rm.calc_reward(video_clip.cuda())
        # TODO!, add statistics
        rewards[start_idx:end_idx] = pred_rewards.cpu().numpy()
        start_idx = end_idx + 1

    output['observation'] = observations
    output['action'] = actions
    output['reward'] = rewards / 100
    output['discount'] = discounts
    output['step_type'] = step_types

    print(rewards.mean(), output['reward'].mean())

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    with h5py.File(out_dir / 'data_relabeled.hdf5', 'w') as shard_file:
        for k, v in output.items():
            shard_file.create_dataset(k, data=v, compression='gzip')



if __name__ == '__main__':
    main()