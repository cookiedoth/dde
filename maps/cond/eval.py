from diffusion_2d.loader import load_model
import torch
from params_proto import ParamsProto, Proto
from ml_logger import logger
from maps.cond.dataset import SatMapDataset, merge_sat_map
from maps.uncond.utils import TimedAction, random_str, save_images, find_fid_loc, get_path
from maps.cond.utils import sample_images_batched
from maps.cond.model import MultiDiffusion, FakeCondModel
from maps.uncond.model import get_kernel
import random


t1_path = '/diffusion-comp/2024/05-17/maps/cond/sweep_train/00.10.12/1/1'
t1_pkl = 'checkpoints/model_0100.pkl'

t2_path = '/diffusion-comp/2024/05-20/maps/cond/sweep_train/02.04.15/1/1'
t2_pkl = 'checkpoints/model_0050.pkl'

base_path = '/diffusion-comp/2024/05-14/maps/cond/sweep_train/13.49.21/1/1/'
base_pkl = 'checkpoints/model_0200.pkl'


t3_path = '/diffusion-comp/2024/05-21/maps/cond/sweep_train/21.35.02/1/1/'
t4_path = '/diffusion-comp/2024/05-21/maps/cond/sweep_train/21.56.39/1/1'



def get_pkl(epoch):
    return f'checkpoints/model_{epoch:04}.pkl'


class Args(ParamsProto):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path = Proto(env='$DATASET_ROOT/maps/maps128.npz')
    flip_cond = True
    image_size = 128
    seed = 42
    batch_size = 64
    num_timesteps = 150

    sample_cnt = 512

    sigma_min = 1e-4
    sigma_max = 20.0
    rho = 7.0
    s_churn = 20.0

    model_path = t2_path
    model_pkl = get_pkl(20)

    multi_diffusion = False
    base_size = 64
    stride = 32


def init_multidiff(base_model):
    return MultiDiffusion(base_model=base_model,
                          base_size=Args.base_size,
                          target_size=Args.image_size,
                          stride=Args.stride,
                          kernel=get_kernel(Args.base_size, 1.0).to(Args.device))


def main(**deps):
    Args._update(deps)
    logger.log_params(Args=vars(Args))
    torch.manual_seed(Args.seed)

    with TimedAction('initialize_model'):
        model = load_model(Args.model_path, Args.model_pkl).to(Args.device)
        # model = FakeCondModel()
        if Args.multi_diffusion:
            model = init_multidiff(model)

    with TimedAction('load_dataset'):
        dataset = SatMapDataset(Args.dataset_path, Args.image_size, flip_cond=Args.flip_cond)

    loc = random_str(10)
    logger.print('Sampling...')
    logger.print(f'Location: {get_path(loc)}')
    indices = random.sample(range(len(dataset)), Args.sample_cnt)
    x_c, gmap = merge_sat_map(dataset, indices, Args.device)

    # This should save samples
    model.eval()
    _ = sample_images_batched(model, x_c, Args, loc=loc, gmap=gmap)

    _, dataset_samples = merge_sat_map(dataset, range(len(dataset)), Args.device)
    save_images(dataset_samples, loc, 'real')

    fid = find_fid_loc(loc, cleanup=False)
    logger.print(f'FID score: {fid}')


if __name__ == '__main__':
    main()
