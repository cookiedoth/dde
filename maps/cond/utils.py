from maps.uncond.utils import sample_images, save_images
from ml_logger import logger
import torch


def save_samples(loc, samples, x_c, gmap, start_from):
    concat = torch.cat((x_c, gmap, samples), dim=3) if gmap is not None else \
             torch.cat((x_c, samples), dim=3)
    save_images(samples, loc, 'samples', start_from)
    save_images(concat, loc, 'all', start_from)


def sample_images_batched(model, x_c, args, loc=None, gmap=None):
    samples = []
    pos = 0
    sample_cnt = x_c.shape[0]
    while pos < sample_cnt:
        logger.print(f'Sampling, {pos}/{sample_cnt}')
        batch_size = min(sample_cnt - pos, args.batch_size) 
        batch_x_c = x_c[pos:pos+batch_size]
        batch_samples = sample_images(lambda x, t: model(x, t, batch_x_c),
                                      batch_size,
                                      args)
        samples.append(batch_samples)
        if loc is not None and gmap is not None:
            save_samples(loc, batch_samples, batch_x_c, gmap[pos:pos+batch_size], pos)
        pos += batch_size
    return torch.cat(samples, dim=0)
