from pathlib import Path

import jaynes
from ml_logger.job import instr
from params_proto.hyper import Sweep

from run import RUN
from diffusion_cubes.evaluation.eval_script import main, Args
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu')
    parser.add_argument('--hostname')
    args = parser.parse_args()
    jaynes.config('tjlab-gpu',
                  launch=dict(ip=args.hostname),
                  verbose=True)
    with Sweep(RUN, Args) as sweep:
        with sweep.product:
            Args.num_labels = [5]


    @sweep.each
    def tail(RUN, Args):
        RUN.job_name = f"{{now:%H.%M.%S}}/num_labels:{Args.num_labels}"


    sweep.save(f'{Path(__file__).stem}.jsonl')

    gpus_to_use = [args.gpu]
    gpu_id = 0

    for kwargs in sweep:
        RUN.CUDA_VISIBLE_DEVICES = str(gpus_to_use[gpu_id % len(gpus_to_use)])
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)
        gpu_id += 1

    jaynes.listen()
    kwargs = {'RUN.job_name': '{{now:%H.%M.%S}}'}
    thunk = instr(main, **kwargs)
    jaynes.run(thunk)
    jaynes.listen()
