import torch
from ml_logger import logger
from params_proto import ParamsProto

from clevr.control.model import ScoreSum
from clevr.evaluation.eval import evaluate_single_label, evaluate_control
from diffusion_2d.loader import load_model


class Args(ParamsProto):
    cube_model_path = "-"
    use_score_sum = True
    w = 20.0
    num_labels = 4
    sampler = 'heun'
    step = 'ode'
    classifier_path = "-"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    type = "control"
    image_size = 64
    seed = 25
    num_timesteps = 100
    batch_size = 1
    num_samples = 30
    log_images = True
    checkpoint = "-"

def main(**deps):
    Args._update(deps)

    print(logger)
    logger.log_text("""
           charts:
           - yKey: accuracy/mean
             xKey: epoch
           - yKey: images_per_sec/mean
             xKey: epoch
           - type: image
             glob: eval_images/0.png
           - type: image
             glob: eval_images/1.png
           - type: image
             glob: eval_images/2.png
           - type: image
             glob: eval_images/3.png
           """, ".charts.yml", dedent=True, overwrite=True)

    torch.manual_seed(Args.seed)
    logger.log_params(Args=vars(Args))
    if Args.use_score_sum:
        base_model = load_model(Args.cube_model_path, "checkpoints/model.pkl").to(Args.device)
        cube_model = ScoreSum(base_model)
    else:
        cube_model = load_model(Args.cube_model_path, Args.checkpoint).to(Args.device)
    classifier_model = load_model(Args.classifier_path, "checkpoints/model_best.pkl").to(Args.device)
    classifier_model.eval()

    if Args.type == "single_label":
        logger.store_metrics({
            'final_accuracy': evaluate_single_label(classifier_model, cube_model, Args)
        })
    elif Args.type == "control":
        logger.store_metrics({
            'final_accuracy': evaluate_control(classifier_model, cube_model, Args, num_labels=Args.num_labels)
        })
    else:
        raise ValueError(f"Unknown type {Args.type}")

    logger.log_metrics_summary()
