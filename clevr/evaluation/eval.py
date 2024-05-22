import torch
import time
from diffusion_2d.utils import sample_image, sample_image_base, sample_control_image
from ml_logger import logger
from matplotlib import pyplot as plt


def get_accuracy(classifier_model, images, labels):
    batch_size = images.shape[0]
    image_size = images.shape[-1]
    num_labels = labels.shape[1]
    outputs = torch.sigmoid(classifier_model((images + 1) / 2))
    int_labels = torch.zeros_like(labels)
    int_labels[:, :, 0] = 1 - labels[:, :, 1]
    int_labels[:, :, 1] = labels[:, :, 0]
    int_labels = torch.round(int_labels * image_size).long()
    int_labels = torch.clip(int_labels, 0, image_size - 1)
    # int_labels of shape (batch_size, num_labels, 2)

    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_labels)
    label_indices_0 = int_labels[:, :, 0]
    label_indices_1 = int_labels[:, :, 1]
    probs = outputs[batch_indices, label_indices_0, label_indices_1]
    probs = (probs > 0.5).long()
    probs = torch.sum(probs, dim=1)
    all_labels = torch.eq(probs, num_labels).long()

    return torch.sum(all_labels).float() / batch_size


def evaluate_single_label(classifier_model, cube_model, args):
    batch_size = args.batch_size
    num_batches = args.num_samples // batch_size
    accuracy = 0.0
    logger.start('log_timer')
    for epoch in range(num_batches):
        with torch.no_grad():
            fx = torch.rand((batch_size,)) * 0.4 + 0.3
            fy = torch.rand((batch_size,)) * 0.4 + 0.3
            y = torch.stack([fx, fy], dim=1).to(args.device)
            images = sample_image(cube_model, y, args, batch_size=batch_size)

            if args.log_images:
                fig, axs = plt.subplots(3, 3, figsize=(8, 8))
                for i in range(9):
                    image = images[i]
                    np_image = image.cpu().detach().numpy().transpose(1, 2, 0)
                    np_image = (np_image + 1) / 2.0
                    row = i // 3
                    col = i % 3
                    axs[row, col].imshow(np_image, interpolation='nearest')
                    axs[row, col].scatter(fx[i] * args.image_size,
                                          (1 - fy[i]) * args.image_size,
                                          color='red',
                                          marker='x',
                                          s=80)
                    axs[row, col].axis('off')
                plt.tight_layout()
                plt.show()
                logger.savefig(f"eval_images/{epoch}.png")

            if args.log_images:
                outputs = torch.sigmoid(classifier_model((images + 1) / 2))
                fig, axs = plt.subplots(3, 3, figsize=(8, 8))
                for i in range(9):
                    image = outputs[i]
                    np_image = image.cpu().detach().numpy()
                    row = i // 3
                    col = i % 3
                    axs[row, col].imshow(np_image, cmap='hot', interpolation='nearest')
                    axs[row, col].scatter(fx[i] * args.image_size,
                                          (1 - fy[i]) * args.image_size,
                                          color='red',
                                          marker='x',
                                          s=120)
                    axs[row, col].axis('off')
                plt.tight_layout()
                plt.show()
                logger.savefig(f"classifier/{epoch}.png")

            labels = y.unsqueeze(1)
            batch_accuracy = get_accuracy(classifier_model, images, labels).cpu().item()
            accuracy += batch_accuracy

            logger.store_metrics(accuracy=batch_accuracy)

            logger.store_metrics({
                'images_per_sec': batch_size / logger.split('log_timer')
            })

            logger.log_metrics_summary(key_values={"epoch": epoch})
            logger.print(f'Completed epoch:{epoch}, time = {time.asctime(time.localtime())}')
    return accuracy / num_batches


def evaluate_control(classifier_model, control_model, args, num_labels=2):
    batch_size = args.batch_size
    num_batches = args.num_samples // batch_size
    accuracy = 0.0
    logger.start('log_timer')
    for epoch in range(num_batches):
        with torch.no_grad():
            coords = torch.zeros((batch_size, num_labels, 2), device=args.device)
            for i in range(batch_size):
                while True:
                    coords[i] = torch.rand((num_labels, 2)).to(args.device) * 0.4 + 0.3
                    bad = False
                    for j in range(num_labels):
                        for k in range(j):
                            if ((coords[i][j][0] - coords[i][k][0])**2 + (coords[i][j][1] - coords[i][k][1])**2) ** 0.5 < 0.15:
                                bad = True
                                break
                        if bad:
                            break
                    if not bad:
                        break
            images = sample_image_base(lambda x, t: control_model(x, t, coords), args, batch_size) if args.use_score_sum else sample_control_image(control_model, coords, args, batch_size=batch_size, num_labels=num_labels)

            batch_accuracy = get_accuracy(classifier_model, images, coords).cpu().item()
            accuracy += batch_accuracy

            logger.store_metrics(accuracy=batch_accuracy)

            logger.store_metrics({
                'images_per_sec': batch_size / logger.split('log_timer')
            })

            if args.log_images:
                coords = coords.cpu().detach().numpy()
                fig, axs = plt.subplots(3, 3, figsize=(8, 8))
                for i in range(9):
                    image = images[i]
                    np_image = image.cpu().detach().numpy().transpose(1, 2, 0)
                    np_image = (np_image + 1) / 2.0
                    row = i // 3
                    col = i % 3
                    axs[row, col].imshow(np_image, interpolation='nearest')
                    for j in range(num_labels):
                        axs[row, col].scatter(coords[i][j][0] * args.image_size,
                                              (1 - coords[i][j][1]) * args.image_size,
                                              color='red',
                                              marker='x',
                                              s=80)
                    axs[row, col].axis('off')
                plt.tight_layout()
                plt.show()
                logger.savefig(f"eval_images/{epoch}.png")

            logger.log_metrics_summary(key_values={"epoch": epoch})
            logger.print(f'Completed epoch:{epoch}, time = {time.asctime(time.localtime())}')
    return accuracy / num_batches
