## Diffusion Domain Expansion: Learning to Coordinate Pre-trained Diffusion Models

This archive contains the code for reproducing our experiments. Before running the code, install the dependencies in `requirements.txt`. The next three section will provide an overview of how to reproduce the results in each of the domains: music, CLEVR, and maps.

We have uploaded the large files that are necessary (datasets, checkpoints) to [Google Drive](https://drive.google.com/drive/folders/1srdG-ySp7veHA7E6-jRns6yMU29Fnuiw?usp=sharing)

### Music

- **Dataset:** 

### Cubes

- **Dataset:**

### Maps

- **Dataset:** to generate the dataset, you will have to use Google Maps API. Run the `parser/main.py` script to generate a folder `dataset`, containing satellite-map pairs. Then, run the `parser/folder_to_npz.py` script to downsample the maps and convert the dataset to a single `.npz` file.

- **Training:** run the script `maps/cond/train.py` to train the map model. You will need to fill in `dataset_path`, `image_size` arguments. Setting `diti_devil` to False will train the base model, setting it to False would require `base_model_path`, and it will train a coordinator model.

- **Evaluation** run the script `maps/cond/eval.py` to evaluate the FID metric. You will need to fill in `dataset_path`, `image_size`, `model_path`, and `sample_cnt`. If you want to evaluate multidiffusion, set `multi_diffusion` to True, choose base size and stride, and use the checkpoint of the base model.

The Google Drive contains the base model used for evaluations, and two trained coordaintor models.
