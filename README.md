## Diffusion Domain Expansion: Learning to Coordinate Pre-trained Diffusion Models

This archive contains the code for reproducing our experiments. Before running the code, install the dependencies in `requirements.txt`. The next three section will provide an overview of how to reproduce the results in each of the domains: music, CLEVR, and maps.

We have uploaded the large files that are necessary (datasets, checkpoints) to [Google Drive](https://drive.google.com/drive/folders/1srdG-ySp7veHA7E6-jRns6yMU29Fnuiw?usp=sharing)

### Music

- **Dataset:** we used the standard Slakh2100 dataset for training. Here https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md you can find the instructions on how to download and preprocess the dataset.
- **Training:** run the script `music/train.py` to train the base model. You will need to specify in the code the path to the train dataset. To train any coordinator model, use the script music/train_coordinator.py and specify the path to the base model, path to the dataset and the type of coordinator architecture you want to train. 
- **Evaluation:** run the script `music/sample_tracks.py `first to sample tracks from the model. It supports multiple options, for example you can sample from MultiDiffusion("--multi") or Concatenation ("--concat"), or from the coordinator model, specified by "--rnn" option. After the sampling has finished, now you need to run music/calc_fad.py script, where you need to specify paths to the generated samples and the real dataset to get the FAD value.
### Cubes

- **Dataset:** we include the dataset in the Google Drive as clevr_pos_data_128_30000.npz. You need to specify path to this dataset in all the training scripts. 
- **Training:** run the script `clevr/train.py` to train the base model. You will need to fill in `dataset_path`parameter. To train the classifier, run the script `clevr/classifier/train.py`. You will need to fill in `data_path` parameter. Finally, to train the coordinator model, run the script `control/train.py`. You will need to fill in cube_model_path(= base model path), classifier_path and dataset_path parameters.
- **Evaluation:** run the script `clevr/evaluation/eval_script.py` to calculate accuracy.  You will need to fill in cube_model_path(= base model or coordinator model) and classifier_path. You can change the sampler you use and whether you want to do sum of scores model or not. 

### Maps

- **Dataset:** to generate the dataset, you will have to use Google Maps API. Run the `parser/main.py` script to generate a folder `dataset`, containing satellite-map pairs. Then, run the `parser/folder_to_npz.py` script to downsample the maps and convert the dataset to a single `.npz` file.

- **Training:** run the script `maps/cond/train.py` to train the map model. You will need to fill in `dataset_path`, `image_size` arguments. Setting `diti_devil` to False will train the base model, setting it to False would require `base_model_path`, and it will train a coordinator model.

- **Evaluation** run the script `maps/cond/eval.py` to evaluate the FID metric. You will need to fill in `dataset_path`, `image_size`, `model_path`, and `sample_cnt`. If you want to evaluate multidiffusion, set `multi_diffusion` to True, choose base size and stride, and use the checkpoint of the base model.

The Google Drive contains the base model used for evaluations, and two trained coordinator models.
