# Fine-tuning AdaFace on images from news papers

### This is a fork of the AdaFace (https://github.com/mk-minchul/AdaFace) repository.

This repository contains a modified version of the AdaFace repository. 
The main modifications were done to the following files:
- `evaluate_utils.py`
- `requirements.txt`
- `train_val.py`
- `utils.py`
- `run_ir50_ms1mv2.sh`

The modifications were done to support a different dataset than what was originally used to train the model and 
especially to make the model runnable on the _metacentrum_ clusters.

## Usage guide

The model was fine-tuned and tested on _metacentrum_. We have had issues when running the scripts on various
different nodes. We have had success with the `zia` and `galdor` clusters. Other ones didn't work.

The first step is to create a `singularity` (if used on _metacentrum_). To do this, run the following command:
```
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:21.09-py3.SIF
```

It is important to use this `singularity`, because the scripts only support `python3.8` and this singularity comes with 
this particular python distribution. (we tried our best to make it run on newer python versions but we failed)

### Installation
The original `requirements.txt` file didn't work on _metacentrum_ at all. When we tried fixing it,
we were unsuccessful. The only solution we found was to manually install some packages one by one using pip. 
Pasting in the sequence of commands below should install all required packages without issues.

```
conda create --name adaface pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 python=3.8.10 -c pytorch
source /opt/conda/bin/activate adaface
conda install matplotlib 
pip install nvidia_cublas_cu11==11.10.3.66 --no-cache-dir
pip install nvidia_cudnn_cu11==8.5.0.96 --no-cache-dir
pip install opencv_python_headless==4.9.0.80 --no-cache-dir
pip install scikit_learn==1.3.2 --no-cache-dir
pip install PyWavelets==1.4.1 --no-cache-dir
pip install aiohttp==3.9.5 --no-cache-dir
pip install menpo --no-cache-dir
pip install -r requirements.txt --no-cache-dir
```


### Testing

To test the model, make sure your pre-trained model `adaface_ir50_ms1mv2.ckpt` is stored in the `/pretrained/` directory.
Then you need to prepare your testing dataset in the `.bin` format. Name it `lfw.bin` and place it in the `/data/faces_webface_112x112/` directory.
Afterward, run the following command to prepare the testing set to be used:
```
python3.8 convert.py --rec_path ./data/faces_webface_112x112 --make_validation_memfiles
```
After that, the data set will be ready, and you can run the testing via the following command:
```
python3.8 main.py --data_root ./data/ --val_data_path faces_webface_112x112 --prefix adaface --use_mxrecord --start_from_model_statedict ./pretrained/adaface_ir50_ms1mv2.ckpt --use_16bit --gpus 1 --evaluate --arch ir_50
```

The results of the test will be printed to stdout and saved to a new file in the `/experiments/` directory.

### Fine-tuning

To fine-tune the pre-trained model on style-transferred data, prepare your training dataset in the `.rec` format, alongside the
`.lst` and `.idx` files and place it into `/data/faces_webface_112x112/train/`. The files must be named `train.rec`, `train.idx` and `train.lst`.
Then place your validation data set into the `/data/faces_webface_112x112/val/` directory in the `.bin` format. It has to be named `lfw.bin`.
After that, it must be preprocessed using the following command:
```
python3.8 convert.py --rec_path ./data/faces_webface_112x112/val --make_validation_memfiles
```

Once all that is done, place your fine-tuned model `adaface_ir50_ms1mv2.ckpt` into the `/pretrained/` directory and run the
command 
```
./scripts/run_ir50_ms1mv2.sh
```

The training will start, and you will have the option to log some training information via _wandb_.