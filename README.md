# Retinaface Face Detection 

A retinaface model for Face Detection trained on widerface dataset.

Notice: This is face detection model's training, evaluation and inference scripts in HuggingFace🤗 style from scratch for practice.

## Train
Run
```bash
python train.py --model_config_file <MODEL_CONFIG_FILE>
```
<MODEL_CONFIG_FILE> can be found in folder `config`.
Model checkpoints will be saved in folder `checkpoints` by default.

## Inference
### Observe logits map and predicted bboxes
Run
```bash
python inference.py --checkpoint_path <CHECKPOINT_PATH>
```
<CHECKPOINT_PATH> is a model folder containing `config.json` and `pytorch_model.bin`.

![inference image](pic/inference.svg)

### Detect
Run
```bash
python detect.py --checkpoint_path <CHECKPOINT_PATH> --image_path <IMAGE_PATH> --save_path <SAVE_PATH>
```
![inference image](pic/detect_result.png)

## References
- [Retinface-pytorch](https://github.com/biubug6/Pytorch_Retinaface)