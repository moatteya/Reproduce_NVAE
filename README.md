# Reproduce_NVAE
## Requirements
NVAE is built in Python 3.7 using PyTorch 1.6.0. The following packages are also required:
pillow 8.2, matplotlib 3.3.4, tensorboard 2.4.1 ,tensorboardX 2.2, lmdb 1.1.1, tfrecord 1.11

## Training
Two 11-GB RTX 2080Ti GPUs are used for training NVAE on dynamically binarized MNIST. Training takes about 14 hours.
```shell script
cd CODE_DIR
python train.py --data DATA_DIR/mnist --root CHECKPOINT_DIR --save EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 2 --use_se --res_dist --fast_adamax 
```
## Evaluation 
You can use the following command to load a trained model and evaluate it on the test datasets:
```shell script
cd $CODE_DIR
python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR/mnist --eval_mode=evaluate --num_iw_samples=1000
```
You can also use the following command to generate samples from a trained model:

```shell script
cd $CODE_DIR
python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=sample --temp=0.6 --readjust_bn
```
Checkpoint for the MNIST model can be found in 
[this Google drive directory](https://drive.google.com/drive/folders/1nMiUFNojIBf-vQafl_7-CVBHJX7JbUOj?usp=sharing) 

## Understanding the implementation
If you are modifying the code, you can use the following figure to map the code to the paper.

<p align="center">
    <img src="img/model_diagram.png" width="900">
</p>
