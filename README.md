## How to run the Code
Open the VisionAndLanguageProjectFileToRun.pynb file and Press Run all. It will collect the required datasets and then train and test. If you do not want to train then click every cell in the pynb file except the cell in which train command is written.

Checkpoints are present at https://drive.google.com/drive/folders/1-ME8FdUiq9fn6Ni6kEoTAJKZSqITShhw?usp=sharing
If the above link is not working then use the phi_000013000.pt checkpoint uploaded on this GitHub repository.

## 📚 How to Train LinCIR
Train LinCIR with ease using the following command:

```bash
$ python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --node_rank 0 \
--master_addr localhost --master_port 5100 train_phi.py \
--batch_size 64 \
--output_dir /path/to/your_experiment \
--cirr_dataset_path /path/to/cir_datasets/CIRR \
--mixed_precision fp16 \
--clip_model_name large \
--validation_steps 1000 \
--checkpointing_steps 1000 \
--seed 12345 \
--lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
--resume <path to latest checkpoint> \
--max_train_steps 20000
```

If you have a powerful GPU machine with 8 GPUs, simply run the above script. For less powerful GPU machine with single GPU, set `--nuproc_per_node` to 1 and adjust `--batch_size` to 256 or 512. Rest assured, the results will be consistent.

If you'd like to use ViT-Large, Huge or Giga as CLIP backbone, change `--clip_model_name` to large, huge, or giga each. You will require GPU RAM(16Gb). Otherwise drop the batchsize to 64 or 128.

## 💯 How to Evaluate LinCIR
### FashionIQ
To evaluate LinCIR on FashionIQ, run the following command:

```bash
$ python validate.py \
--eval-type phi \
--dataset fashioniq \
--dataset-path /path/to/fashioniq \
--phi-checkpoint-name /path/to/trained_your/phi_best.pt \
--clip_model_name large
```
All Credits of the Original work goes to https://github.com/navervision/lincir
