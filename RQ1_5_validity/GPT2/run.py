import os
if __name__ =="__main__":
    #func
    os.system("python gpt_main.py \
  --output_dir=../results/saved_models \
  --model_type=roberta \
  --do_train \
  --do_test \
  --train_data_file=../../data/big-vul_dataset/train.csv \
  --eval_data_file=../../data/big-vul_dataset/val.csv \
  --test_data_file=../../data/big-vul_dataset/test.csv \
  --epochs 10 \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --do_local_explanation \
  --top_k_constant=10 \
  --reasoning_method=attention \
  --evaluate_during_training \
  --model_name gpt.bin \
  --n_gpu 1\
  --seed 123456  2>&1 | tee train.log")

