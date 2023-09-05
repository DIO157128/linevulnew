import os
if __name__ =="__main__":
    #func
    os.system("python codebert_main.py \
  --output_dir=../results/saved_models \
  --model_type=roberta \
  --do_train \
  --num_labels 10\
  --do_test \
  --train_data_file=../../data/big-vul_dataset/cwe_train.csv \
  --eval_data_file=../../data/big-vul_dataset/cwe_val.csv \
  --test_data_file=../../data/big-vul_dataset/cwe_test.csv \
  --epochs 60 \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --model_name codebert.bin \
  --n_gpu 1\
  --seed 123456  2>&1 | tee train.log")

