import os
if __name__ =="__main__":
    #func
    os.system("python codet5_main.py \
  --output_dir=../results/saved_models \
  --model_type=roberta \
  --use_word_level_tokenizer \
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
  --evaluate_during_training \
  --model_name codet5.bin \
  --n_gpu 1\
  --seed 123456  2>&1 | tee train.log")

