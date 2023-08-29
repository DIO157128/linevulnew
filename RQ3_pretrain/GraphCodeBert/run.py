import os
if __name__ =="__main__":
    #func
    os.system("python graphcodebert_main.py \
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
  --evaluate_during_training \
  --model_name graphcodebert.bin \
  --n_gpu 2\
  --seed 123456  2>&1 | tee train.log")
    #line
    os.system("python graphcodebert_main.py \
  --do_sorting_by_line_scores \
  --effort_at_top_k=0.2 \
  --top_k_recall_by_lines=0.01 \
  --top_k_recall_by_pred_prob=0.2 \
  --model_name=graphcodebert.bin \
  --output_dir=../results/saved_models \
  --model_type=roberta \
  --do_test \
  --do_local_explanation \
  --top_k_constant=10 \
  --reasoning_method=attention \
  --train_data_file=../../data/big-vul_dataset/train.csv \
  --eval_data_file=../../data/big-vul_dataset/val.csv \
  --test_data_file=../../data/big-vul_dataset/test.csv \
  --block_size 512 \
  --n_gpu 2\
  --eval_batch_size 512")

