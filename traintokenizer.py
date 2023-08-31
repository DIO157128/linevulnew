import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer

# df = pd.read_csv("./data/big-vul_dataset/train.csv")
# all_funcs = df["processed_func"].tolist()
# for func in tqdm(all_funcs):
#     func = func.split("\n")
#     func = [line for line in func if len(line) > 0]
#     for line in func:
#         with open("./data/tokenizer_train_data.txt", "a",encoding='utf-8') as f:
#             f.write(line + "\n")
# print("Done")

def train(model_name,vocab_size):


    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(vocab_size=vocab_size,
                               special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                               min_frequency=2)

    tokenizer.train(files=['../data/tokenizer_train_data.txt'], trainer=trainer)

    # Save the files
    tokenizer.save("./{}.json".format(model_name))

    print('done')


    print('Training and saving completed.')
if __name__ == '__main__':
    # models = ['microsoft/codebert-base','Salesforce/codet5-base','microsoft/graphcodebert-base','microsoft/unixcoder-base','uclanlp/plbart-base','gpt2']
    # for model_name in models:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #
    #     vocab_size = tokenizer.vocab_size
    #     print("Vocabulary size:", vocab_size)
    # train('codebert',50265)
    # train('codet5',32100)
    # train('graphcodebert',50265)
    # train('unixcoder',51416)
    # train('plbart',50005)
    # train('gpt2',50257)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_tokens(["<pad>"])
    tokenizer.pad_token = "<pad>"
    print(tokenizer.bos_token)
    print(tokenizer.eos_token)
    print(tokenizer.pad_token_id)