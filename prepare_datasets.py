from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

data_dir = './data'
train_file = f"{data_dir}/train.csv"
val_file = f"{data_dir}/val.csv"
test_file = f"{data_dir}/test.csv"

# load the datasets
raw_dataset = load_dataset("csv",
                           data_files={'train': train_file,
                                       'val': val_file,
                                       'test': test_file,
                                       },
                           )

# tokenize the datasets
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# preprocess function, tokenizes text
def preprocess_function(examples):
    to_ret = tokenizer(examples["sentence"],
                       padding='max_length',
                       truncation=True,
                       )
    return to_ret


# preprocess dataset
dataset = raw_dataset.map(
    preprocess_function,
    batched=False,
)

# need to rename 'label' to the expected 'labels' for hugingface
# dataset.rename_column('label', 'labels')
label_names = sorted(set(dataset['train']['label']))
class_label = ClassLabel(names_file='label_names.csv')

# save them to disk
dataset.save_to_disk(f'{data_dir}/dataset')
print('done')
