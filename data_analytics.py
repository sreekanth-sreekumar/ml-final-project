import pandas as pd

train_df = pd.read_csv('./data/multimodal_train.tsv',delimiter='\t')
print(len(train_df))

test_df = pd.read_csv('./data/multimodal_test_public.tsv', delimiter='\t')
print(len(test_df))

val_df = pd.read_csv('./data/multimodal_validate.tsv', delimiter='\t')
print(len(val_df))