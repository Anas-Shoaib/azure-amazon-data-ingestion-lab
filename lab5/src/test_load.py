from load_data import load_cmapss, preprocess
train, test = load_cmapss('../data/train_FD001.txt','../data/test_FD001.txt','../data/RUL_FD001.txt')
train = preprocess(train)
print('Train shape:', train.shape)
print('RUL range:', train['RUL'].min(), '–', train['RUL'].max())