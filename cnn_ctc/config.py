import os

num_chars = 37
base_dir = os.getcwd()
trainset = r'trainset'
valset='valset'
batch_size = 20
model_dir = os.path.join(base_dir,'mdl','mdl')
toks = ['0','1','2','3','4','5','6','7','8','9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '_']