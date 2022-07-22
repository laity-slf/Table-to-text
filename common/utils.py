import os


def create_examples(input_file, unk_token="[UNK]", example_type='train'):
    src = example_type.lower() + '_src.txt'
    tgt = example_type.lower() + '_label.txt'
    with open(os.path.join(input_file, src), 'r') as f:
        lines = f.readlines()
        x = [line.strip('\n').split('<ent>|')[1:] for line in lines]
        x = [xx[:-30] for xx in x]
    with open(os.path.join(input_file, tgt), 'r') as f:
        lines = f.readlines()
        y = [line.strip('\n').split(' ') for line in lines]
        y = [yy[:-30] for yy in y]
    return x, y


def convert_examples_to_features(examples, tokenizer, max_length):
    # examples_tokenized = [[tokenizer(r) for r in tab] for tab in examples[0]]
    print(1)
    pass
