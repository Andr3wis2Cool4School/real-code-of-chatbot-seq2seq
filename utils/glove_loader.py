import numpy as np 
GLOVE_EMBEDDING_SIZE = 300


def load_cur_glove(embedding_path):
    _word2em = {}
    with open(embedding_path, mode='rt', encoding='utf8') as f:
        for line in f:
            words = line.strip().split()
            word = words[0]
            embeds = np.array(words[1:], dtype=np.float32)
            _word2em[word] = embeds
    return _word2em