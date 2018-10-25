import codecs
import os
import zipfile

import numpy as np
from six.moves import urllib

def download_and_unzip(url_base, zip_name, *file_names, data_dir):
    
    print('saving files to %s' % data_dir)
    zip_path = os.path.join(data_dir, zip_name)
    url = url_base + zip_name
    print('downloading %s to %s' % (url, zip_path))
    urllib.request.urlretrieve(url, zip_path)
    out_paths = []
    with zipfile.ZipFile(zip_path, 'r') as f:
        for file_name in file_names:
            print('extracting %s' % file_name)
            out_paths.append(f.extract(file_name, path=data_dir))
    return out_paths


def filter_glove(full_glove_path, filtered_glove_path):
    vocab = set()
    # Download the full set of unlabeled sentences separated by '|'.
    sentence_path = ['data/sick_train_sentenceA_tree.txt', 'data/sick_train_sentenceB_tree.txt',
                      'data/sick_trial_sentenceA_tree.txt', 'data/sick_trial_sentenceB_tree.txt',
                    'data/sick_test_sentenceA.txt', 'data/sick_test_sentenceB.txt']
    for path in sentence_path:
        with open(path, 'r') as f:
            for line in f:
                # Drop the trailing newline and strip backslashes. Split into words.
                vocab.update(line.strip().split())
    nread = 0
    nwrote = 0
    with codecs.open(full_glove_path, encoding='utf-8') as f:
        with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
            for line in f:
                nread += 1
                line = line.strip()
                if not line: continue
                if line.split(u' ', 1)[0] in vocab:
                    out.write(line + '\n')
                    nwrote += 1
    print('read %s lines, wrote %s' % (nread, nwrote))
    return vocab

def load_embeddings(embedding_path):
    """Loads embedings, returns weight matrix and dict from words to indices."""
    print('loading word embeddings from %s' % embedding_path)
    weight_vectors = []
    word_idx = {}
    with codecs.open(embedding_path, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            word_idx[word] = len(weight_vectors)
            weight_vectors.append(np.array(vec.split(), dtype=np.float32))
    # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
    # '-RRB-' respectively in the parse-trees.
    if u'(' in word_idx:
        word_idx[u'-LRB-'] = word_idx.pop(u'(')
    if u')' in word_idx:
        word_idx[u'-RRB-'] = word_idx.pop(u')')
        
    # Random embedding vector for UNKNOWN-WORD.
    point_zero_five = np.zeros(weight_vectors[0].shape, dtype=np.float32)+0.05
    weight_vectors.append(point_zero_five)
    word_idx["UNKNOWN_WORD"] = len(weight_vectors)-1
    
    # Random embedding vector for left_marker.
    five_point_five = np.zeros(weight_vectors[0].shape, dtype=np.float32)+5.5
    weight_vectors.append(five_point_five)
    word_idx["LEFT_MARKER"] = len(weight_vectors)-1
    
    # Random embedding vector for right_marker.
    weight_vectors.append(five_point_five + 0.25)
    word_idx["RIGHT_MARKER"] = len(weight_vectors)-1
    
    # Random embedding vector for right_marker.
    weight_vectors.append(np.zeros(weight_vectors[0].shape, dtype=np.float32))
    word_idx["END_MARKER"] = len(weight_vectors)-1
    
    return np.stack(weight_vectors), word_idx