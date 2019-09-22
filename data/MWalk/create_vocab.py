import json
import csv
import argparse
import os
import sys



# parser = argparse.ArgumentParser()
# parser.add_argument("--data_input_dir", default = '')
# parser.add_argument("--vocab_dir", default="")
# parsed = vars(parser.parse_args())
#
# dir = parsed['data_input_dir']
# vocab_dir = parsed['vocab_dir']


entity_vocab = {}
relation_vocab = {}
entity_counter = 0
relation_counter = 0
for f in [sys.argv[1]]:  #'graph.txt', 'train.txt', 'dev.txt', 'test.txt', 
    with open(f) as raw_file:
        csv_file = csv.reader(raw_file, delimiter='\t')
        for line in csv_file:
            e1,r,e2 = line
            if e1 not in entity_vocab:
                entity_vocab[e1] = entity_counter
                entity_counter += 1
            if e2 not in entity_vocab:
                entity_vocab[e2] = entity_counter
                entity_counter += 1
            if r not in relation_vocab:
                relation_vocab[r] = relation_counter
                relation_counter += 1

with open('entity_vocab.json', 'w') as fout:
    json.dump(entity_vocab, fout)

with open('relation_vocab.json', 'w') as fout:
    json.dump(relation_vocab, fout)


