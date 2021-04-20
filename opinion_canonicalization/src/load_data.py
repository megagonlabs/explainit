import csv
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import word_tokenize as tokenizer
import numpy as np
import random
import torch
import sys

random.seed(4)

class W2VLoader:
	def __init__(self, model_name):
		self.word_vectors = api.load(model_name)
		self.w_emb = self.word_vectors.vectors
		self.unknown_idx = len(self.w_emb)
		self.pad_idx = len(self.w_emb) + 1
		empty_emb = np.zeros((1, self.word_vectors.vectors.shape[1]))
		self.w_emb = np.append(self.w_emb, empty_emb, axis=0)
		self.w_emb = np.append(self.w_emb, empty_emb, axis=0)

	def toks2ids(self, toks, max_seq_length):
		ids = []
		for i in range(max_seq_length):
			is_pad = i >= len(toks)
			tok = toks[i] if i < len(toks) else None
			ids.append(self.tok2id(tok, is_pad))
		return ids

	def tok2id(self, tok, is_pad):
		if is_pad:
			return self.pad_idx
		if tok not in self.word_vectors.vocab:
			return self.unknown_idx
		return self.word_vectors.vocab[tok].index

	def tok2emb(self, tok):
		return self.w_emb[self.tok2id(tok, False)]

	def get_emb_vec(self, row):
		emb = []
		for field in row:
			for tok in tokenizer(field):
				emb.append(self.tok2emb(tok))
		if len(emb) == 0:
			emb = [self.w_emb[self.unknown_idx]]
		return torch.norm(torch.tensor(emb), dim=0)

	def get_emb_matrix(self, rows):
		embs = []
		for row in rows:
			embs.append(self.get_emb_vec(row))
		return torch.stack(embs)

def DataLoader(target_rows, aux_rows, batch_size, w2cloader, max_seq_length=5):
	assert len(target_rows) == len(aux_rows)
	aux_cols = [list(set([r[i] for r in aux_rows])) for i in range(len(aux_rows[0]))]
	aux_c_sizes = [len(c) for c in aux_cols]
	aux_c_counts = [[sum(1 for row in aux_rows if row[i]==val) for val in aux_cols[i]] for i in range(len(aux_cols))]
	aux_weights = [[min(3, val/(min(col)+1.)) for val in col] for col in aux_c_counts]

	# Update batch
	target_ids = [[w2cloader.toks2ids(tokenizer(col), max_seq_length) for col in row] for row in target_rows]
	label_ids = [[aux_cols[i].index(row[i]) for i in range(len(row))] for row in aux_rows]	
	dataset = torch.utils.data.TensorDataset(torch.tensor(target_ids), torch.tensor(label_ids))
	data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

	return data_iter, aux_c_sizes, aux_weights

def PairLoader(pairs, batch_size, w2cloader, max_seq_length=3):
	left = [pair[0] for pair in pairs]
	right = [pair[1] for pair in pairs]
	left_ids = [[w2cloader.toks2ids(tokenizer(col), max_seq_length) for col in row] for row in left]
	right_ids = [[w2cloader.toks2ids(tokenizer(col), max_seq_length) for col in row] for row in right]
	dataset = torch.utils.data.TensorDataset(torch.tensor(left_ids), torch.tensor(right_ids))
	data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

	return data_iter
			
def LabelLoader(labels, batch_size):
	return torch.utils.data.DataLoader(torch.tensor(labels), batch_size=batch_size)

def FileLoader(file_name, delimiter=",", quotechar=None, do_shuffle=True):
	header = None
	rows = []
	num_col = None
	with open(file_name, "r", encoding="utf-8") as f:
		reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
		for line in reader:
			if len(line) < 1:
				continue
			if num_col and len(line)!=num_col:
				continue
			if sys.version_info[0] == 2:
				line = list(unicode(cell, 'utf-8') for cell in line)
			if header == None:
				header = {line[i]:i for i in range(len(line))}
				num_col = len(line)
			else:
				rows.append(line)
	if do_shuffle:
		permutation = [i for i in range(len(rows))]
		random.shuffle(permutation)
		shuffled_row = [rows[i] for i in permutation]
		return header, shuffled_row
	else:
		return header, rows

def FileWriter(file_name, header, rows, labels):
	attr_order = [None for i in range(len(header))]
	for key, idx in header.items():
		attr_order[idx] = key
	with open(file_name, "w") as file:
		file.write(",".join(attr_order+["result"])+"\n")
		for i in range(len(rows)):
			values = [str(rows[i][header[attr]]) for attr in attr_order]
			values.append(str(labels[i]))
			file.write(",".join(values) + "\n")
		file.close()
