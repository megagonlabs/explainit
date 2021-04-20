"""
This implements a multi-task model for two classification tasks:
1. classify sentence into two classes: (a) contain causal rel; (b) otherwise.
2. classify two opinions in the sentence into two classes: 
   (a) op1 explains op2; (b) otherwise.

Structure:
The two classification tasks shares the embedding and lstm layer; 
the second task further combines the context layer from the first task for the
final labels.

Objective:
The loss is the (weighted) sum of losses from both tasks.

Use BERT tokenizer and acquire embeddings from BERT seq output.
"""
import argparse
import csv
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import sys

from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.modeling import BertModel, BertConfig, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

logger = logging.getLogger(__name__)

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, sent, op1, op2, sent_label=None, op_label=None):
		"""Constructs a InputExample.
		Args:
		    guid: Unique id for the example.
		    text_a: string. The untokenized text of the first sequence. For single
		    sequence tasks, only this sequence must be specified.
		    text_b: (Optional) string. The untokenized text of the second sequence.
		    Only must be specified for sequence pair tasks.
		    label: (Optional) string. The label of the example. This should be
		    specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.sent = sent
		self.op1 = op1
		self.op2 = op2
		self.sent_label = sent_label
		self.op_label = op_label


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, guid, input_ids, input_mask, op1_mask, op2_mask, segment_ids, 
				 sent_label_id, op_label_id):
		self.guid = guid
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.op1_mask = op1_mask
		self.op2_mask = op2_mask
		self.segment_ids = segment_ids
		self.sent_label_id = sent_label_id
		self.op_label_id = op_label_id

class DataProcessor(object):
	"""Processor for the data set"""

	def get_examples(self, data_dir, mode):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(data_dir), mode)

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			op_label = line[0]
			sent_label = line[1]
			sent = line[2]
			op1 = line[3]
			op2 = line[4]
			guid = int(line[5])
			examples.append(
			    InputExample(guid=guid, sent=sent, op1=op1, op2=op2, 
			    			 sent_label=sent_label, op_label=op_label))
		return examples

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r", encoding="utf-8") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				if sys.version_info[0] == 2:
					line = list(unicode(cell, 'utf-8') for cell in line)
				lines.append(line)
			return lines

def convert_examples_to_features(examples, label_list, max_seq_length, 
								 tokenizer):
	def _mask_id(input_ids, mask_ids):
		mask = [0 for i in range(len(input_ids))]
		for i in range(len(input_ids)):
			if input_ids[i] in mask_ids:
				mask[i] = 1
		return mask

	label_map = {label : i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in enumerate(examples):
		if ex_index % 100 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		sent = tokenizer.tokenize(example.sent)
		op1 = tokenizer.tokenize(example.op1)
		op2 = tokenizer.tokenize(example.op2)

		# skip example if the sentence length is above the max_seq_length
		if len(sent) > max_seq_length-2:
			continue

		tokens = ["[CLS]"] + sent + ["[SEP]"]
		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		op1_ids = tokenizer.convert_tokens_to_ids(op1)
		op2_ids = tokenizer.convert_tokens_to_ids(op2)

		op1_mask = _mask_id(input_ids, op1_ids)
		op2_mask = _mask_id(input_ids, op2_ids)

		input_mask = [1] * len(input_ids)

		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		op1_mask += padding
		op2_mask += padding
		input_mask += padding
		segment_ids = [0] * (max_seq_length)

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		assert len(op1_mask) == max_seq_length
		assert len(op2_mask) == max_seq_length

		sent_label_id = label_map[example.sent_label]
		op_label_id = label_map[example.op_label]

		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			logger.info("sentence: %s" % " ".join(
			        [str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
			        "opinion 1: %s" % " ".join([str(x) for x in op1_mask]))
			logger.info(
			        "opinion 2: %s" % " ".join([str(x) for x in op2_mask]))
			logger.info("sent label: %s; op label: %s" % (example.sent_label,
														  example.op_label))
		features.append(
                InputFeatures(guid=example.guid,
                			  input_ids=input_ids,
                              input_mask=input_mask,
                              op1_mask=op1_mask,
                              op2_mask=op2_mask,
                              segment_ids=segment_ids,
                              sent_label_id=sent_label_id,
                              op_label_id=op_label_id))
	return features

class Attention_Layer(nn.Module):
	'''
	A general attention that produces the self attention of a given matrix
	'''
	def __init__(self, hidden_dim):
		super(Attention_Layer, self).__init__()
		self.W = nn.Parameter(torch.rand(hidden_dim, hidden_dim))
		self.b = nn.Parameter(torch.rand(hidden_dim))
		self.w = nn.Parameter(torch.rand(hidden_dim))
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.to(device)

	def forward(self, Y, mask_Y):
		'''
		Input: 
			Y: batch_dim, num_seq, hidden_dim
			mask_Y: batch_dim, num_seq
		Variables:
			wY: batch_dim, num_seq, hidden_dim
			wYb: batch_dim, num_seq, hidden_dim
			u: batch_dim, num_seq, hidden_dim
			uw: batch_dim, num_seq, 1
			alpha: batch_dim, num_seq
		Output:
			c_star: batch_dim, hidden_dim
		'''
		batch_dim, num_seq, hidden_dim = Y.size()
		wY = torch.bmm(Y, self.W.unsqueeze(0).expand(batch_dim, *self.W.size()))
		wYb = wY + self.b.unsqueeze(0).unsqueeze(0).expand(batch_dim, num_seq, hidden_dim)
		u = torch.tanh(wYb)
		uw = torch.bmm(u, self.w.unsqueeze(0).expand(batch_dim, hidden_dim).unsqueeze(2))
		alpha = F.softmax(uw.squeeze(2) + (-1000.0 * (1. - mask_Y)), 1)
		c_star = torch.bmm(alpha.unsqueeze(1), Y).squeeze(1)

		return c_star
  
class Alignment_Attention_Layer(nn.Module):
	def __init__(self, hidden_size):
		super(Alignment_Attention_Layer, self).__init__()
		self.hidden_size = hidden_size
		self.W_y = nn.Parameter(torch.rand(hidden_size, hidden_size))
		self.W_h = nn.Parameter(torch.rand(hidden_size, hidden_size))
		self.W_r = nn.Parameter(torch.rand(hidden_size, hidden_size))
		self.w = nn.Parameter(torch.rand(hidden_size))
		_W_t = np.random.randn(hidden_size, hidden_size)
		_W_t_ortho, _ = np.linalg.qr(_W_t)
		self.W_t = nn.Parameter(torch.Tensor(_W_t_ortho))
		self.W_p = nn.Parameter(torch.rand(hidden_size, hidden_size))
		self.W_x = nn.Parameter(torch.rand(hidden_size, hidden_size))
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.to(self.device)

	def _forward_one_word(self, Y, mask_Y, h_t, r_tm1):
		'''
		Input:
			Y: batch_dim, num_seq, hidden_size
			mask_Y: batch_dim, num_seq
			h_t: batch_dim, hidden_size
			r_tm1: batch_dim, hidden_size
		Variables:
			Wy: batch_dim, num_seq, hidden_size
			Wh: batch_dim, hidden_size
			M: batch_dim, num_seq, hidden_size
			alpha: batch_dim, num_seq
		Output:
			r_t: batch_dim, hidden_size
		'''
		batch_size, num_seq, hidden_size = Y.size()
		Wy = torch.bmm(Y, self.W_y.unsqueeze(0).expand(batch_size, hidden_size, hidden_size))
		Wh = torch.mm(h_t, self.W_h)
		if r_tm1 is not None:
			Wh += torch.mm(r_tm1, self.W_r)
		M = torch.tanh(Wy+Wh.unsqueeze(1).expand(batch_size, num_seq, hidden_size))
		wM = torch.bmm(M, self.w.unsqueeze(0).expand(batch_size, hidden_size).unsqueeze(2))
		alpha = F.softmax(wM.squeeze(2) + (-1000.0 * (1. - mask_Y)), 1)
		r_t = torch.bmm(alpha.unsqueeze(1), Y).squeeze(1)
		if r_tm1 is not None:
			r_t += torch.tanh(torch.mm(r_tm1, self.W_t))
		return r_t

	def _forward_weights_by_mask(self, r_new, r_old, mask_t):
		'''
		return r_new when mask_t = 1, otherwise, return r_old
		'''
		return (r_new * mask_t) + (r_old * (1.0 - mask_t))

	def forward(self, Y, mask_Y, hidden_states, mask_H):
		'''
		Input:
			Y: batch_dim, num_seq, hidden_size
			mask_Y: batch_dim, num_seq
			hidden_states: batch_dim, num_seq, hidden_size
			mask_H: batch_dim, num_seq 
		Variables:
			Y: batch_dim, num_seq, hidden_size
			r_tm1: batch_dim, hidden_size
		Output:
			h_star: batch_dim, hidden_size
		'''
		hidden_states = hidden_states.transpose(1, 0)
		mask_H = mask_H.transpose(1,0)
		
		# Initialize r_tm1 as all zeros
		r_tm1 = Variable(torch.zeros(Y.size()[0], Y.size()[2]))
		r_tm1 = r_tm1.to(self.device)
		
		# Iterate all words in the sequence
		for i in range(hidden_states.size()[0]):
			h_t = hidden_states[i] # h_t.size() = batch_size, hidden_size
			mask_t = mask_H[i] # mask_t.size() = batch_size
			r_t = self._forward_one_word(Y, mask_Y, hidden_states[i], r_tm1)
			r_t = self._forward_weights_by_mask(r_t, r_tm1, mask_t.unsqueeze(1))
			r_tm1 = r_t

		# Combine r_t and h_H
		Wp = torch.mm(r_tm1, self.W_p)
		Wx = torch.mm(hidden_states[-1], self.W_x)
		h_star = torch.tanh(Wp + Wx)		

		return h_star

class Lstm_Trio_Attn_BERT(nn.Module):
	"""Add layers on top of BERT."""
	def __init__(self, args, output_size):
		super(Lstm_Trio_Attn_BERT, self).__init__()
		self.args = args
		self.output_size = output_size
		self.bert = BertModel.from_pretrained(args.bert_model)
		self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
		
		self.sent_attn = Attention_Layer(self.bert.config.hidden_size)
		self.op1_to_op2_attn = Alignment_Attention_Layer(self.bert.config.hidden_size)
		self.op2_to_op1_attn = Alignment_Attention_Layer(self.bert.config.hidden_size)
		
		self.sent_classifier = nn.Linear(self.bert.config.hidden_size, self.output_size)
		self.op_classifier = nn.Linear(3*self.bert.config.hidden_size, self.output_size)

	def forward(self, input_ids, input_masks, op1_masks, op2_masks):
		encoded_layer, _ = self.bert(input_ids, None, input_masks, 
									 output_all_encoded_layers=False)
		encoded_layer = self.dropout(encoded_layer)

		sent_attn = self.sent_attn(encoded_layer, input_masks.to(dtype=torch.float32))
		op1_to_op2_attn = self.op1_to_op2_attn(encoded_layer, op1_masks.to(dtype=torch.float32), 
											   encoded_layer, op2_masks.to(dtype=torch.float32))
		op2_to_op1_attn = self.op2_to_op1_attn(encoded_layer, op2_masks.to(dtype=torch.float32),
											   encoded_layer, op1_masks.to(dtype=torch.float32))
		sent_logits = self.sent_classifier(sent_attn)
		op_logits = self.op_classifier(torch.cat((sent_attn, op1_to_op2_attn, 
												  op2_to_op1_attn), dim=1))

		return sent_logits, op_logits

	def fit(self):
		processor = DataProcessor()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		n_gpu = torch.cuda.device_count()
		self.to(device)

		tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, 
											do_lower_case=self.args.do_lower_case)
		train_examples = processor.get_examples(os.path.join(self.args.data_dir, "train"), "train")
		num_train_opt_steps = int(
			len(train_examples) / self.args.train_batch_size / self.args.gradient_accumulation_steps) * args.num_train_epochs
		
		train_features = convert_examples_to_features(
			train_examples, processor.get_labels(), 
			self.args.max_seq_length, tokenizer)

		param_optimizer = list(self.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
			]
		optimizer = BertAdam(optimizer_grouped_parameters,
							 lr=self.args.learning_rate,
							 warmup=self.args.warmup_proportion,
							 t_total=num_train_opt_steps)

		all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
		all_op1_mask = torch.tensor([f.op1_mask for f in train_features], dtype=torch.long)
		all_op2_mask = torch.tensor([f.op2_mask for f in train_features], dtype=torch.long)
		all_sent_label_id = torch.tensor([f.sent_label_id for f in train_features], dtype=torch.long)
		all_op_label_id = torch.tensor([f.op_label_id for f in train_features], dtype=torch.long)

		train_data = TensorDataset(all_input_ids, all_input_mask, 
								   all_segment_ids, all_op1_mask, all_op2_mask,
								   all_sent_label_id, all_op_label_id)
		train_sampler = RandomSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, 
									  batch_size=self.args.train_batch_size)
		self.train()
		for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):
			total_epoch_loss, total_epoch_corrects, total_examples, steps = 0, 0, 0, 0
			for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, op1_mask, op2_mask, sent_label_id, op_label_id = batch

				# define a new function to compute loss values for both output_modes
				#print(input_ids.size(), op1_mask.size())
				sent_logits, op_logits = self.__call__(input_ids, input_mask, op1_mask, op2_mask)

				loss_fct = CrossEntropyLoss()
				sent_loss = loss_fct(sent_logits.view(-1, self.output_size), sent_label_id.view(-1))
				op_loss = loss_fct(op_logits.view(-1, self.output_size), op_label_id.view(-1))

				loss = sent_loss + op_loss

				if self.args.gradient_accumulation_steps > 1:
					loss = loss / self.args.gradient_accumulation_steps
				loss.backward()
				
				if (step + 1) % self.args.gradient_accumulation_steps == 0:
					optimizer.step()
					optimizer.zero_grad()

				total_epoch_loss += loss.item()
				total_epoch_corrects += (torch.max(op_logits, 1)[1].view(op_label_id.size()).data == op_label_id.data).float().sum().item()		
				total_examples += input_ids.size(0)
				steps += 1

			total_epoch_loss = total_epoch_loss/steps
			total_epoch_acc = 100.0 * total_epoch_corrects/total_examples
			logger.info("*** Epoch loss:%f; Epoch acc: %f***" % (total_epoch_loss, total_epoch_acc))

	def evaluate(self, args, mode='dev'):
		processor = DataProcessor()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		n_gpu = torch.cuda.device_count()
		self.to(device)

		tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, 
											do_lower_case=self.args.do_lower_case)
		eval_examples = processor.get_examples(os.path.join(args.data_dir, mode), mode)
		eval_features = convert_examples_to_features(
			eval_examples, processor.get_labels(), 
			self.args.max_seq_length, tokenizer)

		all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
		all_op1_mask = torch.tensor([f.op1_mask for f in eval_features], dtype=torch.long)
		all_op2_mask = torch.tensor([f.op2_mask for f in eval_features], dtype=torch.long)
		all_sent_label_id = torch.tensor([f.sent_label_id for f in eval_features], dtype=torch.long)
		all_op_label_id = torch.tensor([f.op_label_id for f in eval_features], dtype=torch.long)

		eval_data = TensorDataset(all_input_ids, all_input_mask, 
								   all_segment_ids, all_op1_mask, all_op2_mask,
								   all_sent_label_id, all_op_label_id)
		eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
		
		self.eval()
		total_eval_loss, total_corrects, total_examples, steps = 0, 0, 0, 0
		all_preds = []
		for input_ids, input_mask, segment_ids, op1_mask, op2_mask, sent_label_id, op_label_id in tqdm(eval_dataloader, desc="Evaluating"):
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			segment_ids = segment_ids.to(device)
			op1_mask = op1_mask.to(device)
			op2_mask = op2_mask.to(device)
			sent_label_id = sent_label_id.to(device)
			op_label_id = op_label_id.to(device)

			with torch.no_grad():
				sent_logits, op_logits = self.__call__(input_ids, input_mask, 
            										   op1_mask, op2_mask)

			loss_fct = CrossEntropyLoss()
			sent_loss = loss_fct(sent_logits.view(-1, self.output_size), sent_label_id.view(-1))
			op_loss = loss_fct(op_logits.view(-1, self.output_size), op_label_id.view(-1))


			op_predicts = torch.max(op_logits, 1)[1]
			all_preds += (op_predicts).cpu().numpy().tolist()

			total_eval_loss += op_loss.mean().item()
			total_corrects += (op_predicts.view(op_label_id.size()).data == op_label_id.data).float().sum().item()
			total_examples += input_ids.size(0)
			steps += 1
		result = {}
		result['total_loss'] = total_eval_loss/steps
		result['total_examples'] = total_corrects/total_examples
		result['predicts'] = all_preds

		print("*** %s loss:%f; %s acc: %f***" % (mode, result['total_loss'], mode, result['total_examples']))

		return result

	def predict_many(self, instances):
		processor = DataProcessor()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		n_gpu = torch.cuda.device_count()
		self.to(device)

		tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, 
											do_lower_case=self.args.do_lower_case)
		predict_example = []
		predict_indices = []
		for ext_i, ext_j, ext1, ext2, context in instances:
			predict_example.append(InputExample(len(predict_example), context, ext1, ext2, 
				sent_label="0", op_label="0"))
		predict_features = convert_examples_to_features(
			predict_example, processor.get_labels(), 
			self.args.max_seq_length, tokenizer)

		cur = 0
		runs = int((len(predict_features)+self.args.eval_batch_size-1)/self.args.eval_batch_size)
		pbar = tqdm(total = runs)
		all_probs = []

		while cur < len(predict_features):
			upper_bound = min(len(predict_features), cur+self.args.eval_batch_size)
			input_ids = torch.tensor([f.input_ids for f in predict_features[cur:upper_bound]], dtype=torch.long).to(device)
			input_mask = torch.tensor([f.input_mask for f in predict_features[cur:upper_bound]], dtype=torch.long).to(device)
			segment_ids = torch.tensor([f.segment_ids for f in predict_features[cur:upper_bound]], dtype=torch.long).to(device)
			op1_mask = torch.tensor([f.op1_mask for f in predict_features[cur:upper_bound]], dtype=torch.long).to(device)
			op2_mask = torch.tensor([f.op2_mask for f in predict_features[cur:upper_bound]], dtype=torch.long).to(device)
			sent_label_id = torch.tensor([f.sent_label_id for f in predict_features[cur:upper_bound]], dtype=torch.long).to(device)
			op_label_id = torch.tensor([f.op_label_id for f in predict_features[cur:upper_bound]], dtype=torch.long).to(device)
			
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			segment_ids = segment_ids.to(device)
			op1_mask = op1_mask.to(device)
			op2_mask = op2_mask.to(device)
			sent_label_id = sent_label_id.to(device)
			op_label_id = op_label_id.to(device)

			self.eval()
			with torch.no_grad():
				sent_logits, op_logits = self.__call__(input_ids, input_mask, 
	            									   op1_mask, op2_mask)
			op_predict = torch.max(op_logits, 1)[1]
			label = (op_predict).cpu().numpy().tolist()[0]
			probs = torch.nn.functional.softmax(op_logits, dim=1)
			all_probs += probs.cpu().numpy().tolist()

			pbar.update(1)
			cur += self.args.eval_batch_size

		results = {}
		for i in range(len(predict_features)):
			guid = predict_features[i].guid
			results[guid] = all_probs[i]
		return results

	def predict_label(self, context, ext1, ext2):
		processor = DataProcessor()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		n_gpu = torch.cuda.device_count()
		self.to(device)

		tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, 
											do_lower_case=self.args.do_lower_case)
		predict_example = [InputExample(0, context, ext1, ext2, sent_label="0", op_label="0")]
		predict_feature = convert_examples_to_features(
			predict_example, processor.get_labels(), 
			self.args.max_seq_length, tokenizer)

		input_id = torch.tensor([f.input_ids for f in predict_feature], dtype=torch.long).to(device)
		input_mask = torch.tensor([f.input_mask for f in predict_feature], dtype=torch.long).to(device)
		segment_ids = torch.tensor([f.segment_ids for f in predict_feature], dtype=torch.long).to(device)
		op1_mask = torch.tensor([f.op1_mask for f in predict_feature], dtype=torch.long).to(device)
		op2_mask = torch.tensor([f.op2_mask for f in predict_feature], dtype=torch.long).to(device)
		sent_label_id = torch.tensor([f.sent_label_id for f in predict_feature], dtype=torch.long).to(device)
		op_label_id = torch.tensor([f.op_label_id for f in predict_feature], dtype=torch.long).to(device)
		
		self.eval()
		with torch.no_grad():
			sent_logits, op_logits = self.__call__(input_id, input_mask, 
            									   op1_mask, op2_mask)
		op_predict = torch.max(op_logits, 1)[1]
		label = (op_predict).cpu().numpy().tolist()[0]
		probs = torch.nn.functional.softmax(op_logits, dim=1)
		probs = probs.cpu().numpy().tolist()[0]

		result = {"label": label, "probabilities":probs}

		return result

	def save_model(self, path):
		import pickle
		weights_path = os.path.join(path, "weights")
		params_path = os.path.join(path, "args")
		with open(params_path, 'wb') as f:
			pickle.dump(self.args, f)
		torch.save(self.state_dict(), weights_path)

	@staticmethod
	def load_from_pretrained(path):
		import pickle
		weights_path = os.path.join(path, "weights")
		params_path = os.path.join(path, "args")
		with open(params_path, 'rb') as f:
			args = pickle.load(f)
		model = Lstm_Trio_Attn_BERT(args, 2)
		if torch.cuda.is_available():
			model.load_state_dict(torch.load(weights_path))
		else:
			model.load_state_dict(torch.load(weights_path, 
											 map_location='cpu'))
		
		return model

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	## Required parameters
	parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain train/dev/test files for the task.")
	## Other parameters
	parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
	parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
	parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
	parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the test set.")
	parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
	parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
	parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
	args = parser.parse_args()

	if args.do_train:
		model = Lstm_Trio_Attn_BERT(args, 2)
		model.fit()
		if args.model_dir is not None:
			model.save_model(args.model_dir)
	else:
		model = Lstm_Trio_Attn_BERT.load_from_pretrained(args.model_dir)
	if args.do_eval:
		model.evaluate(args, "test")
