"""
This file contains the function to fine-tune the embedding
of the extractions.
AutoEncoder is modified from
https://github.com/stangelid/oposum/tree/master/mate
"""
from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_ as xavier_uniform
from torch.nn.utils import clip_grad_norm
from evaluation import Plot

from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Triplet Margin Cosine Loss.
def triplet_margin_cosine_loss(anchor, positive, negative, margin=1.0, eps=1e-8, sum_loss=False):
	assert anchor.dim() == 2, "Anchor and positive must be 2D matrices."
	assert negative.dim() <= 3, "Negative must be 2D (1 negative sample) or 3D matrix (multiple negatives)."
	assert margin > 0.0, 'Margin should be positive value.'

	d_p, d_n = None, None
	if positive.dim() == 2:
		d_p = F.cosine_similarity(anchor, positive, eps=eps).unsqueeze(1)
	else:
		d_p = F.cosine_similarity(anchor.unsqueeze(1), positive, dim=2, eps=eps)

	if negative.dim() == 2:
		d_n = F.cosine_similarity(anchor, negative, eps=eps).unsqueeze(1)
	else:
		d_n = d_n = F.cosine_similarity(anchor.unsqueeze(1), negative, dim=2, eps=eps)

	dist_hinge = torch.clamp(margin - d_p.unsqueeze(1) + d_n, min=0.0).mean(dim=1)

	if not sum_loss:
		loss = torch.mean(dist_hinge)
	else:
		loss = torch.sum(dist_hinge)

	return loss

class TripletMarginCosineLoss(nn.Module):
	def __init__(self, margin=1.0, eps=1e-8, sum_loss=False):
		super(TripletMarginCosineLoss, self).__init__()
		self.margin = margin
		self.eps = eps
		self.sum_loss = sum_loss

	def forward(self, anchor, positive, negative):
		return triplet_margin_cosine_loss(anchor, positive, negative, self.margin, self.eps, self.sum_loss)


class AttentionEncoder(nn.Module):
	"""Segment encoder that produces segment vectors as the weighted
	average of word embeddings.
	"""
	def __init__(self, vocab_size, emb_size, bias=True, M=None, b=None):
		"""Initializes the encoder using a [vocab_size x emb_size] embedding
		matrix. The encoder learns a matrix M, which may be initialized
		explicitely or randomly.

		Parameters:
			vocab_size (int): the vocabulary size
			emb_size (int): dimensionality of embeddings
			bias (bool): whether or not to use a bias vector
			M (matrix): the attention matrix (None for random)
			b (vector): the attention bias vector (None for random)
		"""
		super(AttentionEncoder, self).__init__()
		self.lookup = nn.Embedding(vocab_size, emb_size)
		self.M = nn.Parameter(torch.Tensor(emb_size, emb_size))
		if M is None:
			xavier_uniform(self.M.data)
		else:
			self.M.data.copy_(M)
		if bias:
			self.b = nn.Parameter(torch.Tensor(1))
			if b is None:
				self.b.data.zero_()
			else:
				self.b.data.copy_(b)
		else:
			self.b = None

	def forward(self, inputs):
		"""Forwards an input batch through the encoder"""
		x_wrd = self.lookup(inputs)
		x_avg = x_wrd.mean(dim=1)

		x = x_wrd.matmul(self.M)
		x = x.matmul(x_avg.unsqueeze(1).transpose(1,2))
		if self.b is not None:
			x += self.b

		x = torch.tanh(x)
		a = F.softmax(x, dim=1)

		z = a.transpose(1,2).matmul(x_wrd)
		z = z.squeeze()
		if z.dim() == 1:
			return z.unsqueeze(0)
		return z

	def set_word_embeddings(self, embeddings, fix_w_emb=True):
		"""Initialized word embeddings dictionary and defines if it is trainable"""
		self.lookup.weight.data.copy_(embeddings)
		self.lookup.weight.requires_grad = not fix_w_emb

class AutoEncoder(nn.Module):
	"""Modified from AspectAutoencoder, however, the goal is to encode
	target spans and fine-tune the embedding of target span by cluster
	assignments and additional classification signals.
	"""
	def __init__(self, target_cols, aux_c_sizes, vocab_size, emb_size, n_cluster, 
			n_sample=10, w_emb=None, c_emb=None, 
			attention=True, fix_w_emb=True, fix_c_emb=False):
		"""Initialization
		Parameters:
			target_cols (String[]): column names that need to be embedded
			aux_c_sizes (list of ints): number of classes for each
				auxiliary classification task.
			vocab_size (int): vocabular size
			emb_size (int): embedding dimension
			n_cluster (int): number of clusters
			num_sample (int): number of positive/negative examples
			w_emb (vocab_size * emb_size tensor): word embedding weights
			c_emb (n_cluster * emb_size tensor): cluster center embeeding 
				weights
			attention (True or False): whether use attention encoder to encode
				target fields
			fix_w_emb (True or False): fix the Embedding layer or not
			fix_c_emb (True or False): fix the center matrix or not
		"""
		super(AutoEncoder, self).__init__()
		self.n_cols = len(target_cols)
		self.aux_c_sizes = aux_c_sizes
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.n_cluster = n_cluster
		self.n_sample = n_sample
		self.attention = attention
		self.fix_w_emb = fix_w_emb
		self.fix_c_emb = fix_c_emb

		# Define Embedding Layer
		self.col_encoders = []
		for i in range(self.n_cols):
			col_encoder = None
			if not self.attention:
				col_encoder = nn.EmbeddingBag(vocab_size, emb_size)
				self.add_module("EmbeddingBag_%d"%i, col_encoder)
			else:
				col_encoder = AttentionEncoder(vocab_size, emb_size)
				self.add_module("AttentionEncoder_%d"%i, col_encoder)
			self.col_encoders.append(col_encoder)

		assert emb_size % self.n_cols == 0, "Cannot recover emb size."
		self.emb_linear = nn.Linear(emb_size, int(emb_size/self.n_cols))

		# Initialize wording embedding weights
		if w_emb is None:
			for col_encoder in self.col_encoders:
				xavier_uniform(col_encoder.weight.data)
		else:
			self._assert_size(w_emb, (vocab_size, emb_size), "Word embedding")
			for col_encoder in self.col_encoders:
				if not attention:
					col_encoder.weight.data.copy_(w_emb)
					col_encoder.weight.requires_grad = not fix_w_emb
				else:
					col_encoder.set_word_embeddings(w_emb, fix_w_emb)

		# Initalize representative weights.
		if c_emb is None:
			self.c_emb = nn.Parameter(torch.tensor(self.n_cluster, emb_size))
			xavier_uniform(self.c_emb.data)
		else:
			self._assert_size(c_emb, (self.n_cluster, emb_size), "Center")
			self.c_emb = nn.Parameter(torch.Tensor(c_emb.size()))
			self.c_emb.data.copy_(c_emb)
			self.c_emb.requires_grad = not fix_c_emb

		# Define cluster classification layer
		self.lin = nn.Linear(emb_size, self.n_cluster)

		# Define other classification layers
		self.aux_lins = []
		for c_size in self.aux_c_sizes:
			lin = nn.Linear(emb_size, c_size)
			self.add_module("Classifier_%d" % c_size, lin)
			self.aux_lins.append(lin)

		self.softmax = nn.Softmax(dim=1)

	def forward_one(self, input_arry):
		# Encode every fields in the targets
		encs = []
		for i in range(len(input_arry)):
			if not self.attention:
				offsets = Variable(torch.arange(0, input_arry[i].numel(), input_arry[i].size(1), out=input_arry[i].data.new().long()))
				encs.append(self.emb_linear(self.col_encoders[i](input_arry[i].view(-1), offsets)))
			else:
				encs.append(self.emb_linear(self.col_encoders[i](input_arry[i])))
		
		# Combine all fields
		#enc = self.enc_combiner(torch.cat(encs, dim=1))
		enc = torch.cat(encs, dim=1)

		# Find cluster assignment distribution
		x = self.lin(enc)
		c_probs = self.softmax(x)
		return enc, x, c_probs

	def forward(self, input_arry, input_other=None):
		enc, x, c_probs = self.forward_one(input_arry)
		
		if input_other is not None:
			enc_other, x_other, c_probs_other = self.forward_one(input_other)
			return c_probs, c_probs_other
		
		reconn_enc = torch.mm(x, self.c_emb)
		# Run classifers
		a_probss = [self.softmax(lin(enc)) for lin in self.aux_lins]
	
		# Select positive and negative examples
		self.set_target(enc, c_probs)
		#self.set_target_by_centers(c_probs)
		return c_probs, a_probss, enc, reconn_enc

	def set_target(self, enc, c_probs):
		batch_size, emb_size = enc.size()
		neg_mask = self._create_mask(batch_size, c_probs, self.n_sample, emb_size, mtype='neg')
		self.negative = Variable(enc.expand(batch_size, batch_size, emb_size).gather(1, neg_mask))
		pos_mask = self._create_mask(batch_size, c_probs, self.n_sample, emb_size, mtype='pos')
		self.positive = Variable(enc.expand(batch_size, batch_size, emb_size).gather(1, pos_mask))

	def get_targets(self):
		assert self.positive is not None, 'Positive targets not set; needs a forward pass first'
		assert self.negative is not None, 'Negative targets not set; needs a forward pass first'
		return self.positive, self.negative

	def set_target_by_centers(self, c_probs):
		batch_size, n_cluster = c_probs.size()
		neg_mask = torch.multinomial(c_probs, self.n_sample).unsqueeze(2).expand(batch_size, self.n_sample, self.emb_size) 
		pos_mask = torch.multinomial(1/(c_probs+1e-10), self.n_sample).unsqueeze(2).expand(batch_size, self.n_sample, self.emb_size)
		#print(neg_mask.size(), self.c_emb.size(), self.c_emb.expand(batch_size, n_cluster, self.emb_size).size())
		self.positive = Variable(self.c_emb.expand(batch_size, n_cluster, self.emb_size).gather(1, neg_mask))
		self.negative = Variable(self.c_emb.expand(batch_size, n_cluster, self.emb_size).gather(1, pos_mask))

	def _create_mask(self, batch_size, probs, n_sample, emb_size, mtype='neg', 
					 eps=1e-20):
		n_sample = min(batch_size - 1, n_sample)
		probs_e = probs.unsqueeze(0).expand(batch_size, batch_size, -1)
		prob_1 = probs_e.flatten(0,1)
		prob_2 = probs_e.transpose(1,0).flatten(0,1)
		sim = F.cosine_similarity(prob_1, prob_2, dim=1)
		weight = 1-sim if mtype=='neg' else sim
		weight = weight.view(batch_size, batch_size)
		diagnal_mask = torch.eye(batch_size, batch_size, device=device).bool()
		weight.masked_fill_(diagnal_mask, torch.tensor(0., device=device))
		weight = torch.max(weight, torch.tensor(eps, device=device))
		mask = torch.multinomial(weight, n_sample)
		mask = mask.unsqueeze(2).expand(batch_size, n_sample, emb_size)
		return mask

	def _assert_size(self, tensor, size, name):
		assert tensor.size() == size, "%s: %s != %s" % (name, 
			str(tensor.size()), str(size))

class EmbEncoder:
	def __init__(self, target_cols, auxiliary_cols, edges, w_emb, n_sample, 
		n_epochs, batch_size, lr=0.5, n_inititer=5, max_attemp=1):
		""" Initialization
		Parameters:
			target_cols (String[]): column names that need to be embedded
			auxiliary_cols (list of dicts): columns that contain additional
				labels, together with the label info.
				e.g., {"col": "attribute", "n_class": 17, "loss_ratio":0.5}

			w_emb (vocab_size * emb_size tensor): initial tensor for Embedding
				layer.
			n_sample (int): number of positive / negative examples used
				for reconstruction loss
			n_epochs (int): number of epochs for training the embedding.
			lr (float): learning rate
			n_inititer (int): number of iterations to fine-tune cluster 
				classification layers (lin)
		"""
		self.target_cols = target_cols
		self.auxiliary_cols = auxiliary_cols
		self.edge_loss_ratio = edges["loss_ratio"]
		self.edge_loss_lr = edges["lr"]
		self.w_emb = torch.tensor(w_emb)
		self.n_sample = n_sample
		self.n_epochs = n_epochs
		self.lr = lr
		self.batch_size = batch_size
		self.n_inititer = n_inititer
		self.max_attemp = max_attemp

	def run(self, aux_c_sizes, row_iter, label_iter, c_emb, spans, gold, 
			output_dir, pair_iter=None, aux_weights=None):
		best_enc = None
		best_loss = None
		best_labels = None
		for i in range(self.max_attemp):
			print("\t", "attemp %d" % i)
			if pair_iter is None:
				fine_tuned_enc, loss, fine_tuned_label = self.refine(aux_c_sizes, row_iter, 
					label_iter, c_emb, spans=spans, gold=gold, 
					output_dir=output_dir, aux_weights=aux_weights)
			else:
				fine_tuned_enc, loss, fine_tuned_label = self.refine_with_pair(aux_c_sizes, 
					row_iter, pair_iter, label_iter, c_emb, spans=spans, gold=gold, 
					output_dir=output_dir, aux_weights=aux_weights)
			if best_loss is None or loss < best_loss:
				best_enc = fine_tuned_enc
				best_loss = loss
				best_labels = fine_tuned_label
		return best_enc, best_loss, fine_tuned_label

	def refine_with_pair(self, aux_c_sizes, row_iter, pair_iter, label_iter, c_emb,
			   spans=None, gold=None, output_dir=None, output_interval=5, aux_weights=None):
		if gold is None or gold[0] is None:
			output_dir = None
		n_cluster, emb_size = c_emb.size()
		vocab_size, _ = self.w_emb.size()

		# Initialize the Encoder model
		model = AutoEncoder(self.target_cols, aux_c_sizes,
			vocab_size, emb_size, n_cluster, n_sample=self.n_sample,
			w_emb=self.w_emb, c_emb=c_emb, attention=True,
			fix_w_emb=True, fix_c_emb=True)
		model.to(device)
		model.train()

		# Define optimizer
		params = filter(lambda p: p.requires_grad, model.parameters())
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		kl_div_optimizer = torch.optim.Adam(model.parameters(), lr=self.edge_loss_lr, amsgrad=True)

		# Define loss functions
		rec_loss = TripletMarginCosineLoss()
		cls_loss = nn.CrossEntropyLoss()
		kl_loss = nn.KLDivLoss(reduction='batchmean')

		fine_tuned_enc = None
		total_loss = None
		fine_tuned_label = None
		pbar = tqdm(total = self.n_epochs)
		times = [0. for i in range(4)]

		aux_losses = []
		if aux_weights is not None:
			for aux_weight in aux_weights:
				aux_losses.append(nn.CrossEntropyLoss(weight=torch.FloatTensor(aux_weight).to(device)))
		
		for epoch in range(self.n_epochs+1):
			encs = []
			probs = []
			curr_loss = 0	
			for idx, (batch, pair_batch, c_labels) in enumerate(zip(row_iter, cycle(pair_iter), label_iter)):			
				start_time = time.time()
				# Process data
				target_idss, a_labelss = batch
				target_idss = target_idss.transpose(1, 0)
				a_labelss = a_labelss.transpose(1, 0)
				target_idss = [ids.to(device) for ids in target_idss]
				a_labelss = [labels.to(device) for labels in a_labelss]

				# Process pairs
				left_ids, right_ids = pair_batch
				left_ids, right_ids = left_ids.transpose(1, 0), right_ids.transpose(1, 0)
				left_ids = [ids.to(device) for ids in left_ids]
				right_ids = [ids.to(device) for ids in right_ids]

				# Process clustered labels
				c_labels = c_labels.to(device)
				c_probs, a_probss, enc, reconn_enc = model(target_idss)
				times[0] += time.time()-start_time
				start_time = time.time()

				# Update fine tune rep_probs loss.
				positives, negatives = model.get_targets()

				if idx < self.n_inititer:
					loss = cls_loss(c_probs, c_labels)
				else:
					loss = rec_loss(reconn_enc, enc, negatives)	
				
				# Update classification loss, e.g., for type or sentiment.
				for i, (a_probs, a_labels) in enumerate(zip(a_probss, a_labelss)):
					#print(a_labels, torch.var(a_labels.double()))
					if aux_weights is None:
						aux_loss = self.auxiliary_cols[i]["loss_ratio"]*cls_loss(a_probs, a_labels)
					else:
						aux_loss = self.auxiliary_cols[i]["loss_ratio"]*aux_losses[i](a_probs, a_labels)
					loss += aux_loss

				# Back propogation
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
				curr_loss += loss.detach().item()

				# Update KL loss for pairs
				if self.edge_loss_ratio > 0:
					prob_left, prob_right = model(left_ids, right_ids)
					edge_loss = self.edge_loss_ratio*(-kl_loss(prob_left.log(), prob_right))
					
					# Back propogation
					kl_div_optimizer.zero_grad()
					edge_loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					kl_div_optimizer.step()
					curr_loss += edge_loss.detach().item()
					#print("\t", edge_loss.detach().item())

					#for name, param in model.named_parameters():
					#	if param.requires_grad:
					#		print(epoch, idx, name, param.grad)
					#		break

				times[1] += time.time() - start_time
				start_time = time.time()
				

				times[2] += time.time()-start_time
				start_time = time.time()

				# Update variables
				encs.append(enc.detach().cpu())
				probs.append(c_probs.detach().cpu().argmax(1))
				times[3] += time.time()-start_time

			#print(curr_loss)

			if total_loss is None or curr_loss < total_loss:
				fine_tuned_enc = torch.cat(encs, dim=0)
				total_loss = curr_loss
				fine_tuned_label = torch.cat(probs, dim=0)

			pbar.update(1)

			# Print intermediate embeddings
			if output_dir is None or epoch % output_interval != 0:
				continue
			#Plot(output_dir, fine_tuned_enc, spans, gold, str(epoch))

		#print(times)
		return fine_tuned_enc, total_loss, fine_tuned_label

	def refine(self, aux_c_sizes, row_iter, label_iter, c_emb, 
			   spans=None, gold=None, output_dir=None, output_interval=5, aux_weights=None):
		""" Fine-turning embeddings
		Parameters:
			aux_c_sizes (list of ints): number of classes for each
				auxiliary classification task.
			row_iter: batches of data points
			label_iter: batches of cluster labels
			c_emb: representative embedding tensor from cluster algorithm
		"""
		if gold is None or gold[0] is None:
			output_dir = None
		n_cluster, emb_size = c_emb.size()
		vocab_size, _ = self.w_emb.size()

		# Initialize the Encoder model
		model = AutoEncoder(self.target_cols, aux_c_sizes,
			vocab_size, emb_size, n_cluster, n_sample=self.n_sample,
			w_emb=self.w_emb, c_emb=c_emb, attention=True,
			fix_w_emb=True, fix_c_emb=True)
		model.to(device)
		model.train()

		# Define optimizer
		params = filter(lambda p: p.requires_grad, model.parameters())
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

		# Define loss functions
		rec_loss = TripletMarginCosineLoss()
		cls_loss = nn.CrossEntropyLoss()

		aux_losses = []
		if aux_weights is not None:
			for aux_weight in aux_weights:
				aux_losses.append(nn.CrossEntropyLoss(weight=torch.FloatTensor(aux_weight).to(device)))

		fine_tuned_enc = None
		total_loss = None
		fine_tuned_label = None
		pbar = tqdm(total = self.n_epochs)
		times = [0. for i in range(4)]
		
		for epoch in range(self.n_epochs+1):
			encs = []
			probs = []
			curr_loss = 0	
			for idx, (batch, c_labels) in enumerate(zip(row_iter, label_iter)):
				start_time = time.time()
				target_idss, a_labelss = batch
				target_idss = target_idss.transpose(1, 0)
				a_labelss = a_labelss.transpose(1, 0)
				target_idss = [ids.to(device) for ids in target_idss]
				a_labelss = [labels.to(device) for labels in a_labelss]
				c_labels = c_labels.to(device)
				c_probs, a_probss, enc, reconn_enc = model(target_idss)
				times[0] += time.time()-start_time
				start_time = time.time()

				# Update fine tune rep_probs loss.
				positives, negatives = model.get_targets()

				if idx < self.n_inititer:
					loss = cls_loss(c_probs, c_labels)
				else:
					loss = rec_loss(enc, positives, negatives)	
				#print(loss.item())
				
				# Update classification loss, e.g., for type or sentiment.
				for i, (a_probs, a_labels) in enumerate(zip(a_probss, a_labelss)):
					#print(a_labels, torch.var(a_labels.double()))
					if aux_weights is None:
						aux_loss = self.auxiliary_cols[i]["loss_ratio"]*cls_loss(a_probs, a_labels)
					else:
						aux_loss = self.auxiliary_cols[i]["loss_ratio"]*aux_losses[i](a_probs, a_labels)
					loss += aux_loss
					#print(aux_loss.item())

				times[1] += time.time() - start_time
				start_time = time.time()

				# Back propogation
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()

				times[2] += time.time()-start_time
				start_time = time.time()

				# Update variables
				curr_loss += loss.detach().item()
				encs.append(enc.detach().cpu())
				probs.append(c_probs.detach().cpu().argmax(1))
				times[3] += time.time()-start_time

			if total_loss is None or curr_loss < total_loss:
				fine_tuned_enc = torch.cat(encs, dim=0)
				total_loss = curr_loss
				fine_tuned_label = torch.cat(probs, dim=0)

			pbar.update(1)
			# Print intermediate embeddings
			if output_dir is None or epoch % output_interval != 0:
				continue
			#Plot(output_dir, fine_tuned_enc, spans, gold, str(epoch))

		return fine_tuned_enc, total_loss, fine_tuned_label
