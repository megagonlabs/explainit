"""
This file contains several basic clustering algorithm:
1. Kmeans
2. GMM
3. Greedy solution for correlation clustering
"""
import math
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from scipy.cluster.vq import kmeans
from scipy.stats import hmean
from sklearn import mixture
from random import shuffle

from tqdm import tqdm

_VALID_METHODS = ["kmeans", "gmm", "greedy", "kmeans_vqvae"]
_INVALID_ERRMSG = "Basic cluster type %s not supported!"
_MISSINGK_ERRMSG = "Must specify number of cluster (K) for %s!"	
_SIM_MIN = 1e-6
_SIM_MAX = 1-1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_nearest(target, source):
	# Represent target with the nearest neighbor in source.
	t_size = target.size(0)
	s_size = source.size(0)

	t_extend = target.unsqueeze(1).expand(t_size, s_size, -1)
	s_extend = source.unsqueeze(0).expand(t_size, s_size, -1)

	dist = torch.norm(t_extend-s_extend, dim=2)
	min_dist_index = dist.argmin(dim=1)
	one_hot = torch.nn.functional.one_hot(min_dist_index, s_size)
	return min_dist_index, one_hot

class Vqvae(torch.nn.Module):
	def __init__(self, n_cluster, emb_size, repr_tensor=None):
		super(Vqvae, self).__init__()	
		self.n_cluster = n_cluster
		self.emb_size = emb_size
		self.repr_tensor = torch.nn.Parameter(torch.Tensor(n_cluster, emb_size))
		if repr_tensor is not None:
			self.repr_tensor.data.copy_(repr_tensor)
		else:
			xavier_uniform(self.repr_tensor.data)

	def reconn_by_nearest(self, target, source):
		_, one_hot = find_nearest(target, source)
		one_hot = one_hot.type(torch.FloatTensor).to(device)
		source = source.type(torch.FloatTensor).to(device)
		return torch.mm(one_hot, source)

	def forward(self, emb_tensor):
		x_recon = self.reconn_by_nearest(emb_tensor, self.repr_tensor)
		repr_x = self.reconn_by_nearest(self.repr_tensor, emb_tensor)
		return x_recon, emb_tensor, self.repr_tensor, repr_x

class KmeansVqvae:
	def __init__(self, emb_tensor, n_cluster, max_attemp=10, max_iter=15):
		self.emb_size = emb_tensor.size(1)
		self.n_cluster = n_cluster
		self.emb_tensor = emb_tensor.float()
		self.max_attemp = max_attemp
		self.max_iter = max_iter
		self.repr_tensor = None

	def run_cluster(self, batch_size=32, lr=0.05, beta=1.0):
		model = Vqvae(self.n_cluster, self.emb_size, repr_tensor=self.select_random())
		model.to(device)
		model.train()
		emb_iter = torch.utils.data.DataLoader(self.emb_tensor, batch_size=batch_size)
		loss_func = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
		#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
		reconn_losses,recons = [], []
		pbar = tqdm(total = self.max_iter)
		for epoch in range(self.max_iter):
			recon = []
			reconn_losses.append(0.)
			for idx, x in enumerate(emb_iter):
				x = x.to(device)
				x_recon, emb_tensor, repr_tensor, repr_x = model(x)
				#print(x_recon.size(), x_recon)
				#print(x.size(), x)
				recon_loss = loss_func(x_recon, x)
				sg_z_to_e_loss = loss_func(repr_x.detach(), repr_tensor)
				z_to_sg_e_loss = loss_func(repr_x, repr_tensor.detach())
				loss = recon_loss + sg_z_to_e_loss + beta*z_to_sg_e_loss

				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()

				reconn_losses[-1] += recon_loss.detach().item()
				recon.append(x_recon.detach().data)
			recons.append(torch.cat(recon, dim=0))
			pbar.update(1)
		repr_tensor = model.repr_tensor.detach().data
		return repr_tensor, reconn_losses[-1]

	def select_random(self):
		weight = torch.ones(self.emb_tensor.size(0))
		indices = torch.multinomial(weight, self.n_cluster)
		repr_tensor = torch.index_select(self.emb_tensor, 0, indices)
		return repr_tensor

	def run(self):
		min_loss = torch.norm(self.emb_tensor).item()
		for i in range(self.max_attemp):
			print("\t", "attemp %d" % i)
			repr_tensor, loss = self.run_cluster()
			if loss < min_loss:
				self.repr_tensor = repr_tensor
		assignment, _ = find_nearest(self.emb_tensor, self.repr_tensor)
		return self.repr_tensor, list(assignment.numpy()), min_loss

class KmeansCluster:
	def __init__(self, emb_tensor, n_cluster, max_attemp=10, max_iter=10000):
		self.emb_tensor = emb_tensor
		self.n_element = len(emb_tensor)
		self.max_attemp = max_attemp
		self.elements = emb_tensor
		self.n_cluster = n_cluster
		self.repr_tensor = None
		self.max_iter = max_iter

	def run(self):
		# Run kmeans for several times, select the best attemp
		min_loss = torch.norm(self.emb_tensor).item()
		emb_array = self.emb_tensor.numpy()
		for i in range(self.max_attemp):
			print("\t", "attemp %d" % i)
			repr_tensor, loss = kmeans(emb_array, self.n_cluster, 
				iter=self.max_iter)
			if loss < min_loss:
				self.repr_tensor = repr_tensor
				min_loss = loss
		self.repr_tensor = torch.tensor(self.repr_tensor)
		assignment, _ = find_nearest(self.emb_tensor, self.repr_tensor)
		return self.repr_tensor, list(assignment.numpy()), min_loss

class GreedyCluster:
	"""
	This greedy correlation clustering contains three major components:
	1. The scoring function: measures the gain/cost of merging two clusters
	2. The greedy merging mechanism: assign data points into clusters
	3. Local search optimization: perform local search to optimize the result
	Modified from Nofar's code.
	"""
	def __init__(self, emb_tensor, threshold=0.90, max_attemp=10, max_step=10):
		"""Initialization
		Parameters:
			emb_tensor (N * emb_size): embedding tensor
			max_attemps (int): maximum attemps for greedy solution;
			max_steps (int): maximum steps in local search.
		"""
		self.max_attemp = max_attemp
		self.elements = emb_tensor
		self.max_step = max_step

		self.n_element = len(self.elements)
		self.clusters = []
		self.score = None
		self.pow = int(math.log(0.5)/math.log(threshold))

	def run(self):
		"""Greedy algorithm:
		1. Run greedy multiple times, take the best attemp;
		2. Run local search to further refine the result.
		"""
		best_score = None
		for i in range(self.max_attemp):
			print("\t", "attemp %d" % i)
			clusters = self.greedy()
			score = 0#self.scoring(clusters)
			if best_score is None or score > best_score:
				self.clusters = clusters
				best_score = score
				self.score = best_score
		#self.local_search()
		return self.get_repr(), self.get_assignment(), self.get_score()


	def scoring(self, clusters):
		"""Calcuate the score according to correlation clustering objective
		function:
		score = sum(log(P(e1, e2))) + sum(log(1-P(e3, e4))), 
		where e1, e2 belong to the same cluster; e3, e4 belong to different
		clusters; P is the similarity measurement for two data points.
		"""
		min_score, max_score = 1, 0
		score = 0
		scores = []
		for cluster in clusters:
			for i in range(len(cluster)):
				for j in range(i+1, len(cluster)):
					sim = self.get_pairwise_similarity(cluster[i], cluster[j])
					score += math.log(sim)
					scores.append(sim)
		# Calculate cross cluster score
		for cid1 in range(len(clusters)):
			for cid2 in range(cid1+1, len(clusters)):
				for eid1 in clusters[cid1]:
					for eid2 in clusters[cid2]:
						sim = self.get_pairwise_similarity(eid1, eid2)
						score += math.log(1-sim)
						scores.append(sim)
		dist = [0 for i in range(10)]
		for i in scores:
			idx = int(i*10)
			dist[idx] += 1
		return score

	def greedy(self):
		"""Greedy solution
		1. Shuffle elements/data points;
		2. Iterate all elements and find the best local assignment.
		"""
		clusters = []
		permutation = [i for i in range(self.n_element)]
		pbar = tqdm(total=len(permutation))
		shuffle(permutation)
		for eid in permutation:
			best_cluster = -1
			best_cluster_score = 0
			for cid in range(len(clusters)):
				cluster_score = 0
				for eid2 in clusters[cid]:
					sim = self.get_pairwise_similarity(eid, eid2)
					gain = math.log(sim)-math.log(1-sim)
					cluster_score += gain if gain >= 0 else -1000.0
					#cluster_score = min(cluster_score, gain)
				if cluster_score > best_cluster_score:
					best_cluster = cid
					best_cluster_score = cluster_score
				#print(eid, clusters[cid], cluster_score)
			if best_cluster < 0:
				clusters.append([eid])
			else:
				clusters[best_cluster].append(eid)
			pbar.update(1)
		return clusters

	def get_pairwise_similarity(self, eid1, eid2):
		"""Simlarity measurement for two elements
		"""
		sim = F.cosine_similarity(self.elements[eid1], 
				self.elements[eid2], dim=0).item()
		sim = min(max(sim, _SIM_MIN), _SIM_MAX)
		sim = math.pow(sim, self.pow)
		return sim

	def local_search(self):
		"""Local search refinement
		Try to move every element from current cluster to other/new clusters;
		Track the best move action;
		Perform the move if it has positive gain.
		"""
		step = 0
		while step < self.max_step:
			best_gain = 0
			best_move = None
			# Iteratively search for possible moves
			for orig_cid in range(len(self.clusters)):
				for eid in self.clusters[orig_cid]:
					# Try to move element to a new cluster
					gain = self.local_search_gain(eid, orig_cid, -1)
					if gain > best_gain:
						best_move = (eid, orig_cid, -1)
						best_gain = gain
					# Try to move element into other clusters
					for dest_cid in range(len(self.clusters)):
						if orig_cid == dest_cid:
							continue
						gain = self.local_search_gain(eid, orig_cid, dest_cid)
						if gain > best_gain:
							best_move = (eid, orig_cid, dest_cid)
							best_gain = gain
		
			if best_gain > 0:
				# perform the move
				eid, orig_cid, dest_cid = best_move
				if dest_cid == -1:
					self.clusters.append([eid])
				else:
					self.clusters[dest_cid].append(eid)
				self.clusters[orig_cid].remove(eid)
				if len(self.clusters[orig_cid]) == 0:
					self.clusters.pop(orig_cid)
			else:
				break
			step += 1
		if step > 0:
			self.score = self.scoring(self.clusters)

	def local_search_gain(self, eid, orig_cid, dest_cid):
		"""Calculate the gain of moving an element from original cluster to
		a new destination cluster.
		"""
		gain = 0
		# Recalculate in-cluster gain from original cluster
		for eid2 in self.clusters[orig_cid]:
			if eid2 == eid:
				continue
			sim = self.get_pairwise_similarity(eid, eid2)
			gain += (-math.log(sim) + math.log(1-sim))
		
		# Calcuate in-cluster gain from destination cluster
		dest_cluster = self.clusters[dest_cid] if dest_cid >= 0 else []
		for eid2 in dest_cluster:
			sim = self.get_pairwise_similarity(eid, eid2)
			gain += (math.log(sim) - math.log(1-sim))
		return gain

	def get_score(self):
		"""Return the score of the clustering
		"""
		if self.score is not None:
			return self.score
		self.score = self.scoring(self.clusters)
		return self.score

	def get_repr(self):
		"""Calculate the representatives for each cluster as the mean of their
		embeddings.
		"""
		repr_vecs = []
		pbar = tqdm(total=len(self.clusters))
		for cluster in self.clusters:
			element_vecs = [self.elements[i] for i in cluster]
			element_vec = torch.mean(torch.stack(element_vecs), dim=0)
			repr_vecs.append(element_vec)
			pbar.update(1)
		return torch.stack(repr_vecs)

	def get_assignment(self):
		"""Return the cluster assignment.
		"""
		assign = [-1 for i in range(self.n_element)]
		pbar = tqdm(total=len(self.clusters))
		for cid in range(len(self.clusters)):
			for eid in self.clusters[cid]:
				assign[eid] = cid
			pbar.update(1)
		return assign

class Cluster:
	def __init__(self, method, n_cluster, max_iter, max_attemp=1):
		assert method in _VALID_METHODS, _INVALID_ERRMSG % method
		assert (n_cluster is not None or method == "greedy"), _MISSINGK_ERRMSG % method
		self.method = method
		self.n_cluster = int(n_cluster)
		self.max_iter = int(max_iter)
		self.max_attemp = max_attemp

	def run(self, emb_tensor):
		"""Run the cluster algorithm.
		Parameters:
			emb_tensor (N * (n_field) * emb_size): embedding tensor
		Output:
			repr_tensor (n_cluster * emb_size): representative emb for each cluster
			assign (N): mapping between input data point to cluster
			loss (float): loss value from the algorithm
		"""	
		# Check if n_cluster is greater than or equals to N.
		N = emb_tensor.size(0)
		if N <= self.n_cluster:
			repr_tensor = emb_tensor
			assign = [i for i in range(N)]
			return repr_tensor, assign, 0.0
		# Perform clustering algorithm.
		repr_tensor, assign, loss = None, None, None
		if self.method == "kmeans":
			repr_tensor, assign, loss = self.run_kmeans(emb_tensor)
		elif self.method == "gmm":
			repr_tensor, assign, loss = self.run_gmm(emb_tensor)
		elif self.method == "kmeans_vqvae":
			repr_tensor, assign, loss = self.run_kmeansvqvae(emb_tensor)
		else:
			repr_tensor, assign, loss = self.run_greedy(emb_tensor)
		return repr_tensor, assign, loss

	def run_kmeansvqvae(self, emb_tensor):
		kmeans_vqvae = KmeansVqvae(emb_tensor, self.n_cluster,
			max_attemp=self.max_attemp, max_iter=self.max_iter)
		repr_tensor, assign, loss = kmeans_vqvae.run()
		return repr_tensor, assign, loss

	def run_kmeans(self, emb_tensor):
		kmeans = KmeansCluster(emb_tensor, self.n_cluster, 
			max_attemp=self.max_attemp, max_iter=self.max_iter)
		repr_tensor, assign, loss = kmeans.run()
		return repr_tensor, assign, loss

	def run_gmm(self, emb_tensor):
		# Fit model
		emb_matrix = emb_tensor.numpy()
		best_score = 0
		best_predicts = None
		best_repr_tensor = None
		for i in range(self.max_attemp):
			gmm = mixture.GaussianMixture(n_components=self.n_cluster, 
				covariance_type='full').fit(emb_matrix)
			predicts = gmm.predict(emb_matrix)
			# Get clusters
			clusters = {}
			for i in range(len(predicts)):
				clusters[predicts[i]] = clusters[predicts[i]]+[i] if predicts[i] in clusters else [i]
			# Calculate repr_tensor
			repr_tensor = []
			for cluster, indices in clusters.items():
				repr_tensor.append(torch.mean(torch.index_select(emb_tensor, 0, torch.tensor(indices)), dim=0))
			repr_tensor = torch.stack(repr_tensor, dim=0)
			score = gmm.score(emb_matrix)
			if score > best_score:
				best_repr_tensor = repr_tensor
				best_predicts = predicts
				best_score = score
		return best_repr_tensor, best_predicts, best_score

	def run_greedy(self, emb_tensor):
		greedy = GreedyCluster(emb_tensor, max_attemp=self.max_attemp)
		repr_tensor, assign, score = greedy.run()
		return repr_tensor, assign, score
