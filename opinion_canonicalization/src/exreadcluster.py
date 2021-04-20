"""
This file contains functions to perform the two steps 
clustering algorithm.
"""
from cluster import Cluster
from embedding import EmbEncoder
from load_data import W2VLoader, DataLoader, PairLoader, LabelLoader, FileLoader
from evaluation import Evaluate, Plot, CrossAttributeCluster
import os
import pickle

class ExreadCluster:
	def __init__(self, target_cols, auxiliary_cols, edges,
		cluster_config, emb_config, max_attemp):
		"""Initialize Two-steps cluster framework.
		Parameters:
			target_cols (String[]): target columns
			auxiliary_cols (list of dicts): columns that contain additional
				labels, together with the label info.
				e.g., {"col": "attribute", "n_class": 17, "loss_ratio":0.5}
			cluster_config (dict): base cluster module parameters
			emb_config (dict): embedding fine-tuning parameters.
		"""
		
		# Load cluster module
		self.target_cols = target_cols
		self.auxiliary_cols = auxiliary_cols
		self.cluster_module = Cluster(cluster_config["method"], 
									  cluster_config["n_cluster"], 
									  cluster_config["max_iter"],
									  max_attemp=max_attemp)
		self.w2v_model = W2VLoader(emb_config["w2v_model"])
		# Load embedding fine-tuning module
		self.emb_module = EmbEncoder(target_cols, auxiliary_cols,
									 edges, self.w2v_model.w_emb, 
									 emb_config["n_sample"],
									 emb_config["n_epochs"], 
									 emb_config["batch_size"],
									 lr=emb_config["lr"], 
									 n_inititer=emb_config["n_inititer"],
									 max_attemp=max_attemp)
		self.batch_size = emb_config["batch_size"]

	def get_basics(self, row_header, rows):
		target_rows, aux_rows = [], []
		gold, spans = [], []
		for row in rows:
			target = [row[row_header[col]] for col in self.target_cols]
			aux_labels = [row[row_header[col["col"]]] for col in self.auxiliary_cols]
			target_rows.append(target)
			aux_rows.append(aux_labels)
			# Update meta information
			gold.append(row[row_header["gold"]] if "gold" in row_header else None)
			spans.append(" ".join(target))
		return target_rows, aux_rows, gold, spans

	def get_pairs(self, rows):
		assert len(rows) == 0 or len(rows[0]) == 2*len(self.target_cols), "Incorrect # of cols in pair file!"
		pairs = []
		for row in rows:
			pair = [row[:len(self.target_cols)],row[len(self.target_cols):]]
			pairs.append(pair)
		return pairs

	def post_processing(self, row_header, rows, labels):
		enforced_rows = []
		for aux_row in self.auxiliary_cols:
			if "enforce" in aux_row and aux_row["enforce"].lower() == "true":
				enforced_rows.append(aux_row["col"])
		# create label map
		label_map = {}
		updated_labels = []
		for i in range(len(rows)):
			label = str(labels[i])
			enforced = "-".join([rows[i][row_header[col]] for col in enforced_rows])
			key = label + ":" + enforced
			if key not in label_map:
				label_map[key] = len(label_map)
			updated_labels.append(label_map[key])
		return updated_labels

	def run(self, data_filename, do_eval=None, output_dir=None): 
		"""Run the two-steps ExreadCluster algorithm (no edge information).
		Parameter:
			row_header (String[]): column names
			rows (String[[]]): array of string arrays.
		Output:
			data2rep (int array): cluster assignment for each data
			r_emb (K * emb_size): representative embedding
			emb_tensor (N * emb_size): fine-tuned input embedding
			cls_loss (float): clustering loss
		"""
		accuracies = []
		# Get data.
		row_header, rows = FileLoader(data_filename)
		target_rows, aux_rows, gold, spans = self.get_basics(row_header, rows)

		# Generate batches.
		row_iter, aux_c_sizes, aux_weights = DataLoader(target_rows, aux_rows, 
			self.batch_size, self.w2v_model)

		# Get the initial embedding tensor as the average w2v embedding.
		emb_tensor = self.w2v_model.get_emb_matrix(target_rows)
		if do_eval is not None:
			with open(os.path.join(output_dir, "emb_before"), "wb") as handle:
				pickle.dump((emb_tensor, spans, gold), handle, 
					protocol=pickle.HIGHEST_PROTOCOL)
		
		# Run the base cluster module
		print("Run 1st module: clustering algorithm")
		c_emb, labels, cls_loss = self.cluster_module.run(emb_tensor)
	
		# Run the embedding fine-tuning module
		print("Run 2nd module: refine embedding")
		enc, emb_loss, emb_labels = self.emb_module.run(aux_c_sizes, row_iter,
			LabelLoader(labels, self.batch_size), c_emb, spans, gold,
			output_dir, aux_weights=aux_weights)
		print("****cluster loss: %f; emb loss:%f****" % (cls_loss, emb_loss))
		
		# Update embedding tensor
		emb_tensor = enc.data

		if do_eval is not None:
			accuracies.append(Evaluate(labels, gold, do_eval))	
		
		print("Run 3rd module: refinement by clustering algorithm")
		# Final refinement
		c_emb, labels, cls_loss = self.cluster_module.run(emb_tensor)		
		labels = self.post_processing(row_header, rows, labels)

		if do_eval is not None:
			accuracies.append(Evaluate(labels, gold, do_eval))	
			with open(os.path.join(output_dir, "emb_after"), "wb") as handle:
				pickle.dump((emb_tensor, spans, gold), handle, 
					protocol=pickle.HIGHEST_PROTOCOL)

		return row_header, rows, labels, c_emb, emb_tensor, cls_loss, accuracies

	def run_with_edge(self, data_filename, pair_filename, do_eval=None, output_dir=None): 
		"""Run the two-steps ExreadCluster algorithm (with edges).
		Parameter:
			row_header (String[]): column names
			rows (String[[]]): array of string arrays.
		Output:
			data2rep (int array): cluster assignment for each data
			r_emb (K * emb_size): representative embedding
			emb_tensor (N * emb_size): fine-tuned input embedding
			cls_loss (float): clustering loss
		"""
		accuracies = []
		# Get data.
		row_header, rows = FileLoader(data_filename)
		target_rows, aux_rows, gold, spans = self.get_basics(row_header, rows)
		

		# Generate batches.
		row_iter, aux_c_sizes, aux_weights = DataLoader(target_rows, aux_rows, 
			self.batch_size, self.w2v_model)

		_, pairs = FileLoader(pair_filename)
		target_pairs = self.get_pairs(pairs)
		pair_iter = PairLoader(target_pairs, self.batch_size, self.w2v_model)

		# Get the initial embedding tensor as the average w2v embedding.
		emb_tensor = self.w2v_model.get_emb_matrix(target_rows)
		if do_eval is not None:
			with open(os.path.join(output_dir, "emb_before"), "wb") as handle:
				pickle.dump((emb_tensor, spans, gold), handle, 
					protocol=pickle.HIGHEST_PROTOCOL)

		# Run the base cluster module
		print("Run 1st module: clustering algorithm")
		c_emb, labels, cls_loss = self.cluster_module.run(emb_tensor)	
	
		# Run the embedding fine-tuning module
		print("Run 2nd module: refine embedding")
		enc, emb_loss, emb_labels = self.emb_module.run(aux_c_sizes, row_iter,
			LabelLoader(labels, self.batch_size), c_emb, spans, gold,
			output_dir, pair_iter=pair_iter, aux_weights=aux_weights)
		print("****cluster loss: %f; emb loss:%f****" % (cls_loss, emb_loss))
		
		# Update embedding tensor
		emb_tensor = enc.data

		if do_eval is not None:
			accuracies.append(Evaluate(labels, gold, do_eval))	
		
		print("Run 3rd module: refinement by clustering algorithm")
		# Final refinement
		c_emb, labels, cls_loss = self.cluster_module.run(emb_tensor)		
		labels = self.post_processing(row_header, rows, labels)

		if do_eval is not None:
			accuracies.append(Evaluate(labels, gold, do_eval))	
			with open(os.path.join(output_dir, "emb_after"), "wb") as handle:
				pickle.dump((emb_tensor, spans, gold), handle, 
					protocol=pickle.HIGHEST_PROTOCOL)

		return row_header, rows, labels, c_emb, emb_tensor, cls_loss, accuracies
