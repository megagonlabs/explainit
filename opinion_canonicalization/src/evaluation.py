"""
This file include different evaluation metrics for the clustering algoirthm.
"""
import os
import sklearn.metrics
import torch
import warnings

def Removal(predict, gold):
	predict_slim, gold_slim = [], []
	for i in range(len(gold)):
		if gold[i] == 0:
			continue
		predict_slim.append(predict[i])
		gold_slim.append(gold[i])
	return predict_slim, gold_slim

# TODO: debug only, to be removed.
def CrossAttributeCluster(row_header, rows, predict, gold):
	cluster2id, gold_cluster2ids = {}, {}
	for i in range(len(predict)):
		if predict[i] not in cluster2id:
			cluster2id[predict[i]] = []
		cluster2id[predict[i]].append(i)
		if gold[i] not in gold_cluster2ids:
			gold_cluster2ids[gold[i]] = []
		gold_cluster2ids[gold[i]].append(i)
	count = 0
	for _, members in cluster2id.items():
		attrs = []
		for i in members:
			attrs.append(rows[i][row_header["attribute"]])
		if len(set(attrs)) > 1:
			count += 1
	print(count)
	sizes = [len(m) for _, m in gold_cluster2ids.items()]
	print(sum(sizes)/(len(sizes)+0.0), max(sizes), min(sizes))

def Evaluate(predict, gold, metrics):
	#print(gold, predict)
	predict = [int(i) for i in predict]
	gold = [int(i) for i in gold]
	predict, gold = Removal(predict, gold)
	result = {"num_item": len(predict), 
			  "num_cluster": len(set(predict)), 
			  "num_true_cluster":len(set(gold))}
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		result['adjusted-rand-index'] = sklearn.metrics.adjusted_rand_score(predict, gold)
		result['mutual-information'] = sklearn.metrics.mutual_info_score(predict, gold)
		result['adjusted-mutual-information'] = sklearn.metrics.adjusted_mutual_info_score(predict, gold)
		result['normalized-mutual-information'] = sklearn.metrics.normalized_mutual_info_score(predict, gold)
		h,c,v = sklearn.metrics.homogeneity_completeness_v_measure(predict, gold)
		result['homogeneity'] = h
		result['completeness'] = c
		result['v-measure'] = v
		result['fowlkes-mallows'] = sklearn.metrics.fowlkes_mallows_score(predict, gold)
	
	for metric in metrics:
		print("\t", metric, ": ", result[metric])
	return result

def Plot(output_dir, emb_tensor, spans, labels, idx, method="TSNE"):
	if output_dir is None:
		return
	import matplotlib.pyplot as plt
	if emb_tensor.dim() > 2:
		emb_tensor = torch.norm(emb_tensor, dim=1)
	path = os.path.join(output_dir, "emb_%s.pdf" % idx)
	data2D = transform(emb_tensor.numpy(), method)
	x_axis = [v[0] for v in data2D]
	y_axis = [v[1] for v in data2D]
	fig, ax = plt.subplots()
	ax.scatter(x_axis, y_axis, s=20, c=labels)
	for i, txt in enumerate(spans):
		ax.annotate(txt, (x_axis[i], y_axis[i]), fontsize=5)
	fig.savefig(path)

def transform(data, method):
	if method == "TSNE":
		from sklearn.manifold import TSNE
		return TSNE(n_components=2).fit_transform(data)
	else:
		from sklearn.decomposition import PCA
		pca = PCA(n_components=2)
		pca.fit(data)
		return pca.transform(data)
