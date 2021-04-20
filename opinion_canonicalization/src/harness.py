import commentjson
import json
import os
import sys

from exreadcluster import ExreadCluster
from load_data import FileLoader, FileWriter

def process_one_file(method, config, input_path, edge_path, output_dir, k, file_name):
	# Create output directory
	try:
		os.mkdir(output_dir)
	except:
		pass

	do_eval = None if len(config["eval"]) < 1 else config["eval"]
	if not os.path.isfile(edge_path):
		print("Regular loss (no edges).")
		row_header, rows, labels, _, _, _, accuracies = method.run(input_path, do_eval=do_eval, 
			output_dir=output_dir)
	else:
		print("Include edge loss.")
		row_header, rows, labels, _, _, _, accuracies = method.run_with_edge(input_path, edge_path, 
			do_eval=do_eval, output_dir=output_dir)

	if do_eval is not None:
		# For experiment only
		stats_name = "%s_%s_stats_%d.json" % (file_name, config["cluster"]["method"], k)
		stats_path = os.path.join(output_dir, stats_name)
		with open(stats_path, "w") as file:
			file.write(json.dumps(accuracies, indent=4))
	
	result_name = "%s_result_%d.json" % (config["cluster"]["method"], k)
	result_path = os.path.join(output_dir, result_name)
	print(result_path)
	FileWriter(result_path, row_header, rows, labels)

if __name__ == "__main__":
	harness_config = sys.argv[1]
	harness_config = commentjson.load(open(harness_config, "r"))
	alg_config = commentjson.load(open(harness_config["config"], "r"))
	input_dir = harness_config["input_dir"]
	output_dir = harness_config["output_dir"]
	file_names = harness_config["file_names"]
	num_runs = int(alg_config["num_runs"]) if "num_runs" in alg_config else 1

	method = ExreadCluster(alg_config["target_cols"], alg_config["auxiliary_cols"], 
			alg_config["edges"], alg_config["cluster"], alg_config["embedding"], 
			alg_config["max_attemp"])

	for file_name in file_names:
		#file_name="1cc50392-5982-4db0-94fe-903a8b91dba2"
		for i in range(num_runs):
			process_one_file(method, alg_config, os.path.join(input_dir, file_name),os.path.join(input_dir, file_name + "_edges"),os.path.join(output_dir, file_name), i, file_name)

	# methods = ["kmeans", "gmm", "kmeans_vqvae", "greedy"]
	# n_iters = [1000, 1000, 100, 100]

	# for i in range(len(methods)):
	# 	alg_config["cluster"]["method"] = methods[i]
	# 	alg_config["cluster"]["max_iter"] = n_iters[i]

	# 	method = ExreadCluster(alg_config["target_cols"], alg_config["auxiliary_cols"], 
	# 		alg_config["edges"], alg_config["cluster"], alg_config["embedding"], 
	# 		alg_config["n_iter"], alg_config["max_attemp"])

	# 	for file_name in file_names:
	# 		process_one_file(method, alg_config, os.path.join(input_dir, file_name),
	# 			os.path.join(input_dir, file_name + "_edges"),
	# 			os.path.join(output_dir, file_name))
	# 	break
