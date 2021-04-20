"""
Utility functions.
"""
import csv
import json
import os

class Util:
	@staticmethod
	def read_input(file_name):
		pass

class Preprocess:
	"""Note: functions in this class should be excluded from code release. 
	"""
	@staticmethod
	def process_groundtruth(key, input_dir, output_dir, groundtruth_dir):
		"""This function prepares the input file from raw labels.
		Parameters:
			-key: ty_id
			-input_dir: input extractions directory
			-output_dir: output directory
			-groundtruth_dir: raw groundtruth files, one .csv file per asepct
		"""
		ground_truth = {}
		cluster_id = 0
		cluster_count = {}

		groundtruth_path = os.path.join(groundtruth_dir, key)
		for aspect_fname in os.listdir(groundtruth_path):
			if aspect_fname == ".DS_Store":
				continue
			with open(os.path.join(groundtruth_path, aspect_fname), "r") as file:
				csv_reader = csv.reader(file, delimiter=',')
				for row in csv_reader:
					if row[0] == 'ignored':
						cur_id = 0
					else:
						cur_id = cluster_id+1
						cluster_id += 1
					if cur_id not in cluster_count:
						cluster_count[cur_id] = 0
					for element in row[1:]:
						ground_truth[element] = cur_id
						cluster_count[cur_id] += 1
		#print(cluster_count)
		input_rows = []
		header = None
		num_notfound = 0
		pruned = 0
		input_path = os.path.join(input_dir, key)
		with open(input_path, "r") as file:
			csv_reader = csv.reader(file, delimiter=',')
			for row in csv_reader:
				if header is None:
					header = row
				else:
					content = [r.lower() for r in row]
					row_key = row[0] + " " + row[1]
					row_key = row_key.lower()
					if row_key not in ground_truth:
						print(row_key, ground_truth)
						raise
					cur_id = ground_truth[row_key] if row_key in ground_truth else 0
					if cluster_count[cur_id] < 4:
						pruned += 1
						continue
					content.append(str(cur_id))
					input_rows.append(content)
					if row_key not in ground_truth:
						num_notfound += 1

		output_path = os.path.join(output_dir, key)
		header.append("gold")
		with open(output_path, "w") as file:
			file_writer = csv.writer(file, delimiter=',')
			file_writer.writerow(header)
			for row in input_rows:
				file_writer.writerow(row)

	@staticmethod
	def prepare_input(path, data_path, output_dir):
		ids = Preprocess.get_ids(path)
		id2extrs = {ids[i]:[] for i in range(len(ids))}
		id2edges = {ids[i]:[] for i in range(len(ids))}
		with open(data_path, "r") as file:
			for line in file:
				review = json.loads(line)
				if review["ty_id"] not in id2extrs:
					continue
				for extr in review["extractions"]:
					id2extrs[review["ty_id"]].append([extr["opinion"], extr["aspect"], extr["attribute"], extr["sentiment"], extr["sid"]])
				for i in range(len(review["extractions"])):
					for j in range(len(review["extractions"])):
						if i == j:
							continue
						if review["extractions"][i]["attribute"] != review["extractions"][j]["attribute"]:
							continue
						if review["extractions"][i]["aspect"] == review["extractions"][j]["aspect"]:
							continue
						id2edges[review["ty_id"]].append([
							review["extractions"][i]["opinion"],
							review["extractions"][i]["aspect"], 
							review["extractions"][j]["opinion"], 
							review["extractions"][j]["aspect"]])	
		header = ("modifier", "aspect", "attribute", "sentiment", "review_id")
		for tyid, extrs in id2extrs.items():
			break
			file_name = os.path.join(output_dir, tyid)
			csvwriter = csv.writer(open(file_name, "w"))
			csvwriter.writerow(header)
			for extr in extrs:
				csvwriter.writerow(extr)
		header = ("modifier", "aspect", "modifier", "aspect")
		for tyid, pairs in id2edges.items():
			file_name = os.path.join(output_dir, tyid + "_edges")
			csvwriter = csv.writer(open(file_name, "w"))
			csvwriter.writerow(header)
			for pair in pairs:
				csvwriter.writerow(pair)

	@staticmethod
	def get_ids(path):
		ids = []
		for folder in os.listdir(path):
			if "DS_Store" not in folder:
				ids.append(folder)
		return ids 
			