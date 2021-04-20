# Labeled opinion clusters
## Attribute names shown in the first row.
For example,

"modifier,aspect,attribute,sentiment,review_id,gold"

"modifier" and "aspect" are the opinion phrases and aspect phrases;

"attribute" and "sentiment" are the aspect, ploarity labels;

"review_id" is the meta information that is not used by the algorithm;

"gold" is the reference cluster id.

To run the algorithm on unlabeled data, simply remove the last column. 

# Code runing instructions
## Embedding fine-tuning configurations
Follow configuration example in "configs/config.json" to config the algorithm.
Follow "configs/config_unlabeled.json" to config the algorithm when reference cluster labels are not available.

## Experiment configuration
Follow "configs/harness.json" to config the experiment: specify the embedding fine-tuning config file, add input/output directory, and filenames for the experiment.

## Run the algorithm
Once the experiment is properly configed, use the following code to run experiment:
```
$ python src/harness.py  configs/harness.json 
```

Note that the current data does not include the predicted edges, thus intra-cluster loss is not used. To use intra-cluster loss, you can prepare edges between opinions in the following format:

modifier,aspect,modifier,aspect"

Name the file as \"<file\_name>\_edges", e.g., "1f92697c-da15-420b-a079-9a1c7c11fcce\_edges", and put it in the input directory, specified in "configs/harness.json"

