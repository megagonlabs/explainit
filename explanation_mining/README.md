# Labeled data for explanation between opinions
## Each row contains the following information (separated by tab):
<Explanation Label, Sentence Label, Sentence, Opinion A, Opinion B, ID>
- Explanation Label: whether Opinion B explains Opinion A;
- Sentence Label: whether the sentence contain explanations;
- Sentence: review sentence;
- Opinion A: an extracted opinion from the sentence;
- Opinion B: another extracted opinion;
- ID: row id.

# Explanation Classification model
## Download Trained Model

```
$ ./download.sh
```

## Evaluation

### To view usage
```
$ python src/lstm_triple_attn_bert.py --help 
```
### Use the following command to run evaluation code with default settings
```
$ python src/lstm_triple_attn_bert.py --data_dir data/ --model_dir weights/ --do_eval
```