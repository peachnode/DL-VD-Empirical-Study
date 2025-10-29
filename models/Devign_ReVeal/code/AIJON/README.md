## 1. Creates raw C Files from Databases

Creates: a new directory data/labels_<total_rows>/ containing raw_code/*.c exports, split_train.txt/split_val.txt/split_test.txt, ggnn_input/cfg_full_text_files.json, and label_map.json
```
python db2cfiles.py --train ./dataset/split_train.db --val ./dataset/split_val.db --test ./dataset/split_test.db

```

## 2. Build Joern

```shell
bash ../code-slicer/joern/build.sh
```
## 3. Create Joern Parsed Files
Consumes: the .c files generated earlier in ../data/aijon/raw_code/

Creates: Joern parsing output under ../data/aijon/parsed/…, with per-file subdirectories (e.g. parsed/tmp/<file>/nodes.csv and edges.csv)

```shell
cd ../code-slicer
bash ./parse_all.sh ../data/aijon
```
## 4.
Consumes: every exported C file in data/aijon/raw_code/, reading each one’s contents to assemble example records.

Creates: data/aijon/full_data_with_slices.json, a JSON array of those records (label + code text).

```

python data_processing/extract_slices.py aijon
```


## 5. Create Word2Vec

Consumes: the paths given via --data_paths, here data/aijon/full_data_with_slices.json, tokenising the code for Word2Vec training.

Creates: the embedding model saved to data/aijon/wv_models/raw_code.100
```
python Vuld_SySe/word2vec_train.py \
  --data_paths data/aijon/full_data_with_slices.json \
  --save_model_dir data/aijon/wv_models \
  --model_name raw_code.100
```
## 6. Create GGNN ready dataset

Consumes: data/aijon/full_data_with_slices.json, the Joern graph files in data/aijon/parsed/tmp/<file>/{nodes,edges}.csv, and the trained Word2Vec model data/aijon/wv_models/raw_code.100.

Creates: a GGNN-ready dataset written to data/aijon/full_experiment_real_data/data.jsonlines (one JSON object per example with graph features)
```
python data_processing/create_ggnn_data.py aijon
```
## 7. 

Consumes: the consolidated GGNN JSONL file data/aijon/full_experiment_real_data/data.jsonlines.

Creates: data/aijon/full_experiment_real_data_processed/aijon-full_graph.jsonlines, copying each record’s full_graph field into the processed directory (creating it if absent).

```
python data_processing/full_data_prep_script.py aijon
```

## 8. 

Consumes: the raw code files, Joern CSVs (data/aijon/parsed/tmp/<file>/nodes.csv / edges.csv), and the Word2Vec embeddings at data/aijon/wv_models/raw_code.100.

Creates: line-level GGNN data in data/aijon/full_experiment_real_data_processed/aijon-line-ggnn.jsonlines.```
```
python - <<'PY'
import sys
sys.argv = ['full_data_prep_script.py', 'aijon']
from data_processing import full_data_prep_script as prep
prep.extract_line_graph_data('aijon')
PY
```

## 9. Generate Splits from list
```
LABEL_MAP = {
    "NOT_HELPFUL": 0,
    "SATURATED": 1,
    "UNREACHED": 2,
    "BUILD_ERROR": 3,
    "SUCCESS": 4,
    "WRONG_FORMAT": 5,
    "INSERT_ERROR": 6,
}
```


Consumes: the aijon-line-ggnn.jsonlines dataset plus split_train.txt, split_val.txt, and split_test.txt produced by db2cfiles.py.

Creates: per-fold splits.csv files (e.g. data/aijon/full_experiment_real_data_processed/v1/splits.csv), mapping dataset indices to split labels.

```
python scripts/generate_splits_from_lists.py \
  --dataset data/aijon/full_experiment_real_data_processed/aijon-line-ggnn.jsonlines \
  --split-dir data/aijon \
  --output data/aijon/full_experiment_real_data_processed

```
## 10. Devign
Consumes: data/aijon/full_experiment_real_data_processed/aijon-full_graph.jsonlines, together with the generated splits.csv in data/aijon/full_experiment_real_data_processed/v1/
```
python Devign/main.py \
  --dataset aijon \
  --input_dir data/aijon/full_experiment_real_data_processed \
  --fold 1 \
  --seed 1 \
  --model_type ggnn \
  --node_tag node_features \
  --graph_tag graph \
  --label_tag targets \
  --batch_size 256 \
  --train \
  --eval_export \
  --save_after_ggnn \
  --vulnerable-labels 4 \
  --max_steps 6000 \
  --dev_every 128 \
  --max_patience 25

```

```
python -m Vuld_SySe.representation_learning.api_test \
  --input_dir data/aijon/full_experiment_real_data_processed \
  --fold 1 --seed 1 --features ggnn \
  --train --eval_export 


```
-> saves the learned metric-learning weights under models/aijon/reveal/v{fold}/{seed}/RepresentationLearningModel.bin
-> models/Devign_ReVeal/code/data/aijon/full_experiment_real_data_processed/v1/eval_export_reveal_1.csv


```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -m Vuld_SySe.representation_learning.tsne_generation_verum \
  --input-dir data/aijon/full_experiment_real_data_processed \
  --fold 1 --seed 1 --split test \
  --output-dir tsnes/aijon --vulnerable-labels 1

```