```
python db2cfiles.py --train ./dataset/split_train.db --val ./dataset/split_val.db --test ./dataset/split_test.db

```


```shell
bash ../code-slicer/joern/build.sh
```

```shell
cd ../code-slicer
bash ./parse_all.sh ../data/aijon
```


```
cd cd models/Devign_ReVeal/code
python data_processing/extract_slices.py aijon
```

```
python Vuld_SySe/word2vec_train.py \
  --data_paths data/aijon/full_data_with_slices.json \
  --save_model_dir data/aijon/wv_models \
  --model_name raw_code.100
```


```
python data_processing/create_ggnn_data.py aijon
```

```
python data_processing/full_data_prep_script.py aijon

```

```
python - <<'PY'
import sys
sys.argv = ['full_data_prep_script.py', 'aijon']
from data_processing import full_data_prep_script as prep
prep.extract_line_graph_data('aijon')
PY

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
python scripts/generate_splits_from_lists.py \
  --dataset data/aijon/full_experiment_real_data_processed/aijon-line-ggnn.jsonlines \
  --split-dir data/aijon \
  --output data/aijon/full_experiment_real_data_processed

```
-> creates splits.csv
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