

python -u /Myhome/slf/work/data-to-text/model/run_transformer.py \
--do_eval \
--data_dir="../data" \
--model_name_or_path="bert-base-uncased"\
2>&1 | tee train.log