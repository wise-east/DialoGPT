# less train_raw.tsv| awk -F '\t' '{print "0.0 "$1"\t1.0 "$2}'> train.tsv

less yesands_train_raw.tsv| awk -F '\t' '{print "0.0 "$1"\t1.0 "$2}'> yesands_train.tsv
less yesands_valid_raw.tsv| awk -F '\t' '{print "0.0 "$1"\t1.0 "$2}'> yesands_valid.tsv
