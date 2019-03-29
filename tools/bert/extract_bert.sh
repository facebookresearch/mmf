N_REM=`expr $3 - 1`

for i in $(seq 0 $N_REM); do
    python tools/extract_bert_embeddings.py --imdb_path $1 --out_path $2 --group_id $i --n_groups $3 &
done
