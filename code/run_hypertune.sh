MODELNAME=linear-hier
export NUMEXPR_MAX_THREADS=10
for i in {1..50}
do
  python3 hypertune.py $MODELNAME
done