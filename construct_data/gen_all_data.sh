mkdir -p ../data/test
mkdir -p ../data/alpaca

for LANG in en de ru tr vi zh
do
  python3 construct_data/make_few_shot_examples.py --dataset xquad --target ${LANG} --typename fewshot
done

