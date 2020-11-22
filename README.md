# Comparing the Value of Labeled and Unlabeled Data in Method-of-Moments Latent Variable Estimation

## Setup

From the root directory, run the following:

Create virtual environent: `python3 -m venv .venv`

Activate environment: `source .venv/bin/activate`

Install requirements: `pip install -r requirements.txt`

## Repository Organization

The `scripts` directory contains things that take a while to run, and save files in `results`. The `notebooks` directory contains notebooks, which usually involve pulling results files from `results` and displaying them.

## Experiments

### Generalization Error

In the generalization error experiment we measure excess generalization error vs. number of points for different models. Our results verify the additional bias due to misspecification associated with learning from unlabeled data, and how a corrected model mitigates this bias. To produce the results for the generalization error notebook, run the following command.

```
python -m scripts.run_generalization_error_experiments
```

### Data Value Ratio

In the data value ratio experiment we measure how the data value ratio changes as the amount of misspecification increases. We observe that the ratio increases with misspecification, that is, labeled data becomes more valuable relative to unlabeled data when more misspecification is present. To produce the results for the data value ratio notebook, run the following command.

```
for d in 0 1 2 4
do
    python -m scripts.run_data_value_ratio_experiments --d=$d --save_path=results/data_value_ratio_results_d=$d;
done
```

### Combined

In the combined experiment we measure the performance of an estimator which combines labeled and unlabeled estimators. We observe that such a combination can outperform learning from either individually. To produce the results for the combined notebook, run the following command.

```
for d in 0 5
do
    for agg in mean median
    do
        python -m scripts.run_combined_experiments --d=$d --agg=$agg --save_path=results/combined_results_d="$d"_agg=$agg;
    done
done
```

## IMDB Real-World Case Study

In the real-world case study we explore how misspecification manifests in real-world datasets, and the difference between learning from labeled and unlabeled data in these settings. To produce the results for the IMDB notebook, run the following commands.

```
python -m scripts.generate_imdb_data;
python -m scripts.run_real_experiments --dataset=imdb;
python -m scripts.run_real_combined_experiments --dataset=imdb;
```

## References

[1] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
