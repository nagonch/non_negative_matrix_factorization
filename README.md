# Robustness of Non-Negative Matrix Factorization

This is an impementation of three papers dealing with label noise:
-  Lee, Seung, Learning the parts of objects by non-negative matrix factorization (`nmf.py/LeeSeungNMF`)
-  Zhang, Lijun, et al. "Robust non-negative matrix factorization" (`nmf.py/RobustNMF`)
- Bin Shen et al. “Robust nonnegative matrix factorization via L 1 norm regularization by multiplicative updating rules” (`nmf.py/RobustL1NMF`)

The repository was used to perform the experiments described [here](https://github.com/nagonch/non_negative_matrix_factorization/blob/main/nmf_robustness.pdf)

## Running the code

Install dependencies: `pip install -r requirements.txt`

To run the estimation of the metrics: `python experiments_quantity.py --downsample 5`

To get the base faces: `python experiments_quantity.py --downsample 5`

Change the "downsample" parameter according to your desire (speeds up the computation)
