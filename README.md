# Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing

In Byzantine robust distributed or federated learning, a central server wants to train a machine learning model over data distributed across multiple workers. However, a fraction of these workers may deviate from the prescribed algorithm and send arbitrary messages. While this problem has received significant attention recently, most current defenses assume that the workers have identical data. For realistic cases when the data across workers are heterogeneous (`noniid`), we design new attacks which circumvent current defenses, leading to significant loss of performance. We then propose a simple bucketing scheme that adapts existing robust algorithms to heterogeneous datasets at a negligible computational cost. We also theoretically and experimentally validate our approach, showing that combining bucketing with existing robust algorithms is effective against challenging attacks. Our work is the first to establish guaranteed convergence for the non-iid Byzantine robust problem under realistic assumptions.

# Table of contents

- [Structure of code](#Code-organization)
- [Reproduction](#Reproduction)
- [License](#license)
- [Reference](#Reference)

# Code organization

The structure of the repository is as follows:
- `codes/`
  - Source code.
- `outputs/`
  - Store the output of the launcher scripts.
- `exp{}.py`: The launcher script for experiments.
# Reproduction

To reproduce the results in the paper, do the following steps

1. Add `codes/` to environment variable `PYTHONPATH`
2. Install the dependencies: `pip install -r requirements.txt`
3. Run `bash run.sh` and select option 2 to 9 to generate the code.
4. The output will be saved to the corresponding folders under `outputs`

Note that if the GPU memory is small (e.g. less than 16 GB), then running the previous commands may raise insufficient exception. In this case, one can decrease the level parallelism in the script by changing the order of loops and reduce the number of parallel processes. 


# License

This repo is covered under [The MIT License](LICENSE).


# Reference
If you use this code, please cite the following paper

```
@misc{karimireddy2021byzantinerobust,
      title={Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing}, 
      author={Sai Praneeth Karimireddy and Lie He and Martin Jaggi},
      year={2021},
      eprint={2006.09365},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```