# pytorch-SelectiveNet

This is an unofficial pytorch implementation of a paper, SelectiveNet: A Deep Neural Network with an Integrated Reject Option [Geifman+, ICML2019].
I'm really grateful to the [original implementation](https://github.com/anonygit32/SelectiveNet) in Keras by the authors, which is very useful.

## Requirements

You will need the following to run the codes:
- Python 3.6+
- Pytorch 1.2+
- TorchVision

Note that I run the code with Ubuntu 18, Pytorch 1.2.0, CUDA 10.1

### Training
Use `scripts/train.py` to train the network. Example usage:
```bash
# Example usage
cd scripts
python train.py --dataset cifar10 --log_dir ../logs/train --coverage 0.7 
```

We also provide cript generator for training by [ABCI](https://abci.ai/) which is the world's first large-scale open AI computing infrastructure.
Use `scripts/experiments/train_abci.py` to generate shell scripts for ABCI. 
Example usage:
```bash
# Example usage
cd scripts
python experiments/train_abci.py -d cifar10 -l ../logs/train --script_root ../logs/abci_script --run_dir . --abci_log_dir ../logs/abci_log --user ${your_abci_user_id} --env ${abci_conda_environment} --coverage 0.7
```

### Testing
Use `scripts/test.py` to test the network. Example usage:
```bash
# test single weight
cd scripts
python test.py --dataset cifar10 --weight ${path_to_saved_weight} --coverage 0.7

# test multiple weights
cd scripts
python experiments/test_multi.py -t ${path_to_root_dir_of_saved_weights} -d cifar10

# test multiple weights (including adversarial robustness)
cd scripts
python experiments/test_multi_adv.py -t ${path_to_root_dir_of_saved_weights} -d cifar10 --attack pgd --attack_norm linf --attack_trg_loss both
```

### Plot Results
Use `scripts/plot.py` to plot the result. Example usage:
```bash
# plot test result. (plot simple)
cd scripts
python plot.py -t ${path_to_test.csv} -x coverage --plot_test

# plot test result. (plot detail results including adversarial robustness)
cd scripts
python plot.py  -t ${path_to_test.csv} -x eps --plot_test_adv --coverage 0.70 --at pgd --at_eps 16 --at_norm linf --attack pgd --attack_norm linf --attack_trg_loss both

# Example usage (plot all training logs)
cd scripts
python experiments/plot_multi.py -t ${path_to_test.csv} -x step --plot_all
```

## References

- [Yonatan Geifman and Ran El-Yaniv. "SelectiveNet: A Deep Neural Network with an Integrated Reject Option.", in ICML, 2019.][1]
   
- [Original implementation in Keras][2]

[1]: https://arxiv.org/abs/1901.09192
[2]: https://github.com/geifmany/selectivenet