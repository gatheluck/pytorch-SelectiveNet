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

### Testing
Use `scripts/test.py` to test the network. Example usage:
```bash
# Example usage (test single weight)
cd scripts
python test.py --dataset cifar10 --weight ${path_to_saved_weight} --coverage 0.7

# Example usage (test multiple weights)
cd scripts
python experiments/test_multi.py -t ${path_to_root_dir_of_saved_weights} -d cifar10
```

### Plot Results
Use `scripts/plot.py` to plot the result. Example usage:
```bash
# Example usage (plot test result)
cd scripts
python plot.py -t ${path_to_test.csv} -x coverage --plot_test

# Example usage (plot all training logs)
cd scripts
python experiments/plot_multi.py -t ${path_to_test.csv} -x step --plot_all
```

## References

- [Yonatan Geifman and Ran El-Yaniv. "SelectiveNet: A Deep Neural Network with an Integrated Reject Option.", in ICML, 2019.][1]
   
- [Original implementation in Keras][2]

[1]: https://arxiv.org/abs/1901.09192
[2]: https://github.com/geifmany/selectivenet