
# MultiNetRelational Verification

This repository contains code for running relational verification experiments across multiple neural networks on image classification datasets like CIFAR-10.

## Setup

First, install the required dependencies using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate <your-env-name>
````

Replace `<your-env-name>` with the environment name specified in the `.yml` file or one of your choosing.

## Running the Code

To run the verification experiments, execute the following Python script:

```bash
python multiNetRelational.py
```

### Configurable Hyperparameters

The script allows you to configure the following hyperparameters:

* `dataset`: The dataset to run verification on. Supported option in the script is:

  * `Dataset.CIFAR10`

* `net_names`: A list of networks to load and verify. Example configuration:

  ```python
  net_names = [config.CIFAR_03, config.CIFAR_04, config.CIFAR_05]
  ```

* `prop_count`: The number of distinct properties to verify. Default is `20`.

* `count_per_prop`: Number of inputs per property. Default is `1`.

* `eps`: The ε (epsilon) value that defines the allowable perturbation radius. Default is `5.0/255`.

You can modify these hyperparameters directly in the `__main__` section of `multiNetRelational.py`.

## Example

```python
if __name__ == "__main__":
    dataset = Dataset.CIFAR10
    net_names = [config.CIFAR_03, config.CIFAR_04, config.CIFAR_05]
    nets = get_net(net_names=net_names, dataset=dataset)

    prop_count = 20
    count_per_prop = 1
    eps = 5.0 / 255
```

Adjust these parameters as needed for your experiments.

