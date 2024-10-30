# DCT-CryptoNets: Scaling Private Inference in the Frequency Domain

<p align="center">
  <img src="visuals/dct-cryptonets.svg" alt="DCT-CryptoNets" width="600"/>
  <p align="center">
    This repository contains code from <i><b>DCT-CryptoNets</b></i>. The paper can be accessed for free on <a href="https://www.arxiv.org/abs/2408.15231"><strong>Arxiv »</strong></a>
</p>

```bibtex
@article{roy2024arxiv,
  title   = {DCT-CryptoNets: Scaling Private Inference in the Frequency Domain}, 
  author  = {Arjun Roy and Kaushik Roy},
  journal = {arXiv},
  year    = {2024}
}
```

## Summary
**DCT-CryptoNets** is a novel Fully Homomorphic Encrypted (FHE) framework that leverages the Discrete Cosine Transform 
(DCT) to perform privacy-preserving neural network inference on encrypted images. By operating in the frequency domain, 
DCT-CryptoNets reduces the computational burden of homomorphic encryption, enabling significant latency improvements and
efficient scaling to larger networks and images. This is achieved by strategically focusing on perceptually relevant 
low-frequency components while discarding noise-inducing high-frequency elements, thereby minimizing the need for 
computationally expensive non-linear activation and homomorphic bootstrapping operations. DCT-CryptoNets offers an 
avenue for deploying secure and efficient deep learning models in privacy-sensitive applications, particularly those 
involving large-scale images.


## Installation
After cloning the repository, install environmental dependencies via Conda (Anaconda, Miniconda, Mamba etc.). The new 
conda environment will have the name `dct-cryptonets`.
```bash
conda env create -f env.yml
```


## Datasets
This study includes analyses on four datasets (`cifar10`, `miniImageNet`, `ImageNette`, and `ImageNet`). `cifar10` is installed via `torchvision` within the training and evaluation scripts. To download `miniImageNet`, `ImageNette`, or `ImageNet` run `install_datasets.sh` as follows
```bash
bash install_datasets.sh --help
Usage: install_datasets.sh -a parameterA -b parameterB -c parameterC -d parameterD
        -a Download and install ImageNette?     type:(Y/N)
        -b Download and install mini-ImageNet?  type:(Y/N)
        -c Download and install ImageNet?       type:(Y/N)
        -d Directory path for datasets          type:PATH

# Example for downloading only Imagenette
bash install_datasets.sh -a Y -b N -c N -d /path/to/download/dataset
```


## Run Experiments
Both training (`run_train.sh`) and homomorphic evaluation (`run_homomorphic_eval.sh`) scripts have user arguments 
embedded. Please update these with your I/O information, model selection, and model hyperparameters. You can run these 
scripts in the background with `nohup` or `tmux` depending on your preference.

### Training
```bash
nohup bash run_train.sh > /path/to/log/output/ &
```

### Homomorphic Evaluation
```bash
nohup bash run_homomorphic_eval.sh > /path/to/log/output/ &
```

## Comparative Benchmarking
### Fully Homomorphic Encrypted Neural Networks (FHENN's) for Image Classification
|          Method          | Dataset  |  Input Dimensions   |   Model    | Homomorphic Scheme | Accuracy  | Latency (s)  | Normalized Latency (s)<br/> (96-threads) |
|:------------------------:|:--------:|:-------------------:|:----------:|:------------------:|:---------:|:------------:|:-------------------------------:|
| Hesamifard et al. (2017) | CIFAR-10 |  3 x 32<sup>2</sup>  | Custom CNN |        BGV         |   91.4%   |    11,686    |                    ~            |
|    Chou et al. (2018)    | CIFAR-10 |  3 x 32<sup>2</sup>  | Custom CNN |       FV-RNS       |   75.9%   |    3,240     |                    ~            |
| SHE (Lou & Jiang, 2019)  | CIFAR-10 |  3 x 32<sup>2</sup>  | Custom CNN |        TFHE        |   92.5%   |    2,258     |                   470           |
|    Lee et al. (2022b)    | CIFAR-10 |  3 x 32<sup>2</sup>  | ResNet-20  |        CKKS        |   92.4%   |    10,602    |                    ~            |
|    Lee et al. (2022a)    | CIFAR-10 |  3 x 32<sup>2</sup>  | ResNet-20  |        CKKS        |   91.3%   |    2,271     |                    ~            |
|    Kim & Guyot (2023)    | CIFAR-10 |  3 x 32<sup>2</sup>  |  Plain-20  |        CKKS        |   92.1%   |     368      |                    ~            |
|    Ran et al. (2023)     | CIFAR-10 |  3 x 32<sup>2</sup>  | ResNet-20  |        CKKS        |   90.2%   |     392      |                    ~            |
| Rovida & Leporati (2024) | CIFAR-10 |  3 x 32<sup>2</sup>  | ResNet-20  |        CKKS        |   91.7%   |     336      |                    ~            |
|  Benamira et al. (2023)  | CIFAR-10 |  3 x 32<sup>2</sup>  |   VGG-9    |        TFHE        |   74.0%   |     570      |                    48           |
|   Stoian et al. (2023)   | CIFAR-10 |  3 x 32<sup>2</sup>  |   VGG-9    |        TFHE        |   87.5%   |    18,000    |                  3,000*         |
|    **DCT-CryptoNets**    | **CIFAR-10** |  **3 x 32<sup>2</sup>**  | **ResNet-20** |      **TFHE**      | **91.6%** |  **1,339**   |                **1,339**        |
|    **DCT-CryptoNets**    | **CIFAR-10** | **24 x 16<sup>2</sup>** | **ResNet-20** |      **TFHE**      | **90.5%** |   **565**    |                 **565**         |
|                          |          |                     |            |                    |           |              |                                 |
| SHE (Lou & Jiang, 2019)  | CIFAR-10 | 3 x 32<sup>2</sup>  | ResNet-18  |        TFHE        |   94.6%   |    12,041    |                  2,509          |
|    **DCT-CryptoNets**    | **CIFAR-10** |  **3 x 32<sup>2</sup>**  | **ResNet-18** |      **TFHE**      | **92.3%** |  **1,746**   |                **1,746**        |
|    **DCT-CryptoNets**    | **CIFAR-10** |  **3 x 32<sup>2</sup>**  | **ResNet-18** |      **TFHE**     | **91.2%** |  **1,004**   |                **1,004**        |
|                          |          |                     |            |                    |           |              |                                 |
| SHE (Lou & Jiang, 2019)  | ImageNet | 3 x 224<sup>2</sup> | ResNet-18 |        TFHE        |   92.5%   |   216,000    |                  45,000         |
|   **DCT-CryptoNets**     | **ImageNet** | **3 x 224<sup>2</sup>** | **ResNet-18** |        **TFHE**        | **91.6%** |  **16,115**  |                **16,115**       |
|    **DCT-CryptoNets**    | **ImageNet** | **64 x 56<sup>2</sup>** | **ResNet-18** |        **TFHE**        |   **90.5%**   |  **8,562**   |                **8,562**        |


## Library Considerations
`DCT-CryptoNets` was developed prior to `v1.6.0` of `Concrete-ML` which introduced a nice feature of _approximate 
rounding_ of model accumulators. This can improve latency with trade-off's in accuracy depending on your use case and
could potentially achieve even faster latency than what is reported in this paper! By simply replacing the line below you can implement 
_approximate rounding_. See [*CIFAR10 Use Case](https://github.com/zama-ai/concrete-ml/tree/main/use_case_examples/cifar/cifar_brevitas_training) and [ResNet Use Case](https://github.com/zama-ai/concrete-ml/tree/main/use_case_examples/resnet) in the `Concrete-ML` repository for 
further examples of _approximate rounding_.
```python
from concrete.ml.torch.compile import compile_brevitas_qat_model

compile_brevitas_qat_model(
    # {other arguments} ...
    
    rounding_threshold_bits=params.rounding_threshold_bits,
    # --- OR ---
    rounding_threshold_bits={
        "n_bits": params.rounding_threshold_bits, 
        "method": "approximate",
    }
)
```


## Acknowledgement
*This work was supported in part from the Purdue Center for Secure Microelectronics Ecosystem – CSME#210205.*

Parts of this code were built upon [DCTNet](https://github.com/kaix90/DCTNet), [PT-MAP-sf](https://github.com/xiangyu8/PT-MAP-sf), and [Concrete-ML](https://github.com/zama-ai/concrete-ml).

We would also like to thank the Zama Concrete-ML team and the community on [FHE Discord](https://fhe.org/community.html) for their support and 
interesting discussions!
