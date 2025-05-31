# EEG-TEA: A Novel Application of Test-time Energy Adaptation in Electroencephalogram Decoding

## Introduction 

This is the official repository for the paper "EEG-TEA: A Novel Application of Test-time Energy Adaptation in Electroencephalogram Decoding". 
In this work, we apply Test-time Energy Adaptation (TEA) [1], a method developed for image data, to the time-series domain,
specifically with electroencephalogram (EEG) data.
EEG-based Deep Learning models often struggle to generalize across subjects due to the high inter-subject variability,
which motivates the need for test-time adaptation (TTA) methods to improve model robustness and performance 
in real world setting.

We conduct our experiments on the task of motor imagery decoding using datasets 2a and 2b from the BCI Competition IV [2].
We have conducted 2 experiments on cross-subjection adaptation and corruption adaptation, to test out the method's effectiveness
on different types of distribution shifts.

In the cross-subject adaptation setting, we train a model on dataset 2a, where the source model is trained on all subjects
excluding the target subject, and then adapt the model to the target subject using TEA. We have tested out two different 
variants of this setting, one where we left out one subject from the training set, and the other where we left out 6, 
to simulate a larger distribution shift. 

In the corruption adaptation setting, we train a model on dataset 2b, where the source model is trained on one subject and 
adapt the model to the artificially corrupted data of the same subject. The corruption is applied by adding random 
Gaussian noise of varying degree to the EEG signals.

We show that TEA can be applied to EEG data and that it is able to improve the performance of the model in both settings. 
#### 

## Related work 

![Overview of tea [1]](tea_overview.png)
*Figure 1: Overview of TEA [1]*


Test-time adaptation (TTA) is a paradigm that eliminates this dependence on the source domain by adapting models on the fly,
with unlabeled data. Recent work [3] applied entropy minimization (EM) to EEG decoding, improving cross-subject performance 
by encouraging confident predictions during inference. TEA [1] is a TTA method which interpret the classifier 
as an energy-based model and aims to reduce the impact of distribution shifts during model deployment by lowering the 
energy of samples from the test distribution (see Figure 1).
We adopt EM as a baseline, as both EM and TEA are unsupervised test-time adaptation methods designed to reduce predictive uncertainty. 
TEA optimizes a more expressive energy-based objective and should offer better stability and adaptability 
under complex distribution shifts than EM.

## Weakness/strength 

## Novel contribution 

Our novel contribution includes the application of TEA, which was originally developed for images, to EEG data. 
Furthermore, we demonstrate that initialising the noise for SGLD sampling, with pink noise, rather than white (Gaussian)
noise, results in the generation of more realistic samples, leading to improved model performance.

## Results 

### Installation
Run the following commands:
  ```bash
  cd eeg
  pip install .
````
### Running EEG-TEA
Run [EEG_TEA.ipynb](eeg/EEG_TEA.ipynb)

Set the parameters stated below for each experiment to reproduce the results.

#### Experiment 1: Cross-subject adaptation


```python
dataset_name = '2a'
dataset_setup = 'loso' 
corruption_level = None
```

#### Experiment 2: Corruption adaptation

```python
dataset_name = '2b'
dataset_setup = 'within' 
corruption_level = 1  # in [1, 2, 3, 4, 5]
```
| Severity | 1        | 2       | 3         | 4       | 5          | Avg. (Std.)     |
|----------|----------|---------|-----------|---------|------------|-----------------|
| Source   | 0.856 (0.000) | 0.851 (0.006) | 0.655 (0.008) | 0.573 (0.002) | 0.529 (0.004) | 0.693 (0.004)   |
| Entropy  | 0.867 (0.000) | 0.863 (0.003) | 0.663 (0.007) | 0.579 (0.008) | 0.533 (0.007) | 0.701 (0.005)   |
| TEA      | **0.877 (0.001)** | **0.871 (0.001)** | **0.685 (0.008)** | **0.586 (0.007)** | **0.537 (0.005)** | **0.710 (0.004)** |


# Conclusion

# Contributions

### Leonard Horns 
- Ran experiment for training on 3 subjects
- Integrated TEA into EEG decoding adaptation pipeline
### Matei Nastase 
- PCA analysis
- Ran experiment for training on 1 subject
### Benjamin Hucko 
- Hyperparameter search 
- Set up results logging and plots
- Set up environment
### Tyme Chatupanyachotikul
- Dataset
- Ran experiment on corruption

## Citation
[1] Yige Yuan, Bingbing Xu, Liang Hou, Fei Sun, Huawei Shen, and Xueqi Cheng. Tea: Test-time energy adaptation, 2024. URL https://arxiv.org/abs/2311.14402.

[2] Sion An, Myeongkyun Kang, Soopil Kim, Philip Chikontwe, Li Shen, and Sang Hyun Park. Bci competition iv-2b, dec 2024. URL https://service.tib.eu/ldmservice/dataset/bci-competition-iv-2b

[3] Martin Wimpff, Mario Döbler, and Bin Yang. Calibration-free online test-time adaptation for electroencephalography motor imagery decoding. In 2024 12th International Winter Conference on Brain-Computer Interface (BCI), pages 1–6. IEEE, 2024.