# EEG-TEA: A Novel Application of Test-time Energy Adaptation in Electroencephalogram Decoding

## Introduction 

This is the official repository for the paper "EEG-TEA: A Novel Application of Test-time Energy Adaptation in Electroencephalogram Decoding". 
In this work, we apply Test-time Energy Adaptation (TEA) [1], a method developed for image data, to the time-series domain,
specifically with electroencephalogram (EEG) data.
EEG-based Deep Learning models often struggle to generalize across subjects due to the high inter-subject variability,
which motivates the need for test-time adaptation (TTA) methods to improve model robustness and performance 
in real real-world setting.

We conduct our experiments on the task of motor imagery decoding using datasets 2a and 2b from the BCI Competition IV [2].
We have conducted 2 experiments on cross-subject adaptation and corruption adaptation, to test out the method's effectiveness
on different types of distribution shifts.

In the cross-subject adaptation setting, we train a model on dataset 2a, where the source model is trained on all subjects
excluding the target subject, and then adapt the model to the target subject using TEA. We have tested out two different 
variants of this setting, one where we left out one subject from the training set, and the other where we left out 6, 
to simulate a larger distribution shift. 

In the corruption adaptation setting, we train a model on dataset 2b, where the source model is trained on one subject and 
adapts the model to the artificially corrupted data of the same subject. The corruption is applied by adding random 
Gaussian noise of varying degrees to the EEG signals.

We show that TEA can be applied to EEG data and that it is able to improve the performance of the model in both settings. 
#### 

## Related work 

![Overview of tea [1]](tea_overview.png)
*Figure 1: Overview of TEA [1]*


Test-time adaptation (TTA) is a paradigm that eliminates this dependence on the source domain by adapting models on the fly,
with unlabeled data. Recent work [3] applied entropy minimization (EM) to EEG decoding, improving cross-subject performance 
by encouraging confident predictions during inference. TEA [1] is a TTA method that interprets the classifier 
as an energy-based model and aims to reduce the impact of distribution shifts during model deployment by lowering the 
energy of samples from the test distribution (see Figure 1).
We adopt EM as a baseline, as both EM and TEA are unsupervised test-time adaptation methods designed to reduce predictive uncertainty. 
TEA optimizes a more expressive energy-based objective and should offer better stability and adaptability 
under complex distribution shifts than EM.

## Weakness/strength 
The original TEA paper presents a simple yet effective unsupervised test-time adaptation method that requires no model modification or source data, making it broadly applicable and easy to implement. Its use of energy-based objectives is both principled and practical, showing strong results across multiple image classification benchmarks. However, the work is limited in scope, focusing solely on the vision domain and primarily on synthetic corruptions. To address this, we applied TEA to EEG data, a modality known for its high natural variability and relevance in neuroscience. This extension highlights TEA’s broader applicability and opens the door for its use in more complex, real-world domains.

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

### Experiment setup 
Set the following parameters in the notebook to reproduce the experiments.
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

# Conclusion
Test-time Energy Adaptation (TEA) can successfully be applied for EEG decoding, demonstrating strong performance across various distribution shifts, including subjects, sessions, and synthetic corruptions. Our method outperformed or matched existing approaches, showing its potential as a robust unsupervised adaptation technique for EEG data. While challenges remain, particularly in handling realistic distribution shifts and generating biologically plausible fake samples, our results suggest TEA is a promising direction for improving EEG robustness.

# Contributions

### Leonard Horns 
- Ran experiment for training on 3 subjects
- Integrated TEA into EEG decoding adaptation pipeline
### Matei Nastase 
- PCA analysis
- Ran experiment for training on 1 subject and baselines
### Benjamin Hucko 
- Hyperparameter search 
- Set up results logging and plots
- Set up environment
### Tyme Chatupanyachotikul
- Dataset
- Ran experiments on corruption

## Citation
[1] Yige Yuan, Bingbing Xu, Liang Hou, Fei Sun, Huawei Shen, and Xueqi Cheng. Tea: Test-time energy adaptation, 2024. URL https://arxiv.org/abs/2311.14402.

[2] Sion An, Myeongkyun Kang, Soopil Kim, Philip Chikontwe, Li Shen, and Sang Hyun Park. Bci competition iv-2b, dec 2024. URL https://service.tib.eu/ldmservice/dataset/bci-competition-iv-2b

[3] Martin Wimpff, Mario Döbler, and Bin Yang. Calibration-free online test-time adaptation for electroencephalography motor imagery decoding. In 2024 12th International Winter Conference on Brain-Computer Interface (BCI), pages 1–6. IEEE, 2024.
