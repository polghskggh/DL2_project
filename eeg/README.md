# EEG-TEA: A Novel Application of Test-time Energy Adaptation in Electroencephalogram Decoding

## Introduction 

This is the official repository for the paper "EEG-TEA: A Novel Application of Test-time Energy Adaptation in Electroencephalogram Decoding". 
In this work, we apply Test-time Energy Adaptation (TEA), a method developed for image data, to the time-series domain,
specifically with electroencephalogram (EEG) data.
EEG-based Deep Learning models often struggle to generalize across subjects due to the high inter-subject variability,
which motivates the need for test-time adaptation (TTA) methods to improve model robustness and performance 
in real world setting.

We conduct our experiments on the task of motor imagery decoding using datasets 2a and 2b from the BCI Competition IV.
We have conducted 2 experiments on cross-subjection adaptation and corruption adaptation, to test out the method's effectiveness
on different types of distribution shifts.

In the cross-subject adaptation setting, we train a model on dataset 2a, where the source model is trained on all subjects
excluding the target subject, and then adapt the model to the target subject using TEA. We have tested out two different 
variants of this setting, one where we left out one subject from the training set, and the other where we left out 3, 
to simulate a larger distribution shift. 

In the corruption adaptation setting, we train a model on dataset 2b, where the source model is trained on one subject and 
adapt the model to the corrupted data of the same subject. 

We show that TEA can be applied to EEG data and that it is able to improve the performance of the model in both settings. 
#### 

## Related work 

## Weakness/strength 

## Novel contribution 

Our novel contribution includes the application of TEA, which was originally developed for images, to EEG data. 
Furthermore, we demonstrate that initialisation the noise for SGLD sampling, with pink noise, rather than white (Gaussian)
noise, results in the generation of more realistic samples, leading to improved model performance.

## Results 

### Installation
- clone this repository
- run `pip install .`

# Conclusion

# Contributions

## Citation
If you find this repository useful, please cite our work
```
@inproceedings{wimpff2024calibration,
  title={Calibration-free online test-time adaptation for electroencephalography motor imagery decoding},
  author={Wimpff, Martin and D{\"o}bler, Mario and Yang, Bin},
  booktitle={2024 12th International Winter Conference on Brain-Computer Interface (BCI)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
