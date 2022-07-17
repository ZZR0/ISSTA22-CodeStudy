## General Introduction
This repo illustrates how to evaluate the artifacts in the paper *An Extensive Study on Pre-trained Models for Program Understanding and Generation* published in ISSTA'22. Specifically, we have extensively investigated the effectiveness and limitations of NL-PL pre-trained models for program understanding and generation tasks. We first discover the performance fluctuation of different pre-trained models over different tasks and datasets, which indicates that it can be challenging to propose an almighty pre-trained model across task types and it is essential for reliable experiments to demonstrate the superiority of proposing new models. Furthermore, we also validate the superiority of pre-trained models over conventional/previous SOTA methods on different downstream tasks. Finally, we perform the first study for NL-PL pre-trained model robustness via adversarial attacks and find that the existing pre-trained models are rather vulnerable, e.g., they can be easily attacked by a simple random attack approach, and current strategies for improving the robustness of pre-trained code models have limited effectiveness. Therefore, researchers should make more efforts on proposing integration schemes of additional information with pre-training.

Due to the random nature of neural networks, users may obtain slightly different results via retraining the models. Please note that such results usually can be tolerated, i.e., they mostly do not conflict with the conclusions of the paper.

## Environment Preparation

- *Hardware*: GPU: 2080Ti * 8; CPU: Intel Core i7, 128GB RAM, 50GB free disk space, or above. 

- python 3.8+
    ```
    argparse               1.4.0
    numpy                  1.20.1
    pandas                 1.2.3
    matplotlib             3.4.1
    sklearn                0.0
    tqdm                   4.59.0
    torch                  1.8.2
    torchaudio             0.8.2
    torchvision            0.9.2
    tensorboardX           2.5.1
    transformers           4.19.2
    tokenizers             0.12.1
    ```
- Ubuntu 18.04
- CUDA Toolkit

## Code Structure

We list the program directories and their files which can be used by artifact evaluations as follows.

- `./DGMS`: Not pre-trained sota approach of code search.
- `./FCDetector`: Not pre-trained sota approach of clone detection.
- `./PLBART`: Source code of PLBART.
- `./rencos`: Not pre-trained sota approach of code summarization.
- `./ReVeal`: Not pre-trained sota approach of defect detection.
- `./Task`: The directory of the 7 program understanding and generation tasks and corresponding pre-trained models. 
    - `Clone-Detection-BigCloneBench/`
    - `Clone-Detection-FCDetector`
    - `Clone-Detection-POJ-104/`
    - `Code-Generation`
    - `Code-Refinement`
    - `Code-Search`
    - `Code-Search-FB-Java`
    - `Code-Summarization`
    - `Code-Summarization-Rencos`
    - `Code-Translation`
    - `Defect-Detection`
    - `Defect-Detection-Reveal`

## Adversarial Attack
The artifact of adversarial attack for pre-trained code models can be found in [CodeAttack](https://github.com/ZZR0/CodeAttack).

## Acknowledgement
Our implementation is adapted from: https://github.com/microsoft/CodeXGLUE, https://github.com/wasiahmad/PLBART, https://github.com/microsoft/CodeBERT, https://github.com/parasj/contracode, https://github.com/agemagician/CodeTrans, https://github.com/salesforce/CodeT5, https://github.com/justinphan3110/CoTexT, https://github.com/ryderling/DGMS, https://github.com/shiyy123/FCDetector, https://github.com/zhangj111/rencos, https://github.com/VulDetProject/ReVeal
