# Recognizing Video Activities in the Wild via View-to-Scene Joint Learning
[Abstract] We propose cost-efficient neural networks (called WildVAR) to learn view-consistency and scene information jointly without any 3D pose ground truth labels, a new approach to recognizing video actions in the wild. Unlike most existing methods, first, we propose a Cubing module to self-learn body consistency between views instead of comprehensive image features, boosting the generalization performance of across-view settings. Specifically, we map 3D representations to multiple 2D features and then adopt a self-adaptive scheme to constrain 2D features from different perspectives. Moreover, we propose temporal neural networks (called T-Scene) to develop a recognizing framework, enabling WildVAR to flexibly learn scenes across time, including key interactors and context, in video sequences. 

This is the implementation of WildVAR in Tensorflow. It contains complete code for preprocessing,training and test. Besides, this repository is easy-to-use and can be developed on Linux and Windows. 

## Getting Started
### 1 Prerequisites  
* Python 3.x  
* Tensorflow 1.x    
* Opencv-python  
* Pandas  

### 2 Datasets
* [Kinetics-400](https://deepmind.com/research/open-source/kinetics) consists of approximately 240,000 videos for training and 20,000 videos for validation, encompassing 400 different action categories.
* [Something-Something v2](https://developer.qualcomm.com/software/ai-datasets/something-something) is a temporal-related video dataset that comprises 169,000 training videos and 20,000 validation videos, with 174 unique action classes to classify.
* [NTU-60 and NTU-120](https://github.com/shahroudy/NTURGB-D)  are benchmarks, each comprising a large-scale collection of videos depicting human actions. NTU-60 contains 57,000 videos of 60 different human activities, while NTU-120 contains 114,000 videos of 120 activities.

### 3 Preprocess
1. Store video data in `../Wild_VAR/Data/{Datasetname}/Raw_Data`.


2. Prepare video clips  
An example to achieve video clips of NTU120 is:
`cd ../Wild_VAR/Data`  
`run python prepare_clips.py`  
   * Clips generated will be saved in the subfolders in   `../Wild_VAR/Data/{Datasetname}/Train`, `../Wild_VAR/Data/{Datasetname}/Val`. These clips will be used for training and validation.  
   * Samples are divided into folders by category.
   * The data collected from two perspectives of the same action are grouped together.
   * We resize every video to a standard size of 256 × 256.
   * We extract a clip from the entire video. We randomly crop the selected clip to a size of 224 × 224 and apply a random horizontal flip to augment the dataset.

### 4 Train model
An example command-line to train WildVAR on NTU120 is:

`cd ../Wild_VAR`  
`run python train.py PB` or `python train.py CHECKPOINT`
 
* The model will be saved in directory `../Wild_VAR/Weight`, where "PB" and "CHECKPOINT" is two ways used for saving model for Tensorflow.  
* The parameters are all defined by [configuration files](Checkpoint).
* The training set and validation set in directory `../Wild_VAR/Data/{Datasetname}/Train` and `../Wild_VAR/Data/{Datasetname}/Val`. 

### 5 Test model(pb)  
An example command-line to train WildVAR on NTU120 is:
 
`cd ../Wild_VAR`  
`run python test.py N`  
* Where N is not more than the number of clips in test set. Note that we do not use min-batch during test. There may be out of memory errors with a large N. In this case, you can modify the `test.py` to use min-batch.    
* The parameters are all defined by [configuration files](Checkpoint).
* The test set in directory `../Wild_VAR/Data/{Datasetname}/Test`. 
### 6 Visualize model using Tensorboard  
`cd ../Wild_VAR`  
`run tensorboard --logdir=Model/`   
* Open the URL in browser to visualize model.  

### 7 Results
Dataset| Top-1 (%) | Top-5 (%) |
------------            |:-----:| :-----:|
Kinetics-400                  |   86.5  |  97.5 | 
Something-Something v2           |  75.7  |  95.8 | 

<table>
    <tr>
        <td><b>Dataset</b></td> 
        <td colspan="2"><b>NTU-60</b></td> 
        <td colspan="2"><b>NTU-120</b></td> 
   </tr>
    <tr>
  		 <td>Methods</td> 
  		 <td>X-sub</td> 
  		 <td>X-view</td> 
  		 <td>X-sub</td> 
  		 <td>X-set</td>
    </tr>
    <tr>
        <td>WildVAR(V)</td> 
        <td>94.3</td> 
        <td>97.1</td> 
        <td>90.2</td> 
        <td>92.1</td>    
    </tr>
    <tr>
        <td>WildVAR(S)</td> 
        <td>94.8</td> 
        <td>97.5</td> 
        <td>90.6</td> 
        <td>92.2</td>    
    </tr>
</table>
