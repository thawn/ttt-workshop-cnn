# Prompt for GitHub copilot agent mode

we are going to teach a train the trainer workshop on convolutional neural networks.

set up a git repository to be distributed via github as jupyter book

notebooks will be executed on google colab.

add readme explaining the technical setup

## Course contents

the total course length will be 3 h

The course schedule is as follows:

* 20 min slide deck (jupyter notebook with slide mode)
  * Explain how convolutions work
  * Explain pooling layers
  * Explain U-Net
  * show example use cases in scientific image analysis (microscopy, segmentation)
  * Teaching methods compared
    * Classical lecture + exercises
    * Pair programming exercises
    * Flipped classroom + exercises
* classical exercise manually programming convolution
* classical exercise train a single convolution layer with pytorch to learn a gaussian filter (use a 32x32 pixel smiley as input and a blurred smiley as label)
* Self-teaching Jupyter notebook explaining u-nets
* Pair programming session implementing a u-net that learns to segment nuclei based on the BBBC039 dataset https://bbbc.broadinstitute.org/BBBC039
* Self teaching notebook on hyperparameters
* Exercise on hyperparameter tuning with optuna
