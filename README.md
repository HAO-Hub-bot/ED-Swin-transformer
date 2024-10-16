# This project is mainly applicable to image classification
# Environment configuration<br>
   python     3.7<br>
   pytorch    1.10.0<br>
   torchvision  0.11.0<br>
   numpy         1.21.6 <br>
   cudatoolkit     10.2<br>
# usage <br>
  ## class_indices.json 
  Classification categories of images
  ## create confusion matrix 
  Draw the confusion matrix of the model using the trained weights
  ##  model
  The overall architecture of ED Transformer
  ## prediction
  Select images and choose appropriate weights to predict which category they belong to
  ## select_incorrect_samples  
  Select the classification samples with prediction errors in the model weights
  ##  train
   Function for model training, set basic training parameters
  ## utils
   Some basic settings, such as dataset partitioning and other operations
