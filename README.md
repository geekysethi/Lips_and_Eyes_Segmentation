# Lips and Eyes Segmentation
 
## Problem Statement
Can we use computer vision algorithms to segment out the lips and both the eyes from Celeb dataset.

## Proposed Solution
In this section I have explained my proposed solution and in the further section I have given the extensive analysis of results.

### Data Preprocessing
For data preprocessing, I combined the four the images of (left eye, right eye, upper lip and lower lip) into one image. So that we can train our model in end to end fashion.

### Deep Learning Model
For Deep Learning Model here I have used Vanila UNet? model for segmentation. In which our input image size was 256 \times 256 \times 3. 
For calculating the loss I used Dice Loss and for metric calculation I have use IOU( Intersection Over Union).

### Choice of Hyperparameters
The hyperparameters which I have used in this assignment are 
* Learning Rate: 5e-4
* Optimization: Adam

### Dataset Used

## Results


|   |Training   |Validation  |testing   |
|---|-----------|--------|--------------|
|Dice Loss      |0.084   |0.098   |0.098|
|Mean IOU       |0.776   |0.763   |0.762|


### Training Error Plot
![errorplot](report_images\training_loss.png)


### Validation Error Plot

![validationplot](report_images\validation_loss.png)

### Training Mean IOU Plot

![meaniou](report_images\train_iou.png)

## Visualizations
![results](report_images\results.png)

