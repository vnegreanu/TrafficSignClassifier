**Traffic Sign Classifier** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Rubric Points

#### Submission Files

This project includes

- The notebook `Traffic_Sign_Classifier.ipynb` (and `signames.csv` for completeness)
- `report.html`, the exported HTML version of the python notebook
- A directory `additional_traffic_signs` containing images found on the web
- `writeup.md`, is the current file

---

### Data Set Summary & Exploration

- Number of training examples = 34799
- Number of valid examples= 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

#### Dataset vizualization
![Data vizualization](output_images/dataset_viz.png)

#### Classes before data augmentation
![Classes before data augmentation](output_images/classes_before_augmentation.png)

#### Classes after data augmentation
![Classes after data augmentation](output_images/classes_after_augmentation.png)


---

### Design and Test a Model Architecture

First the  train data is prepreocessed. Techniques used for preprocessing are:
* Grayscaling
* Normalization
* Adjusting brightness
* Adjusting scaling
* Shuffling
* Spliting is done via 80/20 percentage rule 80% -> train data and 20% -> validation data

#### Common techniques:
- Grayscale - reducing the number of channels therefore the computation time on GPUs
- Normalization was done in range (-1/1).Used to change the range of pixel intensity values. Usually the image data should be normalized so that the data has mean zero and equal variance.

#### Augmentation data techniques: 
They are done to boost the classes that had less than 1000 samples 
- adjust brightness :to get rid of the dark pixels will improve general recognition of the images - Samples are uniformly distributed over the half-open interval [low, high]
- adjust scaling using perspective transformation. 
We need a 3x3 transformation matrix. 
Straight lines will remain straight even after the transformation. 
To find this transformation matrix, we need 4 points on the input image and corresponding points on the output image. 
Among these 4 points, 3 of them should not be collinear.

The model chosen is LetNet5. The model architecture is illustrated in the following table:

| Layer             | Size and  features   | 
| -------------     |:--------------------:| 
| Input             | size : (32, 32, 1) | 
| Conv1             | size (28,28,6); filters : 6; kernel size : (5 x 5); stride : (1 x 1); padding : VALID; activation : RELU |
| Pooling           | size (14,14,6); filters : 6; kernel size: (2 x 2); stride: (2 x 2); padding : VALID;|
| Conv2             | size (10,10,16); filters : 16; kernel size : (5 x 5); stride : (1 x 1); padding : VALID; activation : RELU | 
| Pooling       	| size (5,5,16); filters : 16; kernel size: (2 x 2); stride: (2 x 2); padding : VALID;|
| Flatten 	    	| (5,5,16) => 400 |
| Fully Connected 1 | neurons : 120; activation : RELU |
| Fully Connected 2 | neurons : 84; activation : RELU |
| Output 	    	| size : (43 x 1) |


#### Description of the model:
- model uses two convolutional layers: 
- first will take as input the gray image and outputs a 6 filter matrix with kernel size set to 5 and smale stride for going in detail over the image.
- second will will have much more filters 16 and a bigger stride since this layer is also responsable for recognizing new images that are outside of the train or validation datasets. 
- model uses ReLu as activation function for each of the convolutional layers and itermediate fully connected layers for efficent computation and better gradient propagation trying to eliminate vanishing gradient problems.
- model uses two Pooling layers after each convolutional layers to to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network therefore preventing overfitting.
- model uses a Flatten layer to go from 3D space in 1D space before going into classification.
- model uses three fully connected layers including output layer used for classification and to make the model end-to-end trainable.

#### Parameters of the model:
- Number of epocs: 50 -> As long as the training is longer we can acheive better results.
- Batch size: 128 -> defines how many examples we look at before making a weight update. 
The lower it is, the noisier the training signal is going to be, the higher it is, the longer it will take to compute the gradient for each step but we eliminate noiser data. 
Other values tried were 64, 256, 1024. Didn't show good results.
- Learning rate: 0.001 -> which tells the network how quickly to update the weights.
- mu: 0 - default parameter of the LeNet CNN
- sigma: 0.1 - default parameter of the LeNet CNN
- optimizer: - Adam was chosen because has better performace than Adadelta and RMSprop and also keeps an exponentially decaying average of past gradients.

Final model results were:
* validation set accuracy of 99,4 - target was met
* test set accuracy of 91.6%
* model accuracy on new traffic signs is 100%

### Force model to recognize newly added signs. 
- A directory `additional_traffic_signs` containing images found on the web
- 5 new traffic signs were added I have renamed them accordingly to the entries inside the signames.csv to match the exact class ids.
- Prediction is 100% for this run (e.g. there were previous runs where the prediction found 4 out of 5 signs).
- I used semi-easy images to classify and even modified them slightly. I made them all uniform in size by reshaping them.
- For my images 1, 2, 3 and 5 my model was 100% sure that the results were correct based on softmax probabilities. For image number 4 the probability was 64% and still got it right.

image1->Original: Speed limit (30km/h) Prediction-> 1. Speed limit (30km/h): 100.00% 2. Speed limit (80km/h): 0.00% 3. Speed limit (60km/h): 0.00% 4. Speed limit (50km/h): 0.00% 5. End of speed limit (80km/h): 0.00%

image2->Original: Priority road  Prediction-> 1. Priority road: 100.00% 2. No passing 0.00% 3. Wild animals crossing: 0.00% 4. Stop: 0.00% 5. No vehicles 0.00%

image3->Original: General caution  Prediction-> 1. General caution: 100.00% 2. Go straight or left: 0.00% 3. Bicycles crossing: 0.00% 4. Dangerous curve to the right: 0.00% 5. Go straight or left: 0.00%

image4->Original: Road work  Prediction-> 1. Road work: 64.00%  2. Go straight or left: 35.00% 3. Bicycles crossing: 0.00% 4. Dangerous curve to the right: 0.00% 5. Wild animals crossing: 0.00%

image5->Original: Turn left ahead  Prediction-> 1. Turn left ahead: 100.00% 2. Children crossing: 0.00% 3. Bicycles crossing: 0.00% 4. Beware of ice/snow: 0.00% 5. Ahead only: 0.00% 

#### New images vizualization
![New images vizualization](output_images/new_sings.png)

#### New images with labels after prediction
![New images with labels after prediction](output_images/new_signs_with_label.png)

#### Softmax probabilities:
- tf.nn.softmax will compute the softmax activations and returns a tensor. 
- tf.nn.top_k will return the values and indices (class ids) of the top k predictions. So if k=5, for each sign, it'll return the 5 largest probabilities out of a possible 43 and the correspoding class ids.

#### Discussions and further improvements
- Size of newly added image is 32x32 which is quite good that we can prepare them directly for the CNN without any other transformation.
- The images have high blur but this can be adjusted with the the data augmentations techniques like adjsuting scaling which was previously described
- The contrast is not good for some images like (e.g. in the notebook the left turn) where the sky has the same color as the sign background therefore confusing the model. This can be adjusted by increasing brightness or by wrapping eliminating the blue pixels and leaving just the arrow.
- Model accuracy on the new traffic signs is on current run 100%, while it was 92,8% on the test set. Sometimes the prediction accuracy can be as good as 90% or higher.
- The predictions of information signs like yield, priority road, general cautions, and speed limit like 30km/h are actually close enough.
- The model now confusses work road with animal crossing which are pretty similar. Those images have high clutter.  
- I think to get the consistent correctness, I need more good data by some more data augmentation techniques. Unfortunately these are extremly time consuming and 




