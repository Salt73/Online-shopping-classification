# Online-shopping-classification
Binary classification for the products of online shopping platform  between the footwear and the apparel.
# ____________________________________________________________________

# In the following statements I'll go through the functionalities of this project and the used techniques, I'll try not to make it boring :D



## First of all I've imported the libraries and the frameworks I've used as :
 pandas (to deal with dataframes), matplotlib and seaborn which are used in visualization, os that helped me to navigate and create the folders I needed in addition to shutil that helped me alot to manipulate the folders and it's contents, CV2 to deal with the images (as our dataset is images), tensorflow and all it's relates(keras and what is imported from it as keras layers, optimizers) in addition to using sklearn  which are mainly used in the model building, training fine tuning, testing and evaluation.

## then I mounted my drive to import the data from it (to be used in training, validation and testing)

 
 After that I've define a function(create_dataframe) to generate dataframes for each of the training, validation and testing data as dealing with data frame is much easier for me than dealing with folders You have as we our dataset (image files) is divided into 3 folders (train, test, validation)folders for each folder there are 2 sub-folders ('Apparel' and 'Footwear').
This function(create_dataframe) reads the paths through these folders and creates a list of file paths(images' paths) and their corresponding labels ('Apparel' or 'Footwear').
It then puts this information into a dataframe(looks like a table or an excel-sheet).
It repeats this process forall the main folders (train, validation, test).
Finally, it prints out how many images are in each dataframe.


And for more specific information about the dataset, I've created another function with a very simple role which is telling the number of classes in each data drame and the number of the images in each class and which of them is the maximum(has more images than the other class) and which is the minimum, I've applied this function to all of the 3 dataframes (train,test,validation).
## **************************************************************************


# Now let's go ahead to the data preprocessing and preparation phase

data preprocessing and preparation is considered the most important phase/step in our pipeline, as we know   __*if we let a trash in we get a trash out*__   which means if we used a dirty data to train whatever the model is, we can't expect a good results from this model.

Now, let's go through our data processing funtion, to understand the role of each and how it works.


### 1) setup_data_generators(train_df, valid_df, test_df, img_size, batch_size=20)
  this function sets up data generators to efficiently process image data for training, validation, and testing of a machine learning model

  It takes the following arguments:

  + train_df: DataFrame containing training data with file paths to images and their corresponding labels.
  + valid_df: DataFrame containing validation data with file paths to images and their corresponding labels.
  + test_df: DataFrame containing testing data with file paths to images and their corresponding labels.
  + img_size: __Tuple__ specifying the target size of the images (height, width).
  + batch_size: (Optional) Integer specifying the batch size for training, I've setted it manually to 20.

#### Now let's see this function's steps

1) Setting up Data Generators:
    +  It prepares generators(that generates the augmented data) for training, validation, and testing data.
    +  For training data, it applies transformations like flipping images horizontally and rotating them slightly.
    +  For validation and testing data, it doesn't apply these transformations because we want to keep the original images unchanged

2) Preparing Data for Training, Validation, and Testing :
    + For training data, it creates a generator (train_gen) that returns __batches__ of images and their corresponding labels from the training DataFrame.
    + For validation data, it creates a similar generator (valid_gen) but for the validation DataFrame.
    + __For testing data, it calculates an optimal batch size based on the length of the test DataFrame so that each image gets tested exactly once.__
  
3) Configuring Data Properties :
    + It extracts important information such as the number of classes, their names, and their corresponding indices.
    + It prints out some information like the test batch size and the number of classes.
    + Note that for all the three generators, the (class_mode,color_mode) are the same (binary, RGB) but for the * shuffle * omly the train_gen is allowed to shuffle its images (setted into True) but for val_gen and test_gen it's setted into False as we want to keep the original images unchanged.
  
4) Returning Generators :
    + At this step, we just return our 3 generators for the later use in training and evaluating the model.
  
### 2) show_image_samples(gen):
  this is a very simple function which is used to display a sample batch of images along with their corresponding labels to show the images we have and the transformations (changes) that were applied to it.

  #### Let's say what it does in bries

  1) Fetching a Sample Batch :
      + Fetch a sample batch of images and their labels from the generator (gen). *The next() function is used to retrieve the next batch from the generator*.

  2) Printing Batch Information :
      + It prints the number of images and labels in the batch

  3) Displaying Images :
      +  creates a plot to display the images where the displayed images must be no more than 25 images at once

  4) Setting Up Subplots :
      + Iterates over the images in the batch and plots each one individually
      + sets up subplots in a grid of 5 rows and 5 columns to display the images
      + For each image:
        + It normalizes the pixel values to be between 0 and 1 by dividing by 255(as therange of the colors is 0:255).
        + It plots the image using imshow().
        + It assigns a title to the image based on its label. (If the label is 1, the class is "Footwear"; otherwise, the class "Apparel".
        + It turns off the axis for each subplot(delete the axis as we don't want to see the grid-shape).
    
  5) Displaying the Plot:
      + Finally, it displays the plot showing the sample images with their corresponding labels

## **************************************************************************
     
# Now let's see how I built my model


 ### Here are the steps of how I built my model

1) Setting Image Shape :
   +  I set the shape of the images to be used in the model according to "img_size"(which is a tuple we defined before to be (224,224).
  
2) Defining Model Name:
   + I prefered to use the "mobileNet" due to it's simple and efficient architecture which makes it run on a relativly weak hardware and still give a pretty good results.
  
3) Initializing Base Model :
   +I've initialized the base model, __MobileNetV2__,  using the pre-trained weights from 'imagenet' and setting the input shape according to the specified image shape(224,224,3). Additionally, I've set the pooling method to 'max'.
  
4) Modifying the Model Architecture :
   + I've added a fully connected layer (Dense) with 256 units and ReLU activation function, This layer also applies regularization techniques such as L2  regularization and L1  and bias regularization.
   + Then I added another fully connected layer with a single unit and sigmoid activation function (as we are doing a binary classification), which outputs the final prediction.
   + Finally, I constructed the model as "modell" using the input from the base model and the output layer defined above.
  
5) Setting Learning Rate :
   + I set it manually (randomly) to 0.01
  
6) Compiling the Model :
   + I used the Adamax optimizer with the specified learning rate(0.01).
   + I set the loss function to 'binary_crossentropy'(as we are performing a binary classification).
   + Finally, I used the 'accuracy' metric for the training, validation evaluation as it is the most redable and understandable metric . 

# After building and training our model, we've seen that it's not performing well on the training neither testing, so we have to fine tune it

  ### Let's go through the fine tuning steps quickly

  at the tunung I've used several techniques, But unfortunately, I've used only some of it (Due to the lake of the powerfull hardware problem I've mentioned previously)
#### Those techniques are

1) Grid-Search:
   + I've used grid-search to train the model using a set of hyper parameters I've specified as learning rate and the number of epoch to select the most suitable comination of the the learning rate and number of epochs.
  
2) Early-Stopping:
   + It's a very powerfull and useful technique that makes the training process stops at a certain epoch based on the criteria I specify (the validation accuracy in our case) and it resulted in much better performance on training as it reduces the number of unnecessary computations also reducing overfitting
  
3) Regularization :
   + This is a technique that is used to enhance generalization and reduce overfitting it has several methods; in our case, we have use (Batch-normalization, Droupout,Adamx optimizer, L2 regularization, L1)
  

## Finally I tested both models and draw a plots to compare between there training and validation accuracies the I saved it as the last step 




# I hope you have enjoyed this notebook and found it useful :)
