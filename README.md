# CSE455-Final-Project
A basic handwritten Japanese Optical Character Recognizer

# Introduction
The problem I was trying to solve was creating a application which can take in an image containing handwritten Japanese which it can then recognize and process. However, I didn't want just a direct translation of the recognized text using some machine translation software, but instead I want to application to take the recongized text and parse words and phrases in the text. This application would be used by people who already have a grasp of Japanese, but are not proficent.

# Data
For data I used data from ETL Character Database. Specifically, I used the dataset ETL8. ETL8 has the 71 basic hiragana and 879 of the most commonly used Kanji.

The Hiragana and Kanji data had 160 writers each writer writing each character one time. Additionally, I used Keras ImageDataGenerator to create more samples with 15 degrees of rotation and 0.2 degress of zoom.

# Approach
![Flow Diagram of Application](./Images/FlowDiagram.png)

This application can be broken up into 4 main parts. Preprocessing, segmentation, prediction, and then parsing. 
For preprocessing, I first take in the image given by the user and first make it grayscale because we don't care about color pixels in this case of character recognition. Next I invert the image because the model is trained using images that are white characters set on a black background. Next, I thresh the images to make all pixels in the image be either completely black or completely white. Finally, I crop the image to get rid of extra space that is not needed. 

![Original Example](./Images/original_resize.jpg)
![Grayscale Example](./Images/gray_resize.jpg)
![Invert Example](./Images/invert_resize.jpg)
![Thresh Example](./Images/thresh_resize.jpg)

The second step is segmentation. The program makes a histogram of all the white pixels in each column and row. This way, for each character there will be a high amount of white pixels and when there is no character or a space then there will be no/few white pixels. By tracking the spaces in these histograms we can create proper bounding boxes for the characters. After creating proper bounding boxes, parse each individual detected image out of the full image and crop any extra space.

![x histogram](./Images/x_hist_resize.png)
![y historgam](./Images/y_hist_resize.png)
![bounding boxes](./Images/bounding_boxes_resize.jpg)

The third step uses three convolutional neural networks to predict the character. The Japanese Language has multiple character types, mainly split into Hiragana, Katakana, and Kanji. My program only handles Hiragana and Kanji for now. Since there are different character types, I have one CNN that first identifies which type of Japanese character the character is and then sends the character to either the Hiragana CNN or the Kanji CNN. 

Each CNN takes in a 48x48 pixel input and outputs a classification. All models use Relu activation and generally used 40 epochs.\
Training and testing data was split 80/20

Hiragana CNN Model:

Conv2D-64 > Max Pool > Conv2D-128 > Max Pool > Conv2D-192 > Max Pool > Conv2D-256 > Max Pool > Flatten > Dense-1024 > Dense-1024 > Softmax-71

Kanji CNN Model:

Conv2D-64 > Max Pool > Conv2D-64 > Max Pool > Flatten > Dense-2048 > Softmax-879

General Character Identification:

Conv2D-64 > Max Pool > Conv2D-128 > Max Pool > Conv2D-192 > Max Pool > Conv2D-256 > Max Pool > Flatten > Dense-1024 > Dense-1024 > Softmax-71

|          | Hiragana | Kanji  | All Characters |
| -------- | -------- | ------ | -------------- |
| Accuracy | 99.55%   | 98.61% |    99.71%      |

The final step is just feeding the output of all the identified characters into ichiran and then recieving the output

# Video Link (Demo)
xd

# Experiments and Discussion
Character Segmentation
At first I tried using the built in character segmentator that comes with tesseract-ocr, but found it was extremely sensitive to the test inputs I was giving it and it tended to make bounding boxes too tight. Therefore, I ended up writing my own character segmentor which has several limitations, but works decently. 

CNN Models
Despite what the accuracy shows, in practice the models work poorly with real world example that I give it. Honestly, it could just be that I have bad handwritting but I noticed that there are some characters that the models have extreme difficulty detecting. As such, I tried expanding the models to have more layers and I also wanted to make the input 64x64 pixels instead of the current 48x48 pixels, but due to time constraints I decided not to use a larger model. 
Another issue in regard to models is that for the model that identifies all characters, I did this by having the input take all character samples from both hiragana and kanji. However, because there are so many more kanji than hiragana samples, the models has a tendency to predict many hiragana characters as kanji characters

# Results and Improvements
Overall it kinda works. The main issue is with the models. 

If I had more time I would train more models with more layers and a higher pixel input of 64x64. For the Kanji CNN I had planned on using a model that was structured as such:

Conv2D-64 > Max Pool > Conv2D-128 > Max Pool > Conv2D-512 > Conv2D-512 > Max Pool > Flatten > Dense-4096 > Dense-4096 > Softmax-879

Aside from model training, I would also add more data for other character types so that the data set would be less biased. Additionally, I would also include katakana into the model which is the 3rd Japanese character type. 

For preprocessing, the program can't rotate images, so that is another preoprocessing step I would implement. Currently, the program only really works with ideal inputs.

# Ichiran Open Source Application
https://github.com/tshatrov/ichiran
# References
http://cs231n.stanford.edu/reports/2016/pdfs/262_Report.pdf 
https://towardsdatascience.com/creating-a-japanese-handwriting-recognizer-70be12732889 
https://github.com/Nippon2019/Handwritten-Japanese-Recognition 
https://towardsdatascience.com/segmentation-in-ocr-10de176cf373 

