# CSE455-Final-Project
A basic handwritten Japanese Optical Character Recognizer

# Introduction
The problem I was trying to solve was creating a application which can take in an image containing handwritten Japanese which it can then recognize and process. However, I didn't want just a direct translation of the recognized text using some machine translation software, but instead I want to application to take the recongized text and parse words and phrases in the text. This application would be used by people who already have a grasp of Japanese, but are not proficent.

# Data
For data I used data from ETL Character Database. Specifically, I used the dataset ETL8. ETL8 has the 71 basic hiragana and 879 of the most commonly used Kanji.

The Hiragana and Kanji data had 160 writers each writer writing each character one time. Additionally, I used Keras ImageDataGenerator to create more samples with 15 degrees of rotation and 0.2 degress of zoom.

# Approach
