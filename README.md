# hair-style-recommendation
Using AI to recommend a hair style (in progress)

Step 1: 
![](/assets/fig1.png)

Using MODNet, a state-of-the-art Portrait Matting Tool, I removed the noisy background from this image. MODNet employs ML techniques to subtract the background and return a segmented mask of the foreground

Step 2:
![](/assets/fig2.png)
Next, I scaled, rotated, translated and cropped the image. The idea behind doing this was to ensure consistency among all pictures taken from various angles and positions. Through these steps, you can now compare your standardized image with other images that have undergone a similar pre-processing step. 

Using the facial_recognition library, I found the top, right, bottom and left coordinates of the input facial image, identifying important landmark features like eye location, eyebrows, nose, chin point and jawline. I computed a few additional features like jaw width, face height, face width, avg. eyebrow arch, avg. eye length, nose height, etc.

![](/assets/fig3.png)

Step 3:
Using a dataset consisting of ~5,000 female celebrity images, I used a pre-trained VGG16 with weights from VGG Face to classify the input image according to its facial shape [oval, round, square, heart, oblong].

Step 4: 
(Code in progress)
Using another dataset consisting of ~1,000 images, I trained a CNN to classify the input image according to its hair texture [curly, straight and wavy].

Step 5: 
(Code in progress)
The underlying assumption of this system is that if I have the same face shape, measurements and hair texture as somebody else, then I would be more likely to prefer their hairstyle over some randomly chosen individuals.
I created a dataset by running all images from the face shape detection dataset through the pre-processing pipeline. Every image in that dataset had the following features:
A. Facial measurements [face width and height, nose width and height, etc.]
B. Face shape [oval, round, square, heart, oblong]
C. Hair texture [curly, straight, wavy]
Now, using these features, I created a recommender system to recommend an ideal hairstyle.


Medium Article Link: https://medium.com/@kopalgarg/using-ai-to-pick-the-most-flattering-hairstyle-based-on-your-face-shape-735642791625
