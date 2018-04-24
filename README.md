Deep Learning Captcha Breaker
===================

Convolutional Neural Network (CNN) is a type of deep artificial neural network that has been successfully applied to solve many image-processing problems. It can be used to identify faces, objects, animals, recognize traffic signs in self-driven cars and many other applications. Here I've used a CNN to solve one of the most basic tasks in machine learning: The character recognition. This algorithm is used to break a simple numeric Captcha.

This simple project was built to solve a real problem, where I had to build a Web Scrapping code to gather some desired information. My first solution (the boring solution) included to call 2Captcha API, a russian service that uses human labor to return the captcha answer as text. My second attempt (the funny solution) was to learn some Deep Learning techniques, which I've used to build this simple algorithm.

The most challenging part of this project was to break the Captcha into characters to feed the CNN. The images don't have a fixed number of characters, as in other Captcha breakers projects. To deal with this problem, I've to learn some computer vision techniques and used [OpenCV](https://opencv.org/) to separate the characters. The CaptchaImageProcessor class includes more details of the techniques used in this project.

The images used to build the trainning and test sets are in the Labeled Captchas directory. Most of them were labeled using 2Captcha API, but you can build your own set as you desire. Since this project was entirely made in Java, I included in the [JavaDLCaptchaBreaker](https://github.com/marinelligiovanna/JavaDLCaptchaBreaker/tree/master) repository the code used to adapt the Keras trained model to run in native Tensorflow. For this reason, I didn't build a Deep Learning Decaptcher class in Python. But you can easily find how to use the trained model in Keras. You can also run it in C++/C# using the same pb file.

If you have any doubt, feel free to contact me.


