Deep Learning Captcha Breaker
===================

Convolutional Neural Network (CNN) is a type of deep artificial neural network that has been successfully applied to solve many image-processing problems. It can be used to identify faces, objects, animals, recognize traffic signs in self-driven cars and many other applications. Here we are going to use a CNN to solve one of the most basic tasks in machine learning: The character recognition. We will use it to break a simple numeric Captcha.

This simple project was built to solve a real problem, where I had to build a Web Scrapping code to gather some desired information. My first solution included to call 2Captcha API, a russian service that uses human labor to return the captcha answer as text. My second attempt was to learn some Deep Learning techniques, which I've used to build this simple algorithm.

The images used to build the train and test sets are in the Labeled Captchas directory inside this project. The images were labeled using 2Captcha API, but you can build your own set as you desire. Since my project was entirely made in Java, I included in the JavaDLCaptchaBreaker the code used to adapt the Keras trained model to run in native Tensorflow. If you have any doubt, feel free to contact me!


