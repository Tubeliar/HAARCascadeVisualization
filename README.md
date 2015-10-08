# HAARCascadeVisualization
Perform and visualise object detection with OpenCV's CascadeClassifier. Compile the executable and run it without arguments to get a short description of the available options.

Watch an example of the type of video you can create with this here: https://youtu.be/L0JkjIwz2II

Read about why this was created here: https://timvanoosterhout.wordpress.com/2015/10/08/recreating-a-haar-cascade-visualiser

# How this was done
This project is based on the ufacedetect example. I knew I would have to mess with OpenCV's internals to get the control over the algorithm I wanted so I tracked which functions were called and in which files they lived and started copying the ones I would need to modify. When I had collected what I believed was the minimum set of source files I started to add functionality to report feedback on the algorithm's progress. Then I needed to figure out how to actually draw the features.

When the drawing was done I added some options to speed up the resulting visualisation. If the algorithm runs normally you can easily get visualisations lasting hours for even moderately sized images. Keep in mind that normally no steps are left out and the cascade completes in a few milliseconds!
