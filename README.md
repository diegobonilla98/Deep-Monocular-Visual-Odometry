# Deep-Monocular-Visual-Odometry
A mixture of traditional with new techniques for Visual Odometry and SLAM


~~Weird pipeline, not really working atm:~~

~~1. Feature matching (AKAZE).~~

~~2. 3D monocular estimation. 2D keypoints to 3D.~~

~~3. SolvePnP to get translation and rotation vector.~~

~~4. Add to initial position.~~

~~5. Repeat.~~


~~As said it needs a good debugging to make it work. Also a profound understanding of the matter.~~



## Kinda working now!

I cahanged the pipeline to a 3D-to-3D point correspondences. A lot slower and not very accurate but way better. Not quite working amazingly but... I kinda suck.

![](ezgif.com-video-to-gif.gif)


### Possible Reasons of my failure

- Scale factor problem but in a huge scale.
- 3D estimation is not consisten (is an estimation for god's sake!).
- My bad code.
