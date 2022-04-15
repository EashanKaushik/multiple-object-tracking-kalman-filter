# multible-object-tracking-kalman-filter

## Object Detection - YOLO V4

Since this project revolves around object tracking, I have used a pre-trained network for object detection in each frame of the video. Using the result from YOLO a bounding box was formed around the detected object and centroid (Ground Truth) was calculated. This tells us the true location of the object when it is detected. However, YOLO does not track the object, this means it does not know anything about the trajectory of the object and can't predict its motion. This is where Kalman Filter can be used.

## Object Tracking - Kalman Filter

Kalman Filter helps us to track the object even when it is not detected by YOLO. In this repository we have generated the output of two scenarios: 

1. Tracking object when it is hidden behind some other object (Single Object with Occlusion).
2. Tracking object when it is hidden behind the same object (Multiple Object with Occlusion). 

## Understanding the Problem Statement
  
### Single Object with Occlusion 

In this scenario, we have an object, which in the middle of the clip is being occluded by a piece of paper. When this happens YOLO will know nothing about the object and once it reappears it will treat it as a new object being detected. This means: 
- We are not assigning any unique id to the object to identify it once it reappears.
- We know anything about the object's trajectory when it is being occluded by a piece of paper. 

Point two can be solved by Kalman Filter easily, we have the following formulas which can be used to implement Kalman Filter. 

- Predict

      X_t = F * X_t-1 + B * u # Predicted (a priori) state estimate
      P_t = F * P_t-1 * F.T + Q # Predicted (a priori) estimate covariance
      
- Update

      y = z - H * X_t # Innovation or measurement pre-fit residual
      S = H * P_t * H.T + R # Innovation (or pre-fit residual) covariance
      K = P_t * H.T * inv(S) # Optimal Kalman gain
      X = X_t + k * y # Updated (a posteriori) state estimate
      P = (I - K * H) * P_t # Updated (a posteriori) estimate covariance

<p align="center">
  <img src="https://github.com/EashanKaushik/multiple-object-tracking-kalman-filter/blob/main/markdown/single_input.gif" />
</p>

### Multiple Object with Occlusion

In this scenario we have two objects, object on the right (Object 1) is being occluded by the object on the left (Object 2). When this happens YOLO will not be able to distinguish between the two objects and once object 1 reappears it willl treat it as a new object. 

This can be solved by assigning a Kalman Filter with a unique id to each object being detected. This can be done by comparing the centroid of the object from the centroid of the object in previous frame, and assigning the Kalman filter to the closet object (w.r.t its centroid). 

<p align="center">
  <img src="https://github.com/EashanKaushik/multiple-object-tracking-kalman-filter/blob/main/markdown/multiple_input.gif" />
</p>

## Output

### Single Object with Occlusion 
<p align="center">
  <img src="https://github.com/EashanKaushik/multiple-object-tracking-kalman-filter/blob/main/markdown/single_output.gif" />
</p>

### Multiple Object with Occlusion 
<p align="center">
  <img src="https://github.com/EashanKaushik/multiple-object-tracking-kalman-filter/blob/main/markdown/multiple_output.gif" />
</p>
