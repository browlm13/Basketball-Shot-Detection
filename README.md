# Shot Detection
### L.J. Brown, Zihao Mao, Preston Deason, Austin Wiedebusch
## AI Basketball Dhot Detection and Analysis
Our program is able to detect when a shot occurs and extrapolate the balls flight from captured data. The program calculates the balls initial velocity and launch angle. The program is able to determine the balls flight perpedicular to the camera plane (The z axis). The program is also able to detect when the balls flight is interupted by another object and will drop those data points. In the case of unstable video, the program currently calculates the balls trajectory relative to the person shooting the ball. In the future we will impliment more accurate stabilization techniques.

## Tracking in Unstable Video
![Unstable Video](shot_1.gif)
### World Coordinates
![world coordinates](shot_1_trajectory_extrapolation_points_v1.png)

## Trajectory Extrapolation
### Tolerant to Missing Datapoints
![Trajectory](shot_2.png)
![Shot with missing datapoints](shot_2.gif)
### World Coordinates
![world coordinates](shot_2_trajectory_extrapolation_points_v1.png)
## End of Shot detection 
### (Exclude Data Points After Ball Hits Objects)
![Hits net piecewise linear regression](shot_16.gif)
### (Adjust Trajectories For Shot Angle/Camera Depth)
![Depth Adjustment](depth_adjustment_shot_16.png)
### World Coordinates
![world coordinates](shot_16_trajectory_extrapolation_points_v1.png)
