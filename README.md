# Shot Detection
### L.J. Brown, Zihao Mao, Preston Deason, Austin Wiedebusch
## AI Basketball Shot Detection and Analysis
Our program is able to detect when a shot occurs and extrapolate the balls flight from captured data. The program calculates the balls initial velocity and launch angle. The program is able to determine the balls flight perpedicular to the camera plane (The z axis). The program is also able to detect when the balls flight is interupted by another object and will drop those data points. In the case of unstable video, the program currently calculates the balls trajectory relative to the person shooting the ball. In the future we will impliment more accurate stabilization techniques. Additional note: the program currently requires at least 2 data points of a shot to be captured to perform its anylisis.

## Tracking and anylisis performed on 
* unstable video

![Unstable Video](shot_1.gif)
#### Program output world coordinates:
![world coordinates](shot_1_trajectory_extrapolation_points_v1.png)

#### Tracking and anylisis performed on 
* shot interrupted by person

![Shot with missing datapoints](shot_2.gif)
#### Program output world coordinates:
![world coordinates](shot_2_trajectory_extrapolation_points_v1.png)

### Tracking and anylisis performed on 
* shot interrupted by object
* shot angled with component perpendicular to the camera plane

![Hits net piecewise linear regression](shot_16.gif)
#### Program output world coordinates:
![world coordinates](shot_16_trajectory_extrapolation_points_v1.png)
