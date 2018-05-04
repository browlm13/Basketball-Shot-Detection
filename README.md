# Shot Detection

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



## Formats for representing video frame object bounding boxes detected

### Object Line Format (OL) CSV file
#### Each object's bounding box in a video recoreded on an individual line

| clip_ID | width | height | frame | category | score | x1 | x2 | y1 | y2 | model |
|---------|:-----:|:------:|:-----:|:--------:|:-----:|:--:|:--:|:--:|:--:|:-----:|
| int	  | int   | int    | int   | string   |float  |int |int |int |int |string |

### Mock 1 Frame Line Format (FLM1) CSV file
#### Each frame is represented on an individual line capturing only the highest score bounding box of each category detected
* All frames are represented exactly once
* Designed for videos containing at maximum a single _basketball_ and a single _person_
* Nan values are used with the absence of a detected  _basketball_ or _person_ in a frame

| clip_ID | width | height | frame | x1_basketball | x2_basketball | y1_basketball | y2_basketball | x1_person | x2_person | y1_person | y2_person | 
|---------|:-----:|:------:|:-----:|:-------------:|:-------------:|:-------------:|:-------------:|:---------:|:---------:|:----------:|:-----:|

### Mock 1 Basketball Tracking Format (BTM1) CSV file
#### Each line is an individual frame and contains the centerpoint coordinates of the highest scoring basketball detected as well as the radius and "free" column
* All frames are represented exactly once
* The free column is True if the highest scoring basketballs bounding box has no overlap with the highest scoring persons bounding box
* The radius is ``((x2 - x1) + (y2 - y1))/2``
* Nan values are used with the absence of a detected  _basketball_

| clip_ID | width | height | frame | x | y | radius | free |
|---------|:-----:|:------:|:-----:|:-:|:-:|:------:|:----:|
| int	  | int   | int    | int   |int|int| float  | bool |

### LabelImg Annotation Format (LI) XML file
#### [link to LabelImg](https://github.com/tzutalin/labelImg)
#### Each image has a corresponding LI.xml file containing all objects detected
* This is used to verify the accuracy of the models detections
* Multiple objects are possible for each image
```
 <annotation>
            <folder></folder>
            <filename></filename>
            <path></path>
            <source>
                <database></database>
            </source>
            <size>
                <width></width>
                <height></height>
                <depth></depth>
            </size>
            <segmented></segmented>
            <object>
                <name></name>
                <pose></pose>
                <truncated></truncated>
                <difficult></difficult>
                <bndbox>
                    <xmin></xmin>
                    <ymin></ymin>
                    <xmax></xmax>
                    <ymax></ymax>
                </bndbox>
            </object>
        </annotation>
```

### Converting between Object Line Format (OL) and LabelImg Annotation Format (LI)

| OL | LI |
|:--:|:--:|
|clip_ID|folder|
|frame|**file**|
|width|width|
|height|height|
|category|name|
|**score**| |
|x1|xmin|
|x2|xmax|
|y1|ymin|
|y2|ymax|
|**model**| |

* frame in this repository is the **file** name minus its extension
* **score** is ``100.0`` if annotated by a human
* **model** is "human" if annotated by a human

### Clip Info Bundel Format (CIB) JSON file
#### output of image_evaluator
```
{
	"PATH/TO/FRAME/IMAGE" : 

	{

		"image_path" 		: "PATH/TO/FRAME/IMAGE",
		"image_folder" 		: "IMAGE_FOLDER"
		"image_filename" 	: "IMAGE_FILENAME",
		"image_height" 		: HEIGHT_IN_PIXELS (int),
		"image_width" 		: WIDTH_IN_PIXELS (int),
		"image_items_list" : 

			[
				"category" : "NAME",
				"score" : ACCURACY_SCORE (float),
				"box" : [x1,x2,y1,y2] (ints),
				"model" : "EVALUATION_MODEL"
			]
	}
}
```

## Data Directory Structure
```
data
│
└───clips
│   |
│   | CLIP_ID1.mp4
|   | CLIP_ID2.mp4
│   | ...
|
└───verified_li_annotations
│   │
│   └───CLIP_ID1
|   |   │
|   |   └───frames
|   |   |   | 1.jpg
|   |   |   | 2.jpg
|   |   |   | ...
|   |   |
|   |   └───li_annotations
|   |   |   | 1.xml
|   |   |   | 2.xml
|   |   |   | ...
|   |   
│   └───CLIP_ID2
│   ...
│   
└───ol_annotations
    │   ol_annotations.csv
```
