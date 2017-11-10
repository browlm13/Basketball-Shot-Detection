//
//  ViewController.h
//  OpenCV_Objective_C
//
//  Created by Zihao Mao on 11/9/17.
//  Copyright © 2017 Zihao Mao. All rights reserved.
//

#import <UIKit/UIKit.h>

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import "opencv2/highgui/ios.h"
#endif


@interface ViewController : UIViewController<CvVideoCameraDelegate>


@end

