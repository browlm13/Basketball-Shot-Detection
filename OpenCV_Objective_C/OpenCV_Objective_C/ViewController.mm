//
//  ViewController.m
//  OpenCV_Objective_C
//
//  Created by Zihao Mao on 11/9/17.
//  Copyright © 2017 Zihao Mao. All rights reserved.
//

#import "ViewController.h"


#include <iostream>
#include <stdlib.h>
using namespace std;
using namespace cv;

const Scalar RED = Scalar(0,0,255);
const Scalar GREEN = Scalar(0,255,0);


@interface ViewController ()
@property  UIImageView *imageView;
@property  UIImageView *liveView;

@property CvVideoCamera *videoCamera;

@property CascadeClassifier ball_cascade;
@property vector<cv::Rect> ball;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    int viewWidth = self.view.frame.size.width;
    int viewHeight = (352 * viewWidth)/288;
    int viewOffset = (self.view.frame.size.height - viewHeight)/2;
    _liveView = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, viewOffset, viewWidth, viewHeight)];
    

    [self.view addSubview:_liveView];
    
    _videoCamera = [[CvVideoCamera alloc] initWithParentView:_liveView];
    _videoCamera.delegate = self;
    
    _videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    _videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;
    _videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    
    self -> _videoCamera.rotateVideo = YES;
    
    NSString *ballCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_default"
                                                                ofType:@"xml"];
    const CFIndex CASCADE_NAME_LEN = 2048;
    char *CASCADE_NAME = (char *) malloc(CASCADE_NAME_LEN);
    CFStringGetFileSystemRepresentation((CFStringRef)ballCascadePath,CASCADE_NAME, CASCADE_NAME_LEN);
    
    if(!_ball_cascade.load(CASCADE_NAME)){
        cout << "Unable to load the Cascade!!!" << endl;
        exit(-1);
    }
    
    
    [_videoCamera start];
}

-(void)processImage:(Mat &)image;
{
    Mat gray; cvtColor(image,gray, CV_RGBA2GRAY);
    
    _ball_cascade.detectMultiScale(gray, _ball, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(50,50));
    
    if( _ball.size()>0 ){
        for (int i = 0; i < _ball.size(); i++){
            rectangle(image,_ball[i], RED);
        }
    }
    
    
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
