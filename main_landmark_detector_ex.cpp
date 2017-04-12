//
// Created by akomandyr on 28.03.16.
//

//#include "landmark_detector.h"
#include "face_morpher.h"
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
//#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
//#include <dlib/image_io.h>
//#include <iostream>

//using namespace dlib;
using namespace std;
int main(int argc, char** argv)
{

    try
    {

        if (argc != 5)
        {
            cout << "Call this program like this:" << endl;
            cout << "./main_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces1.jpg faces2.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }

        //frontal_face_detector detector = get_frontal_face_detector();
    	//FaceLandmarkDetectorPtr m_landmarkDetector = std::make_shared<FaceLandmarkDetector>(argv[1]);
        FaceMorpherPtr morpherPtr = std::make_shared<FaceMorpher>(argv[1]);
        morpherPtr->MorphFace(argv[2], argv[3], argv[4]);
    	//FaceLandmarkDetector m_landmarkDetector = FaceLandmarkDetector(argv[1]);
        //shape_predictor sp;
        //deserialize(argv[1]) >> sp;

        //image_window win, win_faces;


	    //array2d<rgb_pixel> img;
	    //load_image(img, argv[2]);
        //cv::Mat src = cv::imread(argv[2]);
        //dlib::array2d<rgb_pixel> img;
        //dlib::assign_image(img, dlib::cv_image<bgr_pixel>(src));

        //pyramid_up(img);
        //win.clear_overlay();
        //win.set_image(img);
        //std::vector<dlib::point> src1_points = m_landmarkDetector->DetectFaceLandmarks(img);
        //std::cout <<"landmark points:"<< src1_points.size()<<std::endl;

        //std::vector<rectangle> dets = detector(img);
        //std::vector<full_object_detection> shapes;
        /*
        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            full_object_detection shape = sp(img, dets[j]);
            cout << "number of parts: "<< shape.num_parts() << endl;
            cout << "pixel position of first part:  " << shape.part(0) << endl;
            cout << "pixel position of second part: " << shape.part(1) << endl;
            shapes.push_back(shape);
        }

        win.clear_overlay();
        win.set_image(img);
        win.add_overlay(render_face_detections(shapes));
        dlib::array<array2d<rgb_pixel> > face_chips;
        extract_image_chips(img, get_face_chip_details(shapes), face_chips);
        win_faces.set_image(tile_images(face_chips));

        */
        cout << "Hit enter to process the next image..." << endl;
        cin.get();

    }
    catch (std::exception& e)
    {
        std::cout << "\exception thrown!"<< std::endl;
        std::cout <<e.what() <<std::endl;

    }

}
