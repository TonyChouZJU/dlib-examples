//
// Created by akomandyr on 28.03.16.
//

#include "landmark_detector.h"

FaceLandmarkDetector::FaceLandmarkDetector(const std::string &shapePredictor)
{
    m_detector = dlib::get_frontal_face_detector();
    dlib::deserialize(shapePredictor.c_str()) >> m_shape_predictor;
};

//std::vector<dlib::point> FaceLandmarkDetector::DetectFaceLandmarks(const cv::Mat& src)
std::vector<dlib::point> FaceLandmarkDetector::DetectFaceLandmarks(const dlib::array2d<dlib::rgb_pixel>& src)
{
    //dlib::cv_image<dlib::bgr_pixel> img(src);
    //pyramid_up(img);
    std::vector<dlib::rectangle> dets = m_detector(src);
    //std::cout << "Number of faces detected: " << dets.size() << std::endl;

    dlib::full_object_detection shape = m_shape_predictor(src, dets[0]);
    //win.add_overlay(dlib::render_face_detections(shape));

    std::vector<dlib::point> points;
    points.reserve(shape.num_parts());

    for (unsigned long k = 0; k < shape.num_parts(); ++k)
    {
        points.push_back(shape.part(k));
    }

    std::cout <<"landmark detects: " << points.size() <<std::endl;
    return points;
}

FaceLandmarkDetector::~FaceLandmarkDetector()
{

}
