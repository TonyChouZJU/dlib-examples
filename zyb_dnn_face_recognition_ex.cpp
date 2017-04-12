// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  In it, we will show how to do face recognition.  This example uses the
    pretrained dlib_face_recognition_resnet_model_v1 model which is freely available from
    the dlib web site.  This model has a 99.38% accuracy on the standard LFW face
    recognition benchmark, which is comparable to other state-of-the-art methods for face
    recognition as of February 2017. 
    
    In this example, we will use dlib to do face clustering.  Included in the examples
    folder is an image, bald_guys.jpg, which contains a bunch of photos of action movie
    stars Vin Diesel, The Rock, Jason Statham, and Bruce Willis.   We will use dlib to
    automatically find their faces in the image and then to automatically determine how
    many people there are (4 in this case) as well as which faces belong to each person.
    
    Finally, this example uses a network with the loss_metric loss.  Therefore, if you want
    to learn how to train your own models, or to get a general introduction to this loss
    layer, you should read the dnn_metric_learning_ex.cpp and
    dnn_metric_learning_on_images_ex.cpp examples.
*/

#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, the jittering you can see below in jitter_image() was used during
// training, and the training dataset consisted of about 3 million images instead of 55.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "Run this example by invoking it like this: " << endl;
        cout << "   ./dnn_face_recognition_ex faces/bald_guys.jpg" << endl;
        cout << endl;
        cout << "You will also need to get the face landmarking model file as well as " << endl;
        cout << "the face recognition model file.  Download and then decompress these files from: " << endl;
        cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" << endl;
        cout << endl;
        return 1;
    }

    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    // Display the raw image on the screen
    //image_window win(img); 

    std::ifstream train_first_file("/home/zyb/py-faster-rcnn/faces_examples/first_match_path.txt", ios::in);
    std::ifstream train_second_file("/home/zyb/py-faster-rcnn/faces_examples/second_match_path.txt", ios::in);
    string s;
    std::vector<string> pair_first_file_vec;
    while(getline(train_first_file, s)) {
        string::size_type position = s.find_last_of(" ");
        if(position == s.npos)
            std::cout <<"not found"<< std::endl;
        else{
            pair_first_file_vec.push_back(s.substr(0, position));
        }
    }
    std::vector<string> pair_second_file_vec;
    while(getline(train_second_file, s)) {
        string::size_type position = s.find_last_of(" ");
        if(position == s.npos)
            std::cout <<"not found"<< std::endl;
        else{
            pair_second_file_vec.push_back(s.substr(0, position));
        }
    }
    std::cout <<"load train file list" <<std::endl;


    std::vector<matrix<rgb_pixel>> faces_all1;
    std::vector<matrix<rgb_pixel>> faces_all2;
    std::vector<string> label_all;
    for (int i= 0; i < pair_first_file_vec.size(); i++) {

        std::vector<matrix<rgb_pixel>> faces1;
        matrix<rgb_pixel> img1;
        load_image(img1, pair_first_file_vec[i]);

        std::vector<matrix<rgb_pixel>> faces2;
        matrix<rgb_pixel> img2;
        load_image(img2, pair_second_file_vec[i]);


        auto face_vector1 = detector(img1);
        auto face_vector2 = detector(img2);
        if (face_vector1.size() ==0|| face_vector2.size()==0){
            cout << "No faces found in image!" << endl;
            continue;
        }
        auto shape1 = sp(img1, face_vector1[0]);
        matrix<rgb_pixel> face_chip1;
        extract_image_chip(img1, get_face_chip_details(shape1,150,0.25), face_chip1);
        
        auto shape2 = sp(img2, face_vector2[0]);
        matrix<rgb_pixel> face_chip2;
        extract_image_chip(img2, get_face_chip_details(shape2,150,0.25), face_chip2);

        faces_all1.push_back(face_chip1);
        faces_all2.push_back(face_chip2);
        if(i>=500)
            label_all.push_back(string("0"));
        else 
            label_all.push_back(string("1"));

    }
    std::cout <<"pairs1 size:"<<faces_all1.size() <<std::endl;

    std::vector<matrix<float,0,1>> face_descriptors1 = net(faces_all1);
    std::vector<matrix<float,0,1>> face_descriptors2 = net(faces_all2);

    string score_file("/home/zyb/py-faster-rcnn/faces_examples/dlib_scores.txt");
    std::ofstream fout(score_file, std::ios::out);
    for (size_t i = 0; i < faces_all1.size(); i++)
    {
            std::cout << face_descriptors1[i] <<std::endl;
            float score = length(face_descriptors1[i]-face_descriptors2[i])/ (length(face_descriptors1[i])* length(face_descriptors2[i]));
            fout << score << " " + label_all[i]+"\n";
    }
    fout.close();

}

catch (std::exception& e)
{
    cout << e.what() << endl;
}
