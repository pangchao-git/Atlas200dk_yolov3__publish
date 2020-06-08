#include "Main.h"
#include <sys/time.h>

#include <string>
#define kSleepInterval 200000
using namespace std;
# if 0
const static std::vector<string> label = {"background", "person", "bicycle", "car", "motorbike",
                                          "aeroplane","bus", "train", "truck", "boat",
                                          "traffic light", "fire hydrant", "stop sign", "parking meter",
                                          "bench", "bird", "cat", "dog", "horse",
                                          "sheep", "cow", "elephant", "bear", "zebra",
                                          "giraffe", "backpack", "umbrella", "handbag","tie",
                                          "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                                          "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                          "tennis racket", "bottle", "wine glass", "cup",
                                          "fork", "knife", "spoon", "bowl", "banana",
                                          "apple", "sandwich", "orange", "broccoli", "carrot",
                                          "hot dog", "pizza", "donut", "cake", "chair",
                                          "sofa", "potted plant", "bed", "dining table", "toilet",
                                          "TV monitor", "laptop", "mouse", "remote", "keyboard",
                                          "cell phone", "microwave", "oven", "toaster", "sink",
                                          "refrigerator", "book", "clock", "vase","scissors",
                                          "teddy bear", "hair drier", "toothbrush"};
#else
const static std::vector<string> label = {"background", "defect_product"};
#endif
void Print(std::vector<Bbox> &bbox){
    if(0==bbox.size()){
        return;
    }
    for(auto box: bbox){
        cout<<"score: "<<box.score<<" ,classname: "<<box.classname<<" Rect[ "<<box.Left_up.x<<", "<<box.Left_up.y
            <<", "<<box.Right_down.x<<", "<<box.Right_down.y<<"]"<<endl;
    }
}
void print_param(Input_param &param){
    cout<<"param:" <<param.model_width<<","<<param.model_height<<","<<param.image_path<<endl;
    cout<<"config:"<<param.yolov3_cfg_params.BoxTensorLabel<<","<<param.yolov3_cfg_params.numClasses<<endl;

}



int main(int argc, char *argv[]) {
    if(argc!=4){
        cout<<"param -model_w model_h image_path"<<endl;
        return -1;
    }
    struct  timeval  start;
    struct  timeval  start1;
    struct  timeval  start2;
    struct  timeval  end;
    unsigned long timer;

    std::shared_ptr<Atlasdk_process> atlas= nullptr;
    MAKE_SHARED_NO_THROW(atlas,Atlasdk_process);
    //1 Init
    HIAI_StatusT status = atlas->m_custom->Init_Atlas200dk();
    if(status != HIAI_OK){
        INFO_LOG("atlas->m_custom->Init_Atlas200dk()!,ret=%d\n", status);
        exit(-1);
    }

    std::vector<Bbox> bbox;//post_process out
    Input_param input_params;
    hiai::EnginePortID engine_id;

    //命令行参数
    input_params.model_width = std::atoi(argv[1]);
    input_params.model_height= std::atoi(argv[2]);
    input_params.save_path="./out";
    std::string image_path=argv[3];


    //设置yolov3后处理的配置项
    input_params.yolov3_cfg_params.label=label;
    input_params.yolov3_cfg_params.numClasses=label.size()-1;
    input_params.yolov3_cfg_params.BoxTensorLabel=input_params.yolov3_cfg_params.numClasses+5;

    //设置传输的engine_id
    engine_id.graph_id = atlas->m_custom->kGraphId;
    engine_id.engine_id = atlas->m_custom->kFirstEngineId;
    engine_id.port_id = atlas->m_custom->kFirstEnginePortId;

    //2 set call back,and infer data save in atlas->m_custom->m_enginetrans
    int loop_count = 5;
    gettimeofday(&start,NULL);
    for(int i=0;i<loop_count;i++)
    {

        input_params.image_path=image_path+std::to_string(i)+".jpg";
        printf("Running count is  image is %s %d/%d\n",input_params.image_path.c_str(), i+1,loop_count);
        atlas->m_custom->SetCallBack(atlas->m_custom);
        bbox.clear();
       // usleep(kSleepInterval);
        atlas->m_custom->SetGrgument(engine_id,input_params);
        atlas->m_custom->GetData(bbox);

    }
    gettimeofday(&end,NULL);
    timer = (1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec)/(1000*loop_count);
    printf("yolov3 detect cost avg 100 counts time is   %ld ms\n",timer);
    Print(bbox);



    //4 Destory
    status=atlas->m_custom->Destory();
    if (status != HIAI_OK) {
        ERROR_LOG("Failed to call atlas->m_custom->Destory(), ret=%d.", status);
        return kHiaiInitFailed;
    }
    cout<<"end!"<<endl;
    return 0;
}
