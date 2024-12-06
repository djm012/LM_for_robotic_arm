import requests
import base64
import cv2

# 测试视觉定位功能
def test_vg():
    url = "http://172.18.32.132:5000/vg"

    # url = "http://172.18.32.132:5001/vg"
    para_data = {}

    # para_data["text"] = "the black umbrella"
    # para_data["text"] = "the red umbrella"
    # para_data["text"] = "盒子"
    para_data["text"] = "television"

    # img = cv2.imread("sample/umbrella_12.png")
    # img = cv2.imread("sample/garbage_15.png")
    # img = cv2.imread("sample/trash can_23.png")
    # img = cv2.imread("sample/rgsofa_10.png")
    # img = cv2.imread("sample/chair_9.jpg")
    # img = cv2.imread("sample/carton_8.jpg")
    img = cv2.imread("sample/22sf.jpg")

    file = {"file": ("file_name.jpg", cv2.imencode(".jpg", img)[1].tobytes(), "image/jpg")}
    res = requests.post(url=url, files=file, data=para_data)
    res_data = res.json()

    if res_data:
        print(res_data) # dict: {'xmax': 689.3292846679688, 'xmin': 452.29229736328125, 'ymax': 503.06304931640625, 'ymin': 315.6756591796875}
    else:
        print("没有定位到目标") # res_data:None

# 测试垃圾分类功能
def test_garbage_cls():
    url = "http://172.18.32.132:5000/garbage_cls"
    # url = "http://172.18.32.132:5001/garbage_cls"

    img = cv2.imread("sample/222.jpg")

    file = {"file": ("file_name.jpg", cv2.imencode(".jpg", img)[1].tobytes(), "image/jpg")}
    res = requests.post(url=url, files=file)
    res_data = res.json()
    print(res_data)

# 测试检测模型
def test_damoyyolos():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    # object_detect = pipeline(Tasks.image_object_detection,model='damo/cv_tinynas_object-detection_damoyolo')
    object_detect = pipeline(Tasks.image_object_detection,model='damo/cv_tinynas_object-detection_damoyolo')
    pipeline('image-object-detection', 'damo/cv_tinynas_object-detection_damoyolo-m')
    img_path ='sample/zhiwu1.jpg'
    result = object_detect(img_path)
    print("++++")
    print(result)

# 测试分割模型
def test_u2net_salient_detection():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.outputs import OutputKeys
    import cv2
    salient_detect = pipeline(Tasks.semantic_segmentation, model='damo/cv_u2net_salient-detection')
    img_path ='sample/55.png'
    result = salient_detect(img_path)

    cv2.imwrite('./result.jpg',result[OutputKeys.MASKS])

    # 绘制连通区域
    gray_img = result[OutputKeys.MASKS]
    ret, th = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    gray_img = th
    # 搜索图像中的连通区域
    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(th)
    for i, stat in enumerate(stats):
        #绘制连通区域
        cv2.rectangle(gray_img, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (25, 25, 255), 3)
        #按照连通区域的索引来打上标签
        cv2.putText(gray_img, str(i+1), (stat[0], stat[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
    cv2.imwrite('./result111111.jpg',gray_img)



if __name__ == "__main__":
    test_vg()
    # test_garbage_cls()
    # test_damoyyolos()
    # test_u2net_salient_detection()
