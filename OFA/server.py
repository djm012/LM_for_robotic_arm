from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

# 导入Flask类库
from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import json
import time
from datetime import datetime
import requests

# 创建应用实例
app = Flask(__name__)

ts1 = time.time()
# 二、视觉定位模型
# large-通用场景-英文 OFA模型
ofa_pipe = pipeline(
    Tasks.visual_grounding,
    model='damo/ofa_visual-grounding_refcoco_large_en',
    device="gpu:0"
    )
# 三、图像分类模型
image_classification = pipeline(
    Tasks.image_classification,
    model='damo/cv_beitv2-large_image-classification_patch16_224_pt1k_ft22k_in1k',
    device="gpu:0")
# 四、垃圾分类模型
garbage_classification = pipeline(
    Tasks.image_classification,
    model='damo/cv_convnext-base_image-classification_garbage',
    device="gpu:0")
# 五、分割模型
salient_detect = pipeline(
    Tasks.semantic_segmentation, 
    model='damo/cv_u2net_salient-detection',
    device="gpu:0")
ts2 = time.time()
print("【加载定位、图像分类、垃圾分类模型、分割模型用时】: ", ts2-ts1)

# 汉译英
def zh_to_en(zh_str):
    token = '24.3c5c278934cf7d503809ae3cb6b0182f.2592000.1693546593.282335-37062527'
    url = 'https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=' + token

    # For list of language codes, please refer to `https://ai.baidu.com/ai-doc/MT/4kqryjku9#语种列表`
    from_lang = 'zh' # example: en
    to_lang = 'en' # example: zh
    term_ids = 'oywe22je5k' # 术语库id，多个逗号隔开

    # Build request
    headers = {'Content-Type': 'application/json'}
    payload = {'q': zh_str, 'from': from_lang, 'to': to_lang, 'termIds' : term_ids}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    print("+\n"*3)
    print(result)

    # Show response
    # print(json.dumps(result, indent=4, ensure_ascii=False))
    en_str = result["result"]["trans_result"][0]["dst"]

    return en_str.lower()


# 判断视觉定位结果的可信度
def is_vg_reliable(imgclass_result, text):
    text_array = text.split()
    threshold = 0.35
    top_n_index = [0,1,2,3,4]
    img_labels = imgclass_result["labels"]
    img_scores = imgclass_result["scores"]
    for index in top_n_index:
        label = img_labels[index]
        score = img_scores[index]
        if score >= threshold:
#            for keyword in text_array:
#                if keyword in label:
                    return True
    return False


# 存储模型运行结果
def store_model_result(input_text, img_file_path, ofa_result, imgclass_result, is_vg_reliable):
    result_filepath = img_file_path.replace(".jpg", "_%s.json"%str(is_vg_reliable).lower())
    imgclass_result["scores"] = [x.item() for x in imgclass_result["scores"]]

    result = {"input_text":input_text,
              "ofa_result":ofa_result,
              "imgclass_result":imgclass_result}
    with open(result_filepath, 'w') as fh:
        json.dump(result, fh, indent=4, ensure_ascii=False)
    

# 视图函数（路由）
@app.route("/")
def hello_world():
    return "Hello, World!"

# 视图函数（路由）
@app.route("/vg", methods=['POST'])
def visual_grounding():
    input_text = request.form.get("text")
    print("【输入指令】:", input_text)
    input_text = input_text.replace("请","")
    input_text = input_text.replace("去","")
    input_text = input_text.replace("附近","")
    input_text = input_text.replace("清扫","")
    input_text = input_text.replace("打扫","")
    input_text = input_text.replace("一下","")
    print("【视觉定位指令】:", input_text)
    # text = zh_to_en(input_text)
    text = input_text
    print("【指令翻译】:", text)

    current_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    img_file_name = "%s_%s.jpg" % (current_dt, int(round(time.time() * 1000))%1000)
    img_file_path = "tmp/%s" % img_file_name
    img_file = request.files["file"]
    img_file.save(img_file_path)
    print("【图像文件路径】: ", img_file_path)
    ts3 = time.time()

    # 执行视觉定位模型
    input = {'image': img_file_path, 'text': text}
    ofa_result = ofa_pipe(input)
    ts4 = time.time()
    print("【视觉定位用时】: ", ts4-ts3)

    # 截取、标记视觉定位区域
    xmin, ymin, xmax, ymax = ofa_result[OutputKeys.BOXES][0]
    img = cv2.imread(img_file_path)
    img_ymax, img_xmax =  img.shape[0], img.shape[1]
    print("【视觉定位结果】: ", ofa_result[OutputKeys.BOXES][0])
    if xmin < 5 or xmax > (1280-5):
        print("【视觉定位失败】:物品还没有完全进入视野")
        return jsonify()

    rect_ymin,rect_ymax,rect_xmin,rect_xmax = int(ymin)-5,int(ymax)+5,int(xmin)-5,int(xmax)+5
    rect_ymin = 0 if rect_ymin<0 else rect_ymin
    rect_ymax = img_ymax if rect_ymax>img_ymax else rect_ymax
    rect_xmin = 0 if rect_xmin<0 else rect_xmin
    rect_xmax = img_xmax if rect_xmax>img_xmax else rect_xmax
    img_rect = img[rect_ymin:rect_ymax, rect_xmin:rect_xmax]
    
    if img_rect.size == 0:
        print("【视觉定位失败】:img_rect.size==0")
        return jsonify()
    img_rect_file_path = "tmp/%s" % img_file_name.replace(".jpg", "_rect.jpg")
    cv2.imwrite(img_rect_file_path, img_rect)

    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1)  
    cv2.imwrite(img_file_path, img)

    # 执行图像分类模型
    imgclass_result = image_classification(img_rect_file_path)
    ts5 = time.time()
    print("【图像分类用时】: ", ts5-ts4)

    # 确认定位结果的可靠性
    vg_flag = is_vg_reliable(imgclass_result, text) # True 定位成功，False 定位失败
    print("【视觉定位结果】: ", vg_flag)

    # 存储模型运行结果
    store_model_result(input_text, img_file_path, ofa_result, imgclass_result, vg_flag)

    if vg_flag:
        return jsonify(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    else:
        return jsonify()


# 视图函数（路由）
@app.route("/garbage_cls", methods=['POST'])
def garbage_cls():
    current_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    img_file_name = "%s_%s_gb.jpg" % (current_dt, int(round(time.time() * 1000))%1000)
    img_file_path = "tmp/%s" % img_file_name
    img_file = request.files["file"]
    img_file.save(img_file_path)
    print("【垃圾图像文件路径】: ", img_file_path)
    img = cv2.imread(img_file_path)
    ts1 = time.time()

    # 执行图像分割模型,并存储分割结果
    detect_result = salient_detect(img_file_path)
    detect_result_filepath = img_file_path.replace(".jpg", "_det.jpg")
    cv2.imwrite(detect_result_filepath, detect_result[OutputKeys.MASKS])

    # 绘制连通区域
    gray_img = detect_result[OutputKeys.MASKS]
    ret, th = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    # 搜索图像中的连通区域,获取图像最下方的连通域
    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(th)
    stat = stats[-1]
    xmin, ymin = stat[0], stat[1]
    xmax, ymax = stat[0] + stat[2], stat[1] + stat[3]
    # 存储最下方连通域的图片
    img_rect = img[ymin:ymax, xmin:xmax]
    detect_rect_result_filepath = img_file_path.replace(".jpg", "_det_rect.jpg")
    cv2.imwrite(detect_rect_result_filepath, img_rect)
    
    # 对挑选出的连通域，执行垃圾分类模型
    result = garbage_classification(detect_rect_result_filepath)
    # result: {'scores': [0.7597316, 0.22564557, 0.006449062, 0.0019648746, 0.0018799941], 'labels': ['可回收物-卡片', '可回收物-地铁票', '可回收物-登机牌', '可回收物-不锈钢制品', '其他垃圾-车票']}
    result_filepath = detect_rect_result_filepath.replace(".jpg", ".json")
    result["scores"] = [x.item() for x in result["scores"]]
    with open(result_filepath, 'w') as fh:
        json.dump(result, fh, indent=4, ensure_ascii=False)

    result_label = result["labels"][0]
    result_label = result_label.split("-")[1]

    ts2 = time.time()
    print("【垃圾分类用时】: ", ts2-ts1)
    print("【垃圾分类结果】: ", result)
    return jsonify(label=result_label, xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax))

    



# 启动服务
# 后台运行命令 nohup python -u server.py > out_server.log 2>&1 &
# 查看任务运行状态的命令 ps aux|grep server.py


if __name__ == '__main__':
    # production
    app.run(debug = False, host="172.18.32.132", port=5000)

    # development
    # app.run(debug = True, host="172.18.32.132", port=5001)


