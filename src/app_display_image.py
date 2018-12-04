import os
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt 
import numpy as np
import model
import time
import os  

__author__ = 'ibininja'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        if filename == "grant.jpg" :
            return render_template("boss.html")
        ai_result = evaluate_one_image(destination)

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete_display_image.html", image_name=filename, ai_result=ai_result)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

def evaluate_one_image(filepath):  

    image = Image.open(filepath)  
    # plt.ion()
    # plt.imshow(image)
    # plt.pause(5)  #显示秒数
    # plt.close()
    # #plt.show()

    image = image.resize([208, 208])  
    image_array = np.array(image)
      
    with tf.Graph().as_default():  
        BATCH_SIZE = 1  # 因为只读取一副图片 所以batch 设置为1
        N_CLASSES = 2  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
        # 转化图片格式
        image = tf.cast(image_array, tf.float32)  
        # 图片标准化
        image = tf.image.per_image_standardization(image)

        # 图片原来是三维的 [208, 208, 3] 重新定义图片形状 改为一个4D  四维的 tensor
        image = tf.reshape(image, [1, 208, 208, 3])  
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)  
        # 因为 inference 的返回没有用激活函数，所以在这里对结果用softmax 激活
        logit = tf.nn.softmax(logit)  

        # 用最原始的输入数据的方式向模型输入数据 placeholder
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])  

        # 我门存放模型的路径
        logs_train_dir = './saveNet/'   
        # 定义saver 
        saver = tf.train.Saver()  

        with tf.Session() as sess:  
              
            print("从指定的路径中加载模型。。。。")
            # 将模型加载到sess 中 
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)  
            if ckpt and ckpt.model_checkpoint_path:  
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  
                saver.restore(sess, ckpt.model_checkpoint_path)  
                print('模型加载成功, 训练的步数为 %s' % global_step)  
            else:  
                print('模型加载失败，，，文件没有找到')  
            # 将图片输入到模型计算
            prediction = sess.run(logit, feed_dict={x: image_array})
            # 获取输出结果中最大概率的索引
            max_index = np.argmax(prediction)  
            
            if max_index==0:  
                result = '猫的概率 %.6f' %prediction[:, 0]
                print('猫的概率 %.6f' %prediction[:, 0])
                
            else:  
                print('狗的概率 %.6f' %prediction[:, 1]) 
                result = '狗的概率 %.6f' %prediction[:, 1]
    return result    




if __name__ == "__main__":
    app.run(host='0.0.0.0',port=4555, debug=True)
