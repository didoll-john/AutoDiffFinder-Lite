import cv2
import numpy as np
import gradio as gr

def align_and_crop_images(img1, img2):
    """
    使用SIFT特征匹配对齐两张图片
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 创建SIFT特征检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # 创建特征匹配器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # 应用比率测试来筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        return center_crop_images(img1, img2)
    
    # 获取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 对齐图片
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 计算变换后的图片尺寸
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners1, H)
    
    # 计算变换后图片的边界
    x_min = min(transformed_corners[:, 0, 0].min(), 0)
    y_min = min(transformed_corners[:, 0, 1].min(), 0)
    x_max = max(transformed_corners[:, 0, 0].max(), w2)
    y_max = max(transformed_corners[:, 0, 1].max(), h2)
    
    # 调整变换矩阵以确保所有内容都在视图中
    translation_matrix = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    H = translation_matrix.dot(H)
    
    # 执行透视变换
    output_size = (int(x_max - x_min), int(y_max - y_min))
    aligned_img1 = cv2.warpPerspective(img1, H, output_size)
    aligned_img2 = cv2.warpPerspective(img2, np.dot(translation_matrix, np.eye(3)), output_size)
    
    # 裁剪到相同大小
    min_h = min(aligned_img1.shape[0], aligned_img2.shape[0])
    min_w = min(aligned_img1.shape[1], aligned_img2.shape[1])
    
    aligned_img1 = aligned_img1[:min_h, :min_w]
    aligned_img2 = aligned_img2[:min_h, :min_w]
    
    return aligned_img1, aligned_img2

def center_crop_images(img1, img2):
    """
    当特征匹配失败时的备用方案：简单的居中裁剪
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    target_height = min(h1, h2)
    target_width = min(w1, w2)
    
    start_y1 = (h1 - target_height) // 2
    start_x1 = (w1 - target_width) // 2
    start_y2 = (h2 - target_height) // 2
    start_x2 = (w2 - target_width) // 2
    
    cropped1 = img1[start_y1:start_y1+target_height, start_x1:start_x1+target_width]
    cropped2 = img2[start_y2:start_y2+target_height, start_x2:start_x2+target_width]
    
    return cropped1, cropped2

def split_image_from_array(image):
    """
    从图像数组分割图片
    """
    if image is None:
        raise ValueError("无效的图像数据")

    height = image.shape[0]
    middle_y = height // 2
    
    # 提取上下图片
    offset = 10  # 设置较小的偏移量
    upper_image = image[0:middle_y-offset, :]
    lower_image = image[middle_y+offset:height, :]
    
    # 对齐和裁剪至相同大小
    upper_image, lower_image = align_and_crop_images(upper_image, lower_image)
    
    return upper_image, lower_image

def process_image(input_image):
    """
    处理上传的图片并返回增强后的差异图
    """
    # 将输入图片从RGB转换为BGR (因为Gradio使用RGB，而OpenCV使用BGR)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # 分割图片
    upper_image, lower_image = split_image_from_array(input_image)
    
    # 计算差异
    diff = cv2.absdiff(upper_image, lower_image)
    enhanced_diff = cv2.multiply(diff, 3.5)
    
    # 将结果转换回RGB用于Gradio显示
    enhanced_diff_rgb = cv2.cvtColor(enhanced_diff, cv2.COLOR_BGR2RGB)
    
    return enhanced_diff_rgb

# 创建Gradio界面
def create_ui():
    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(),
        outputs=gr.Image(label="差异增强图"),
        title="找不同游戏分析器",
        description="上传一张找不同游戏的图片，系统将自动分析并显示差异部分。",
        examples=["Examples/example.jpg"]
    )
    return interface

# 启动Gradio应用
if __name__ == "__main__":
    ui = create_ui()
    ui.launch(share=False)

