import streamlit as st
import torch
import numpy as np
from PIL import Image
from blazeface import BlazeFace, FaceExtractor, VideoReader
from architectures import fornet, weights
from isplutils import utils
from torch.utils.model_zoo import load_url
import tempfile  # 임시 파일 처리를 위한 모듈
from scipy.special import expit
import matplotlib.pyplot as plt
import os 
import cv2


# Configurations
org_dir_path = '/home/ubuntu/workspace/kwanwoo/capstone'
net_model = 'EfficientNetAutoAttB4'
train_db = 'DFDC'
device = torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32


# 배경 색상 설정


def styled_progress_bar(value):
    """Styled progress bar with custom color."""
    st.markdown(
        f"""
        
        <style>
         .rating-label {{
                color: #FFFFFF;
                text-align: center;
                margin-top: 20px;
            }}
        .progress-container {{
            width: 100%;
            background-color: #e0e0e0; /* 배경 색상 */
            border-radius: 10px; /* 모서리 둥글게 */
            overflow: hidden; /* 넘치는 내용 숨기기 */
        }}
        .progress-bar {{
            height: 20px; /* 바 높이 */
            width: {value * 100}%; /* 진행률에 따라 너비 설정 */
            background-color: #ff0000; /* 프로그레스 바 색상 */
            text-align: center; /* 텍스트 가운데 정렬 */
            line-height: 20px; /* 텍스트 높이 설정 */
            color: white; /* 텍스트 색상 */
            border-radius: 10px 0 0 10px; /* 모서리 둥글게 */
        }}
        </style>
        <div class="progress-container">
            <div class="progress-bar">{int(value * 100)}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )





@st.cache_resource
def initialize_resources():
    """Initialize and load the model and BlazeFace detector."""
    # Load model
    model_url = weights.weight_url[f'{net_model}_{train_db}']
    net = getattr(fornet, net_model)().eval()
    net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))

    # Load BlazeFace detector
    facedet = BlazeFace()
    facedet.load_weights("blazeface/blazeface.pth")
    facedet.load_anchors("blazeface/anchors.npy")

    # Transformer for face preprocessing
    transformer = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

    return net, facedet, transformer

# Load resources at the start of the app
net, facedet, transformer = initialize_resources()

def extract_faces_from_video(facedet):
    """Extract faces from the video using BlazeFace."""
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)
    return face_extractor

def preprocess_faces(vid_faces):
    """Preprocess the extracted faces for model prediction."""
    return torch.stack([
        transformer(image=frame['faces'][0])['image']
        for frame in vid_faces if len(frame['faces'])
    ])

def predict_faces(net, faces_t):
    """Predict whether the faces are real or fake."""
    import time
    start_time = time.time()
    with torch.no_grad():
        predictions = net(faces_t).cpu().numpy().flatten()
    end_time = time.time()
    print(end_time - start_time)
    return predictions

def make_fig(vid_name,vid_real_faces,faces_real_pred):
    fig,ax = plt.subplots(figsize=(8,5))
    fig.patch.set_facecolor('#3C3D37')
    ax.set_facecolor('#3C3D37')
    ax.stem([f['frame_idx'] for f in vid_real_faces if len(f['faces'])],expit(faces_real_pred),linefmt='r', markerfmt='ro')
    ax.set_title(vid_name)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Score')
    ax.set_ylim([0,1])
    ax.grid(True)
    return fig

def make_video(temp_path):

    # OpenCV로 동영상 읽기
    cap = cv2.VideoCapture(temp_path)
    
    # 임시로 처리된 동영상을 저장할 경로
    output_path = 'temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # mp4 코덱 설정

    # 원본 동영상의 너비, 높이, FPS 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter 객체 생성 (출력 비디오 설정)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 동영상 프레임 처리
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 텍스트 추가 (프레임 중앙 하단에 'Deepfake Detection' 표시)
        cv2.putText(
            frame, 
            "Deepfake Detection", 
            (50, 50),  # 위치 (좌측 상단)
            cv2.FONT_HERSHEY_SIMPLEX,  # 폰트
            1,  # 폰트 크기
            (0, 0, 255),  # 색상 (BGR: 빨간색)
            2,  # 두께
            cv2.LINE_AA  # 선 타입
        )

        # 처리된 프레임을 출력 비디오에 저장
        out.write(frame)

    cap.release()
    out.release()

    return output_path

def main():
    
    st.title("딥페이크 탐지 데모")
    st.write("동영상을 업로드해 딥페이크 여부를 확인합니다.")

    # File uploader
    uploaded_file = st.file_uploader("동영상을 업로드하세요", type=["mp4"])
    # st.video('/home/ubuntu/workspace/kwanwoo/capstone/temp.mp4')
    if uploaded_file is not None:
        if uploaded_file.name.endswith("mp4"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            #동영상 뿌러주기
            # output_path = make_video(temp_path)
            # new_vid_path = os.path.join(org_dir_path,output_path)
            # video_file = open(new_vid_path,'rb')
            st.video(temp_path)
            # Extract faces from the video
            vid_name = uploaded_file.name
            vid_faces = extract_faces_from_video(facedet)
            vid_real_faces = vid_faces.process_video(temp_path)
            if len(vid_real_faces) > 0:
                faces_t = preprocess_faces(vid_real_faces)
                prediction = predict_faces(net, faces_t)
                fig = make_fig(vid_name,vid_real_faces,prediction)
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig)
                with col2:
                    idx_lst = [f['frame_idx'] for f in vid_real_faces if len(f['faces'])]
                    probabilities_lst = expit(prediction).tolist()
                    max_prob_index = probabilities_lst.index(max(probabilities_lst))
                    min_prob_index = probabilities_lst.index(min(probabilities_lst))
                    suspicion_rating = expit(prediction.mean())
                    st.markdown('<div class="rating-label">Overall Suspicion Rating</div>', unsafe_allow_html=True)
                    styled_progress_bar(suspicion_rating)
                    st.write(" ")
                    # st.write(f"{vid_name} 영상은`{int(suspicion_rating*100)}%`의 확률로 딥페이크 영상입니다.")
                    # st.write(f"가장 큰 확률의 프레임은`{idx_lst[max_prob_index]}`이였으며 확률은`{int(probabilities_lst[max_prob_index]*100)}`%입니다.")
                    # st.write(f"가장 작은 확률의 프레임은`{idx_lst[min_prob_index]}`이였으며 확률은`{int(probabilities_lst[min_prob_index]*100)}`%입니다.")
                    st.markdown(f"""<p style='font-size: 16px;'>
                        <code>{vid_name}</code> 영상은<code>{int(suspicion_rating*100)}%</code>의 확률로 딥페이크 영상입니다. </br>
                        가장 큰 확률의 프레임은<code>{idx_lst[max_prob_index]}</code>이였으며 확률은<code>{int(probabilities_lst[max_prob_index]*100)}%</code>입니다. </br>
                        가장 작은 확률의 프레임은<code>{idx_lst[min_prob_index]}</code>이였으며 확률은<code>{int(probabilities_lst[min_prob_index]*100)}%</code>입니다.
                    </p>""", unsafe_allow_html=True)
                
            else:
                st.warning("얼굴을 감지하지 못했습니다.")

if __name__ == "__main__":
    main()
