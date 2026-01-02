import numpy as np
import cv2
import os

# 1. 파일 경로 설정 (사용자님 경로에 맞게 수정하세요)
npy_path = "./logs/test/videos/pomaria_1_int/ep_Pomaria_1_int_342-frames_1000.npy"
output_video_name = "./logs/test/videos/pomaria_1_int/output_video.mp4"

def npy_to_mp4(npy_file, output_name, fps=10):
    print(f"Loading {npy_file}...")
    try:
        # 데이터 로드
        frames = np.load(npy_file)
        print(f"데이터 로드 성공! 총 프레임 수: {len(frames)}")
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    if len(frames) == 0:
        print("프레임이 비어 있습니다.")
        return

    # 프레임 크기 확인 (Height, Width, Channel)
    # 보통 (N, H, W, 3) 또는 (N, H, W, 4) 형태입니다.
    height, width, layers = frames[0].shape
    print(f"영상 크기: {width}x{height}, 채널: {layers}")

    # 비디오 코덱 설정 (mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    print("비디오 변환 시작...")
    for frame in frames:
        # OpenCV는 BGR을 쓰는데, 보통 시뮬레이터는 RGB로 저장합니다.
        # 색감이 이상하면 아래 줄을 활성화하세요.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 데이터 타입이 float(0~1)인 경우 255를 곱해줘야 함
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
            
        video.write(frame)

    video.release()
    print(f"변환 완료! 저장된 파일: {output_name}")

if __name__ == "__main__":
    # 경로가 맞는지 확인 후 실행
    if os.path.exists(npy_path):
        npy_to_mp4(npy_path, output_video_name)
    else:
        print(f"경로 에러: {npy_path} 파일이 없습니다.")
        # 현재 폴더에서 파일 찾아보기
        print(f"현재 폴더 파일 목록: {os.listdir('.')}")