import trimesh
import PIL.Image

if not hasattr(PIL.Image.Image, 'tostring'):
    PIL.Image.Image.tostring = PIL.Image.Image.tobytes

def view_glb_trimesh(file_path):
    try:
        # 1. 파일 로드
        print(f"Loading {file_path}...")
        scene = trimesh.load(file_path)

        # 2. 정보 출력 (선택사항)
        if isinstance(scene, trimesh.Scene):
            print(f"Scene bounds: {scene.bounds}")
            print(f"Number of geometries: {len(scene.geometry)}")
        else:
            print(f"Mesh vertices: {len(scene.vertices)}")

        # 3. 시각화
        scene.show()
        
    except Exception as e:
        print(f"Error loading GLB: {e}")
        # 디버깅을 위해 상세 에러 로그 출력
        import traceback
        traceback.print_exc()

# 실행 예시
glb_path = "./visualize_scene/scenes/Rs_int_furnished.glb"
view_glb_trimesh(glb_path)