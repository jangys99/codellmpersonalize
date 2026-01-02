import os
import yaml
import trimesh
import numpy as np
import trimesh.transformations as tf
from scipy.spatial.transform import Rotation as R

# ================= 설정 부분 =================
SCENE_ID = "Rs_int" # 씬 이름

# [중요] 도커 내부 경로로 설정
BASE_DIR = "./data/scene_datasets/igibson"
SCENE_GLB_PATH = os.path.join(BASE_DIR, "scenes", f"{SCENE_ID}.glb")

# 가구 정보가 있는 메타데이터 (assets 폴더에 있음)
METADATA_PATH = os.path.join(BASE_DIR, "assets", f"{SCENE_ID}", "metadata_v2.yaml") 

# [핵심 변경] 실제 3D 모델(obj)이 들어있는 폴더 (assets_assemble)
ASSEMBLE_DIR = os.path.join(BASE_DIR, "assets_assemble", f"{SCENE_ID}")

OUTPUT_FILENAME = f"./visualize_scene/scenes/{SCENE_ID}_furnished.glb"
# ===========================================

def load_scene_and_furniture_obj():
    # 1. 빈 집(Scene) 로드
    print(f"Loading House Shell: {SCENE_GLB_PATH}...")
    try:
        scene = trimesh.load(SCENE_GLB_PATH)
    except Exception as e:
        print(f"Error loading scene GLB: {e}")
        return

    print("Removing ceiling objects based on name...")
    
    # 삭제할 객체들의 이름을 담을 리스트
    names_to_remove = []
    
    # scene.geometry의 키(이름)들을 순회하며 확인
    for geom_name in scene.geometry.keys():
        # 이름에 'ceiling'이 포함되어 있으면 삭제 목록에 추가
        # (대소문자 구분 없이 찾기 위해 lower() 사용)
        if "ceiling" in geom_name.lower():
            names_to_remove.append(geom_name)
            
    # 실제 삭제 수행
    if not names_to_remove:
        print("  [Warning] No objects with 'ceiling' in name found.")
    else:
        for name in names_to_remove:
            print(f"  - Deleting geometry: {name}")
            scene.delete_geometry(name)

    x_rot = tf.rotation_matrix(np.radians(-90), [1, 0, 0])
    scene.apply_transform(x_rot)

    # 2. 메타데이터 로드
    target_metadata_path = METADATA_PATH
    if not os.path.exists(target_metadata_path):
        alt_path = os.path.join(BASE_DIR, "assets", f"{SCENE_ID}", "metadata_assembled.yaml")
        if os.path.exists(alt_path):
            target_metadata_path = alt_path
        else:
            print(f"Metadata not found: {target_metadata_path}")
            return

    print(f"Loading Metadata from: {target_metadata_path}")
    with open(target_metadata_path, 'r') as f:
        meta = yaml.safe_load(f)

    # 데이터 구조 대응
    objects_data = meta.get('urdfs', meta) 

    print(f"Placing Furniture from {ASSEMBLE_DIR}...")

    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)

    success_count = 0
    
    for obj_filename, props in objects_data.items():
        # obj_filename 예시: "bottom_cabinet_15_0.urdf"
        
        # 1. 이름에서 확장자(.urdf) 제거 -> 폴더 이름 획득
        obj_name = obj_filename.split('.')[0]  # "bottom_cabinet_15_0"
        
        # 2. assets_assemble 안의 해당 폴더 경로 구성
        obj_folder = os.path.join(ASSEMBLE_DIR, obj_name)
        
        # 3. 로드할 모델 파일 찾기 (.obj)
        # 사진을 보니 'model.obj' 또는 '폴더명.obj'가 섞여 있을 수 있음
        candidates = [
            os.path.join(obj_folder, "model.obj"),      # 1순위: model.obj
            os.path.join(obj_folder, f"{obj_name}.obj") # 2순위: 이름.obj
        ]
        
        model_path = None
        for path in candidates:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            # print(f"  [Skip] OBJ not found for: {obj_name}")
            continue

        try:
            # 4. OBJ 로드 (URDF보다 훨씬 단순하고 빠름)
            furniture = trimesh.load(model_path, force='mesh')
        except Exception as e:
            print(f"  [Error] Failed to load {obj_name}: {e}")
            continue
        
        # 5. 위치/회전 적용
        pos = props.get('pos', [0, 0, 0])
        rot = props.get('rot', [0, 0, 0, 1]) # [x, y, z, w]
        
        transform = np.eye(4)
        if len(rot) == 4:
            # iGibson/Habitat은 보통 Quaternion [x, y, z, w] 사용
            r = R.from_quat(rot)
            transform[:3, :3] = r.as_matrix()
        transform[:3, 3] = pos
        
        # 6. 씬에 추가
        if isinstance(furniture, trimesh.Scene):
            # OBJ 안에 여러 파트가 있는 경우
            for name, geom in furniture.geometry.items():
                geom_copy = geom.copy()
                geom_copy.apply_transform(transform)
                # 텍스처(Visual) 정보가 있으면 유지됨
                scene.add_geometry(geom_copy, geom_name=f"{obj_name}_{name}")
        else:
            furniture.apply_transform(transform)
            scene.add_geometry(furniture, geom_name=obj_name)
            
        success_count += 1

    print(f"Done! Placed {success_count}/{len(objects_data)} objects.")
    print(f"Exporting to {OUTPUT_FILENAME}...")
    scene.export(OUTPUT_FILENAME)
    print("Export Complete.")

if __name__ == "__main__":
    load_scene_and_furniture_obj()