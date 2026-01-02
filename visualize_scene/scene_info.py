import os
import trimesh
import numpy as np

# 경로 설정 (사용자님 환경에 맞게)
SCENE_ID = "Pomaria_2_int"
BASE_DIR = "./data/scene_datasets/igibson"
SCENE_GLB_PATH = os.path.join(BASE_DIR, "scenes", f"{SCENE_ID}.glb")

def inspect_scene_components():
    print(f"Loading Scene for inspection: {SCENE_GLB_PATH}...")
    try:
        scene = trimesh.load(SCENE_GLB_PATH)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "="*60)
    print(f"{'Geometry Name':<30} | {'Z-Min':<10} | {'Z-Max':<10} | {'Guess'}")
    print("="*60)

    # scene.geometry는 {이름: 메쉬객체} 형태의 딕셔너리입니다.
    for geom_name, mesh in scene.geometry.items():
        
        # 메쉬의 경계상자(Bounds)를 구합니다.
        # bounds[0] = [min_x, min_y, min_z]
        # bounds[1] = [max_x, max_y, max_z]
        bounds = mesh.bounds
        z_min = bounds[0][2]
        z_max = bounds[1][2]
        
        # 높이 정보를 바탕으로 정체를 추측해봅니다.
        guess = ""
        if z_max < 0.2:
            guess = "Floor (바닥 추정)"
        elif z_min > 2.0:
            guess = "Ceiling (천장 추정)"
        elif (z_max - z_min) > 2.0:
            guess = "Wall (벽 추정)"
        else:
            guess = "Object (기타)"

        # 이름이 너무 길면 자르기
        display_name = (geom_name[:27] + '..') if len(geom_name) > 27 else geom_name
        
        print(f"{display_name:<30} | {z_min:>10.2f} | {z_max:>10.2f} | {guess}")

    print("="*60)
    print("Inspection Complete.")

if __name__ == "__main__":
    inspect_scene_components()