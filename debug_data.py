import gzip
import json

# 파일 경로 (업로드한 파일이 있는 위치)
file_path = '/workspace/codellmpersonalize/data/datasets/cos_eor_v11_pruned/val/Merom_1_int.json.gz'

try:
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 전체 구조 확인
    print(f"=== 기본 정보 ===")
    print(f"최상위 키(Keys): {list(data.keys())}")
    
    episodes = data.get('episodes', [])
    print(f"총 에피소드 개수: {len(episodes)}")
    print("=" * 30 + "\n")

    # 2. 처음 15개 에피소드 상세 분석 (Task 변화 확인)
    print("=== 에피소드 0번 ~ 14번 분석 (5회 단위 패턴 확인) ===")
    
    for i in range(min(15, len(episodes))):
        ep = episodes[i]
        ep_id = ep.get('episode_id', 'N/A')
        
        # 물체 목록 추출 (Task를 구분하는 핵심 지문)
        # 보통 'objects' 리스트 안에 각 물체의 'id'나 'template' 정보가 들어있습니다.
        obj_names = []
        if 'objs_keys' in ep:
            # 물체 ID나 템플릿 이름만 가져와서 구성이 같은지 봅니다.
            # 예: '003_cracker_box|...' -> '003_cracker_box'
            for obj in ep['objs_keys']:
                obj_names.append(obj)
        
        # 정렬해서 비교 (순서만 다르고 구성이 같은지 확인하기 위해)
        obj_names.sort()
        
        print(f"[Episode {i}] (ID: {ep_id})")
        print(f" - 등장 물체({len(obj_names)}개): {obj_names}")
        
        # 5번째마다 구분선을 넣어 패턴이 바뀌는지 확인
        if (i + 1) % 5 == 0:
            print("\n" + "-" * 20 + " ▲ 위 5개가 한 그룹(Task)인지 확인 ▲ " + "-" * 20 + "\n")

except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {file_path}")
    print("파일 경로를 다시 확인해주세요.")
except Exception as e:
    print(f"오류 발생: {e}")