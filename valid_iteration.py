import json
import glob
import os
from collections import defaultdict

# ==========================================
# [설정] 로그 파일이 있는 폴더 경로 (기존과 동일하게 수정해주세요)
LOG_FOLDER = "/workspace/codellmpersonalize/logs/IL_scene3_v4_0.9_cp4_pair_eval/demo/pomaria_2_int"
# ==========================================

def get_task_signature(episode):
    """
    에피소드에서 '태스크 고유 ID'를 생성합니다.
    태스크 ID = (정답 매핑 정보 + 시작 시 물체 위치)
    """
    # 1. 정답 매핑 (Goal State)
    correct_map = episode.get('correct_mapping', {})
    sorted_correct_items = sorted(correct_map.items())
    goal_signature = tuple(
        (k, tuple(sorted(v))) for k, v in sorted_correct_items
    )

    # 2. 시작 시 물체 위치 (Initial State)
    start_map = episode.get('current_mapping', {}).get('start', {})
    start_signature = tuple(sorted(start_map.items()))

    return (goal_signature, start_signature)

def check_repetitions():
    log_pattern = os.path.join(LOG_FOLDER, "data_*.json")
    files = sorted(glob.glob(log_pattern))
    
    if not files:
        print(f"❌ '{LOG_FOLDER}' 경로에서 파일을 찾을 수 없습니다.")
        return

    print(f"📂 Found {len(files)} log files. Scanning...")

    task_counts = defaultdict(int)
    total_episodes = 0

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # [수정된 부분] 리스트인 경우, '첫 번째 요소(Initial State)'만 봅니다.
                target_episode = None
                
                if isinstance(data, list):
                    if len(data) > 0:
                        target_episode = data[0] # 첫 번째 스텝만 사용!
                elif isinstance(data, dict):
                    target_episode = data
                
                if target_episode:
                    sig = get_task_signature(target_episode)
                    task_counts[sig] += 1
                    total_episodes += 1
                    
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    # --- 결과 출력 ---
    print(f"\n📊 Summary:")
    print(f"  - Total Episodes Processed: {total_episodes}") # 이제 125가 나와야 함
    print(f"  - Unique Tasks Identified: {len(task_counts)}") # 25가 나와야 함
    
    incorrect_counts = {sig: count for sig, count in task_counts.items() if count != 5}
    
    if not incorrect_counts:
        print("\n✅ [SUCCESS] 완벽합니다! 파일이 정확히 5회씩 반복되는 태스크로 구성되어 있습니다.")
        if len(task_counts) == total_episodes / 5:
            print(f"   (총 {len(task_counts)}개의 고유 태스크 × 5회 반복 = {total_episodes} 에피소드)")
    else:
        print(f"\n❌ [WARNING] {len(incorrect_counts)}개의 태스크가 5번 반복되지 않았습니다:")
        for i, (sig, count) in enumerate(incorrect_counts.items()):
            print(f"   Task #{i+1}: {count}회 반복됨")
            

if __name__ == "__main__":
    check_repetitions()