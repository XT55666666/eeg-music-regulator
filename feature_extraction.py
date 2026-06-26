import numpy as np


def extract_time_varying_patterns(normalized_features, time_win=10):
    patterns = []
    for i in range(len(normalized_features) - time_win + 1):
        window = normalized_features[i:i + time_win]
        patterns.append(np.mean(window)) 
    return np.array(patterns)


def build_emotion_space(patterns_list, emotion_labels):
    emotion_space = {}
    for label, patterns in zip(emotion_labels, patterns_list):
        emotion_space[label] = patterns
    return emotion_space


def get_realtime_trajectory(normalized_features, time_win=10):
    return extract_time_varying_patterns(normalized_features, time_win)


if __name__ == "__main__":
    test_features = np.random.randn(100) 
    patterns = extract_time_varying_patterns(test_features)
    emotion_space = build_emotion_space([patterns], ["test_emotion"])
    realtime_traj = get_realtime_trajectory(test_features)
    print("提取的时变模式长度:", len(patterns))
    print("构建的情绪空间示例:", emotion_space)
    print("实时情绪轨迹长度:", len(realtime_traj))