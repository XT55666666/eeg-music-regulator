import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cal_local_similarity(realtime_traj, group_traj):
    similarity = []
    min_len = min(len(realtime_traj), len(group_traj))
    for i in range(min_len):
        sim = cosine_similarity([[realtime_traj[i]]], [[group_traj[i]]])[0][0]
        similarity.append(sim)
    return np.mean(similarity)


def multi_scale_transfer(realtime_features, historical_features, scales=[1, 2, 4]):
    transferred_features = []
    for scale in scales:
        realtime_seg = np.array([np.mean(realtime_features[i:i + scale]) for i in range(0, len(realtime_features), scale)])
        historical_seg = np.array([np.mean(historical_features[i:i + scale]) for i in range(0, len(historical_features), scale)])
        transferred = realtime_seg + 0.5 * (historical_seg - np.mean(historical_seg)) 
        transferred_features.extend(transferred)
    return np.array(transferred_features)


def build_calibration_network(transferred_features):
    return transferred_features  


if __name__ == "__main__":
    realtime_traj = np.random.randn(20)
    group_traj = np.random.randn(20)
    realtime_feats = np.random.randn(100)
    historical_feats = np.random.randn(100)
    
    similarity = cal_local_similarity(realtime_traj, group_traj)
    transferred = multi_scale_transfer(realtime_feats, historical_feats)
    calib_net = build_calibration_network(transferred)
    
    print("局部相似性均值:", similarity)
    print("多尺度迁移后特征长度:", len(transferred))
    print("校准网络输出长度:", len(calib_net))