import numpy as np

class MusicElement:
    def __init__(self, harmony_complexity, rhythm_density, timbre_saturation):
        self.harmony_complexity = harmony_complexity
        self.rhythm_density = rhythm_density
        self.timbre_saturation = timbre_saturation


def adjust_music_elements(trajectory_fluctuation):
    base_harmony = 0.5
    base_rhythm = 0.5
    base_timbre = 0.5
    
    harmony_adj = base_harmony + 0.2 * trajectory_fluctuation
    rhythm_adj = base_rhythm + 0.3 * trajectory_fluctuation
    timbre_adj = base_timbre + 0.1 * trajectory_fluctuation
    
    return MusicElement(harmony_adj, rhythm_adj, timbre_adj)


def generate_music_sequence(music_element, length=10):
    music_seq = []
    for _ in range(length):
        harmony = np.random.normal(music_element.harmony_complexity, 0.1)
        rhythm = np.random.normal(music_element.rhythm_density, 0.1)
        timbre = np.random.normal(music_element.timbre_saturation, 0.1)
        music_seq.append((harmony, rhythm, timbre))
    return music_seq


if __name__ == "__main__":
    traj_fluc = np.random.randn(1)[0]
    element = adjust_music_elements(traj_fluc)
    music_seq = generate_music_sequence(element)
    print("调整后的音乐元素 - 和声复杂度:", element.harmony_complexity, 
          "节奏密度:", element.rhythm_density, 
          "音色饱和度:", element.timbre_saturation)
    print("生成的音乐序列示例:", music_seq[:3])