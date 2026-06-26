import numpy as np
import tkinter as tk
from tkinter import ttk
import data_acquisition as da
import feature_extraction as fe
import migration_index as mi
import rehabilitation_evaluation as re_eval
import scene_adjustment as sa
import music_interaction as mi_inter
import optimization_strategy as os_opt


class EmotionRegulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("情绪调控系统")
        
        self.emotion_states = ["happy", "sad", "neutral"]
        self.current_emotion = tk.StringVar(value="happy")
        self.realtime_traj = None
        self.music_sequence = None
        self.knowledge_base = {}
        
        self.create_widgets()
        
    def create_widgets(self):
        # 情绪选择
        ttk.Label(self.root, text="选择情绪状态:").grid(row=0, column=0, padx=5, pady=5)
        emotion_combobox = ttk.Combobox(self.root, textvariable=self.current_emotion, values=self.emotion_states)
        emotion_combobox.grid(row=0, column=1, padx=5, pady=5)
        
        # 采集脑电按钮
        ttk.Button(self.root, text="采集脑电信号", command=self.acquire_eeg).grid(row=1, column=0, columnspan=2, pady=10)
        
        # 展示特征按钮
        ttk.Button(self.root, text="提取情绪轨迹", command=self.extract_traj).grid(row=2, column=0, columnspan=2, pady=5)
        
        # 生成音乐按钮
        ttk.Button(self.root, text="生成个性化音乐", command=self.generate_music).grid(row=3, column=0, columnspan=2, pady=5)
        
        # 评估效果按钮
        ttk.Button(self.root, text="评估调控效果", command=self.eval_effect).grid(row=4, column=0, columnspan=2, pady=5)
        
        # 优化模型按钮
        ttk.Button(self.root, text="优化调控方案", command=self.optimize_model).grid(row=5, column=0, columnspan=2, pady=5)
        
        # 结果展示区域
        self.result_text = tk.Text(self.root, height=15, width=60)
        self.result_text.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
        
    def acquire_eeg(self):
        emotion = self.current_emotion.get()
        eeg, filtered, features = da.acquire_eeg(emotion)
        self.result_text.insert(tk.END, f"采集 {emotion} 状态脑电信号完成，标准化特征长度: {len(features)}\n")
        self.normalized_features = features
        self.historical_features = features 
        
    def extract_traj(self):
        patterns = fe.extract_time_varying_patterns(self.normalized_features)
        self.realtime_traj = fe.get_realtime_trajectory(self.normalized_features)
        self.result_text.insert(tk.END, f"提取时变模式，实时情绪轨迹长度: {len(self.realtime_traj)}\n")
        
    def generate_music(self):
        traj_fluc = np.std(self.realtime_traj) if self.realtime_traj is not None else 0
        music_element = sa.adjust_music_elements(traj_fluc)
        self.music_sequence = sa.generate_music_sequence(music_element)
        self.result_text.insert(tk.END, f"生成个性化音乐，序列长度: {len(self.music_sequence)}\n")
        
    def eval_effect(self):
        before_signals = re_eval.collect_multimodal_signals()
        after_signals = re_eval.collect_multimodal_signals()
        eval_result = re_eval.cross_modal_analysis(np.array([before_signals]), np.array([after_signals]))
        
        music_params = {
            "harmony": [seq[0] for seq in self.music_sequence],
            "rhythm": [seq[1] for seq in self.music_sequence],
            "timbre": [seq[2] for seq in self.music_sequence]
        }
        interaction_graph = mi_inter.build_interaction_graph(eval_result, music_params)
        self.knowledge_base = mi_inter.build_knowledge_base([interaction_graph])
        
        self.result_text.insert(tk.END, f"调控效果评估结果: {eval_result}\n")
        self.result_text.insert(tk.END, f"构建音乐 - 脑电交互图谱，知识库更新\n")
        
    def optimize_model(self):
        if not self.knowledge_base:
            self.result_text.insert(tk.END, "请先评估调控效果，构建知识库！\n")
            return
        
        current_params = {
            "harmony": np.mean([seq[0] for seq in self.music_sequence]),
            "rhythm": np.mean([seq[1] for seq in self.music_sequence]),
            "timbre": np.mean([seq[2] for seq in self.music_sequence])
        }
        optimized_params = os_opt.reinforce_learning_optimize(self.knowledge_base, 0.6, current_params)
        self.result_text.insert(tk.END, f"优化后调控参数: {optimized_params}\n")
        self.result_text.insert(tk.END, "个性化音乐调控方案更新！\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRegulationApp(root)
    root.mainloop()