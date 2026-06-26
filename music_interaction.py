import numpy as np


def build_interaction_graph(eval_result, music_params):
    graph = {}
    for param_name, param_val in music_params.items():
        graph[param_name] = eval_result * param_val  
    return graph


def build_knowledge_base(graph_list):
    knowledge_base = {}
    for graph in graph_list:
        for param, val in graph.items():
            if param not in knowledge_base:
                knowledge_base[param] = []
            knowledge_base[param].append(val)
    for param in knowledge_base:
        knowledge_base[param] = np.mean(knowledge_base[param])
    return knowledge_base


if __name__ == "__main__":
    eval_res = 0.7
    music_params = {"harmony": 0.8, "rhythm": 0.6, "timbre": 0.9}
    graph = build_interaction_graph(eval_res, music_params)
    knowledge_base = build_knowledge_base([graph])
    
    print("构建的音乐 - 脑电交互图谱:", graph)
    print("生成的音乐干预知识库:", knowledge_base)