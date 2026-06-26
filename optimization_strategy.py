import numpy as np
def reinforce_learning_optimize(knowledge_base, eval_result, current_params, learning_rate=0.1):
    optimized_params = {}
    for param in current_params:
        reward = eval_result * knowledge_base.get(param, 1.0)
        optimized_params[param] = current_params[param] + learning_rate * reward
    return optimized_params


if __name__ == "__main__":
    kb = {"harmony": 0.8, "rhythm": 0.7, "timbre": 0.9}
    eval_res = 0.6
    current_params = {"harmony": 0.5, "rhythm": 0.5, "timbre": 0.5}
    
    optimized = reinforce_learning_optimize(kb, eval_res, current_params)
    print("优化前参数:", current_params)
    print("优化后参数:", optimized)