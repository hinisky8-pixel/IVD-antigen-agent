import torch
from smolagents import tool
from typing import Any

@tool
def predict_stability_score(sequence: str, mutation: str) -> float:
    """
    使用预加载的 ESM 模型计算突变稳定性得分。
    Args:
        sequence: 蛋白质氨基酸序列。
        mutation: 突变描述，如 'A15V'。
    """
    # 显式从执行环境中获取注入的对象
    try:
        wt_aa, mt_aa = mutation[0], mutation[-1]
        pos = int(mutation[1:-1]) - 1 
        
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model_esm(**inputs).logits[0, pos + 1]
            
        score = float(logits[tokenizer.convert_tokens_to_ids(mt_aa)] - 
                      logits[tokenizer.convert_tokens_to_ids(wt_aa)])
        return round(score, 4)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_hydrophobicity_info(mutation: str) -> dict:
    """
    计算突变前后的疏水性变化。
    Args:
        mutation: 突变描述，如 'A15V'。
    """
    kd_scale = {
        'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
        'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
        'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
    }
    wt_aa, mt_aa = mutation[0], mutation[-1]
    delta = kd_scale.get(mt_aa, 0) - kd_scale.get(wt_aa, 0)
    risk = "高风险" if delta > 0.5 else "低风险"
    return {"delta_h": round(delta, 2), "risk_level": risk}
