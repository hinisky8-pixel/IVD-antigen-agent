import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from smolagents import tool

@tool
def predict_stability_score(sequence: str, mutation: str) -> float:
    """
    使用预加载的 ESM 模型计算突变稳定性得分。
    Args:
        sequence: 蛋白质氨基酸序列。
        mutation: 突变描述，如 'A15V'。
    """
    # 核心：直接调用 Colab 环境中的全局变量
    global tokenizer, model_esm, device
    
    try:
        wt_aa, mt_aa = mutation[0], mutation[-1]
        pos = int(mutation[1:-1]) - 1 
        
        # 推理逻辑
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model_esm(**inputs).logits[0, pos + 1]
            
        # 计算对数似然差值
        score = float(logits[tokenizer.convert_tokens_to_ids(mt_aa)] - 
                      logits[tokenizer.convert_tokens_to_ids(wt_aa)])
        return round(score, 4)
    except Exception as e:
        print(f"ESM 推理出错: {e}")
        return -99.0

@tool
def get_hydrophobicity_info(mutation: str) -> dict:
    """
    计算突变前后的疏水性变化(Kyte-Doolittle scale)。
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
    
    risk = "低风险"
    if delta > 2.0: risk = "⚠️ 高风险"
    elif delta > 0.5: risk = "中风险"
    
    return {"delta_h": round(delta, 2), "risk_level": risk}
