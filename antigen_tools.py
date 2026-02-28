import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from smolagents import tool

def load_esm_local(model_path):
    """
    专门用于从 Kaggle 下载的本地路径加载 ESM 模型。
    Args:
        model_path: Kaggle 模型下载后的本地存储路径。
    """
    print(f"📦 正在从本地路径加载 ESM 权重: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 强制从本地路径读取，不联网
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model, device

@tool
def predict_stability_score(sequence: str, mutation: str, tokenizer, model_esm, device) -> float:
    """
    利用本地加载的 ESM 模型计算突变稳定性得分。
    Args:
        sequence: 抗原序列。
        mutation: 突变描述 (如 'T15V')。
        tokenizer, model_esm, device: 由外部传入的本地模型组件。
    """
    try:
        wt_aa, mt_aa = mutation[0], mutation[-1]
        # 解析位置 (注意 ESM 使用 1-based indexing 且包含起始占位符)
        pos = int(mutation[1:-1]) - 1 
        
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model_esm(**inputs).logits[0, pos + 1]
        
        # 计算突变型与野生型的对数几率差
        score = float(logits[tokenizer.convert_tokens_to_ids(mt_aa)] - logits[tokenizer.convert_tokens_to_ids(wt_aa)])
        return round(score, 4)
    except Exception as e:
        print(f"⚠️ 稳定性计算出错: {e}")
        return -99.0

@tool
def get_hydrophobicity_info(mutation: str) -> dict:
    """
    使用 Kyte-Doolittle 量表计算突变的疏水性变化(ΔH)。
    Args:
        mutation: 突变描述 (如 'T15V')。
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
