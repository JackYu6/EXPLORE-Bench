def init_llm(llm_name, model_path, enable_thinking):
    llm = None
    llm_name = llm_name.lower()
    llm_names = [ 
        "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "ovis2.5", "minicpm-v4.5", "keye-vl1.5",
        "mimo_vl2508", "internvl3.5", "llava_onevision1.5", "step3-vl", "glm4.6v-flash", 
        "egothinker", "embodiedreasoner",
        "gemini-3-flash-preview", "gemini-3-pro-preview", "azure-gpt-5_2-chat"
                ]
    
    # ---------- Open-source Models ----------
    if "qwen3-vl" in llm_name:
        from models.Qwen_VL.Qwen3_VL import Qwen3_VL
        llm = Qwen3_VL(model_path)
    elif "ovis2.5" in llm_name:
        from models.Ovis.Ovis2_5 import Ovis2_5
        llm = Ovis2_5(model_path, enable_thinking)
    elif "minicpm-v4.5" in llm_name:
        from models.MiniCPM_V.MiniCPM_V4_5 import MiniCPM_V4_5
        llm = MiniCPM_V4_5(model_path, enable_thinking)
    elif "keye-vl1.5" in llm_name:
        from models.Keye_VL.Keye_VL1_5 import Keye_VL1_5
        llm = Keye_VL1_5(model_path, enable_thinking)
    elif "mimo_vl2508" in llm_name:
        from models.MiMo_VL.MiMo_VL2508 import MiMo_VL2508
        llm = MiMo_VL2508(model_path, enable_thinking)
        
    # ---------- Open-source Non-Thinking Models ----------
    elif "qwen2-vl" in llm_name:
        from models.Qwen_VL.Qwen2_VL import Qwen2_VL
        llm = Qwen2_VL(model_path)
    elif "qwen2.5-vl" in llm_name:
        from models.Qwen_VL.Qwen2_5_VL import Qwen2_5_VL
        llm = Qwen2_5_VL(model_path)
    elif "internvl3.5" in llm_name:
        from models.InternVL.InternVL3_5 import InternVL3_5
        llm = InternVL3_5(model_path) 
    elif "llava_onevision1.5" in llm_name:
        from models.LLaVA_OneVison.LLaVA_OneVision1_5 import LLaVA_OneVision1_5
        llm = LLaVA_OneVision1_5(model_path)
    
    # ---------- Open-source Thinking Models ---------- 
    elif "step3-vl" in llm_name:
        from models.Step_VL.Step3_VL import Step3_VL
        llm = Step3_VL(model_path)
    elif "glm4.6v-flash" in llm_name:
        from models.GLM_V.GLM4_6V_Flash import GLM4_6V_Flash
        llm = GLM4_6V_Flash(model_path)
        
    # ---------- Embodied/Egocentric Models ----------
    elif "embodiedreasoner" in llm_name:
        from models.EmbodiedReasoner.EmboidedReasoner import EmboidedReasoner
        llm = EmboidedReasoner(model_path)
    elif "egothinker" in llm_name:
        from models.EgoThinker.EgoThinker import EgoThinker
        llm = EgoThinker(model_path)
    
    # ---------- Proprietary Models ----------
    elif "gemini" in llm_name:
        from models.Gemini.Gemini import Gemini
        llm = Gemini(llm_name)
    elif "gpt" in llm_name:
        from models.GPT.GPT import GPT
        llm = GPT(llm_name)
        
    else:
        raise f"{llm_name} not supported"
    return llm


