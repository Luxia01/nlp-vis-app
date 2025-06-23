import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

# 设置页面标题
st.set_page_config(page_title="ALBERT 可视化系统", layout="centered")

st.title("🌟 基于 ALBERT 的文本情感分析可视化系统")

# 文本输入
text = st.text_input("请输入一句文本：", "我真的非常喜欢这部电影！")

# 缓存模型加载
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2")
    return tokenizer, model

tokenizer, model = load_model()

# 分析按钮
if st.button("开始分析"):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
    labels = ["消极", "积极"]
    pred_label = labels[probs.argmax().item()]

    st.subheader("🧾 模型输出结果")
    st.write(f"**预测标签：** {pred_label}")
    st.write(f"**分类概率：** {probs.tolist()}")

    # 概率条形图
    st.subheader("📊 分类概率图")
    fig, ax = plt.subplots()
    ax.bar(labels, probs.tolist(), color=["red", "green"])
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    # Attention 高亮（模拟效果）
    st.subheader("🧠 模型关注词（模拟 Attention 高亮）")
    tokens = tokenizer.tokenize(text)
    importance = torch.rand(len(tokens))  # 使用随机数模拟权重

    highlighted = ""
    for token, score in zip(tokens, importance):
        opacity = round(float(score), 2)
        token_clean = token.replace("▁", "")
        highlighted += f"<span style='background-color: rgba(255,255,0,{opacity}); padding:2px'>{token_clean}</span> "

    st.markdown(highlighted, unsafe_allow_html=True)

