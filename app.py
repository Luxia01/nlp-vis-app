import streamlit as st
from transformers import pipeline, AutoTokenizer

# 标题
st.title("🌟 ALBERT 文本情感分析可视化系统")

# 输入框
text = st.text_input("请输入一句文本：", "我真的非常喜欢这部电影！")

try:
    # 显式加载tokenizer并禁用快速模式
    tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2", use_fast=False)
    
    # 加载模型
    classifier = pipeline(
        "sentiment-analysis",
        model="textattack/albert-base-v2-SST-2",
        tokenizer=tokenizer
    )
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# 执行分析
if st.button("开始分析"):
    try:
        result = classifier(text)[0]
        st.subheader("🧾 模型输出结果")
        st.write(f"**标签：** {result['label']}")
        st.write(f"**置信度：** {result['score']:.4f}")
    except Exception as e:
        st.error(f"分析失败: {str(e)}")
# 使用更稳定的替代模型
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
