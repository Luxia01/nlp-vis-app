import streamlit as st
from transformers import pipeline

# 标题
st.title("🌟 ALBERT 文本情感分析可视化系统")

# 输入框
text = st.text_input("请输入一句文本：", "我真的非常喜欢这部电影！")

# 加载模型（ALBERT 情感分类）
classifier = pipeline("sentiment-analysis", model="textattack/albert-base-v2-SST-2")

# 执行分析
if st.button("开始分析"):
    result = classifier(text)[0]
    st.subheader("🧾 模型输出结果")
    st.write(f"**标签：** {result['label']}")
    st.write(f"**置信度：** {result['score']:.4f}")
