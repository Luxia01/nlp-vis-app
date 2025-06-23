import streamlit as st

# 初始化 session_state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# 登录函数
def login():
    st.title("🔐 登录 NLP 可视化系统")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    login_btn = st.button("登录")
    
    if login_btn:
        if username == "liming" and password == "123456":
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("登录成功！请点击左上角 [重新运行] 或刷新页面进入系统")
        else:
            st.error("用户名或密码错误")

# 登录控制逻辑
if not st.session_state.authenticated:
    login()
    st.stop()

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

# 设置页面
st.set_page_config(page_title="DistilBERT 情感分析可视化", layout="centered")
st.title("🌟 基于 DistilBERT 的文本情感分析可视化系统")

# 文本输入
text = st.text_input("请输入一句文本：", "我真的非常喜欢这部电影！")

# 缓存模型加载
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model

tokenizer, model = load_model()

# 执行分析
if st.button("开始分析"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

    labels = ["消极", "积极"]
    pred_label = labels[probs.argmax().item()]

    st.subheader("🧾 模型输出结果")
    st.write(f"**预测标签：** {pred_label}")
    st.write(f"**分类概率：** {probs.tolist()}")

    # 条形图展示
    st.subheader("📊 分类概率图")
    fig, ax = plt.subplots()
    ax.bar(labels, probs.tolist(), color=["red", "green"])
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    # 模拟 Attention 高亮（可视化示例）
    st.subheader("🧠 模拟 Attention 高亮")
    tokens = tokenizer.tokenize(text)
    importance = torch.rand(len(tokens))  # 模拟注意力强度
    highlighted = ""
    for token, score in zip(tokens, importance):
        opacity = round(float(score), 2)
        token_clean = token.replace("##", "")
        highlighted += f"<span style='background-color: rgba(255,255,0,{opacity}); padding:2px'>{token_clean}</span> "
    st.markdown(highlighted, unsafe_allow_html=True)


