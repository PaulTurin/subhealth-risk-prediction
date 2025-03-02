import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = '/System/Library/Fonts/STHeiti Medium.ttc'  
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False


file_path = '/Users/paul/Desktop/data_imputed_V14.0.0.0.1.xlsx'
data = pd.read_excel(file_path)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)


model = joblib.load('/Users/paul/Desktop/risk_prediction_model.pkl')


def get_optimal_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


categorical_features = {
    '兴趣爱好': {
        'labels': ['没有', '一般（拥有1 - 3个兴趣爱好）', '广泛（拥有大于3个兴趣爱好）'],
        'values': [0, 1, 2]
    },
    '性格类型': {
        'labels': ['急躁好胜', '忍气吞声', '焦虑易怒', '成熟稳重'],
        'values': [0, 0, 0, 1]
    },
    '不良饮食习惯': {
        'labels': ['无', '饮食时间不规律', '节食', '暴饮暴食', '偏食挑食', '口味偏咸', '以零食替代正餐'],
        'values': [0, 1, 1, 1, 1, 1, 1]
    },
    '体育锻炼': {
        'labels': ['几乎不', '2～4天/周', '5～7天/周'],
        'values': [0, 1, 2]
    },
    '睡眠持续时长': {
        'labels': ['小于7小时/天', '大于7小时/天'],
        'values': [0, 1]
    },
    '网络使用（非工作学习需要）': {
        'labels': ['小于1小时/天', '1～4小时/天', '大于4小时/天'],
        'values': [0, 1, 2]
    },
    '户籍类型': {
        'labels': ['农村', '城镇', '三、四线城市', '一、二线城市'],
        'values': [0, 1, 2, 3]
    },
    '居住环境舒适度': {
        'labels': ['绿化好', '空气质量好', '噪音小', '住房精致', '居住宽敞', '邻里和睦', '以上都没有'],
        'values': []  
    },
    '近期重大生活事件（近半年）': {
        'labels': ['退休', '爱情或婚姻出现问题', '与人不和', '身边人病重或去世', '自己生病已痊愈',
                   '丢失贵重财务', '出现经济问题', '陷入法律纠纷', '发生事故意外', '子女教育问题', '以上都没有'],
        'values': []  
    }
}


question_order = [
    '年龄',
    '户籍类型',
    '居住环境舒适度',
    '兴趣爱好',
    '性格类型',
    '体育锻炼',
    '睡眠持续时长',
    '不良饮食习惯',
    '网络使用（非工作学习需要）',
    '近期重大生活事件（近半年）'
]


st.write("## 亚健康风险预测工具")

st.write("本工具由南方医科大学许军教授课题组研发，内核采用CatBoost机器学习算法，结合课题组自主研发的SHMS V1.0量表和构建的CURSCS数据库，为您提供科学、精准的亚健康风险评估服务。只需简单输入您的个人信息和生活习惯，即可快速获取个性化的健康风险等级评估，并获得健康建议，助您轻松掌握健康主动权！")

input_features = []
valid_selection = True  

for feature_name in question_order:
    if feature_name == '不良饮食习惯':
        
        selected_labels = st.multiselect(
            f"请选择 {feature_name}",
            categorical_features[feature_name]['labels']
        )
        if '无' in selected_labels and len(selected_labels) > 1:
            st.error("不能同时选择“无”和其他不良饮食习惯选项，请重新选择。")
            valid_selection = False
        elif '无' in selected_labels:
            feature_value = 0
        elif selected_labels:
            feature_value = 1
        else:
            st.error("请选择至少一个不良饮食习惯选项或者选择“无”。")
            valid_selection = False
        if valid_selection:
            input_features.append(feature_value)
    elif feature_name == '居住环境舒适度':
        
        selected_labels = st.multiselect(
            f"请选择 {feature_name}",
            categorical_features[feature_name]['labels']
        )
        if '以上都没有' in selected_labels and len(selected_labels) > 1:
            st.error("不能同时选择“以上都没有”和其他选项，请重新选择。")
            valid_selection = False
        elif '以上都没有' in selected_labels:
            feature_value = 0
        elif not selected_labels:
            st.error("请选择至少一个选项或者选择“以上都没有”。")
            valid_selection = False
        else:
            num_selected = len(selected_labels)
            if num_selected == 1:
                feature_value = 0
            elif 2 <= num_selected <= 4:
                feature_value = 1
            else:
                feature_value = 2
        if valid_selection:
            input_features.append(feature_value)
    elif feature_name == '近期重大生活事件（近半年）':
        
        selected_labels = st.multiselect(
            f"请选择 {feature_name}",
            categorical_features[feature_name]['labels']
        )
        if '以上都没有' in selected_labels and len(selected_labels) > 1:
            st.error("不能同时选择“以上都没有”和其他选项，请重新选择。")
            valid_selection = False
        elif '以上都没有' in selected_labels:
            feature_value = 0
        elif not selected_labels:
            st.error("请选择至少一个选项或者选择“以上都没有”。")
            valid_selection = False
        else:
            feature_value = 1
        if valid_selection:
            input_features.append(feature_value)
    elif feature_name in categorical_features:
        
        label = st.selectbox(
            f"请选择 {feature_name}",
            categorical_features[feature_name]['labels']
        )
        value_index = categorical_features[feature_name]['labels'].index(label)
        feature_value = categorical_features[feature_name]['values'][value_index]
        input_features.append(feature_value)
    elif feature_name == '年龄':
        
        feature_value = st.number_input(f"请输入{feature_name}", value=0.0)
        input_features.append(feature_value)

if valid_selection and st.button("风险预测"):
    input_features = np.array(input_features).reshape(1, -1)
    y_pred_proba = model.predict_proba(input_features)[:, 1]
    
    train_y_pred_proba = model.predict_proba(X_train)[:, 1]
    optimal_threshold = get_optimal_threshold(y_train, train_y_pred_proba)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    if y_pred_proba[0] < optimal_threshold:
        relative_prob = (y_pred_proba[0] / optimal_threshold) * 0.5
    else:
        relative_prob = 0.5 + ((y_pred_proba[0] - optimal_threshold) / (1 - optimal_threshold)) * 0.5
    power = 2.5  
    if relative_prob < 0.5:
        amplified_relative_prob = 0.5 * (relative_prob / 0.5) ** power
    else:
        amplified_relative_prob = 0.5 + 0.5 * ((relative_prob - 0.5) / 0.5) ** power

  
    st.write(f"**您的亚健康发生风险预测概率: {amplified_relative_prob * 100:.2f}%**")
    
    if amplified_relative_prob < 0.5:
        risk_level = "低风险"
        st.write("根据我们的模型，**您的亚健康风险较低**，健康状况良好！继续保持健康的生活习惯，定期关注自身健康状况，预防胜于治疗！")
    elif 0.5 <= amplified_relative_prob < 0.6:
        risk_level = "中风险"
        st.write("根据我们的模型，**您的亚健康风险处于中等水平**，可能存在一些健康隐患。建议您更加关注健康风险，适当调整生活习惯，避免过度疲劳和压力。定期进行健康检查，及时了解身体状况，必要时可咨询健康专家获取建议。")
    else:
        risk_level = "高风险"
        st.write("根据我们的模型，**您的亚健康风险较高**。请高度重视您的健康状况，及时调整生活方式，避免不良习惯。建议尽快咨询医疗保健提供者，进行进一步评估和干预，确保健康风险得到有效控制。")


    st.write("无论您的风险评估结果如何，健康管理都是一个需要长期关注的过程。定期关注自身健康，了解身体的变化和需求，是预防疾病、保持活力的关键。您的健康是我们最关心的事，让我们一起努力，拥抱健康，远离风险！")


    st.write("### 健康风险分析图")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(input_features)
    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=input_features[0],
            feature_names=question_order
        ),
        show=False
    )
    st.pyplot(fig)

    
    explanation_text_1 = "*这张图展示了影响您健康风险的主要因素及其贡献程度。图中，红色部分表示这些因素增加了您的健康风险，而蓝色部分则表示这些因素降低了您的健康风险。条形的长度代表了该因素对您健康风险的影响大小，条形越长，影响越大。通过这张图，您可以直观地看到哪些因素对您的健康风险影响最显著。*"
    explanation_text_2 = "*例如，如果 “睡眠时长” 的条形为红色且较长，说明睡眠不足可能显著增加了您的健康风险；如果 “体育锻炼” 的条形为蓝色且较长，则说明规律运动对降低您的健康风险有积极作用。*"

    st.markdown(f"<small>{explanation_text_1}</small></span>", unsafe_allow_html=True)
    st.markdown(f"<small>{explanation_text_2}</small></span>", unsafe_allow_html=True)

declaration_text = "声明：本工具由许军教授课题组硕士研究生刘琛主要负责开发，目前处于测试阶段，其所有权归属于许军教授课题组。该工具现仅作为刘琛本人硕士毕业论文展示使用，未经许可，任何个人或组织不得将其用于商业用途或公开发布。本工具所涉及的代码、算法、模型等相关内容均受版权法保护，未经授权，禁止任何形式的复制、修改、传播或用于其他目的。如有任何疑问或需要进一步了解，请联系liuchen_scires@sina.cn。"
st.markdown(f"<span style='color:red;'><small>{declaration_text}</small></span>", unsafe_allow_html=True)
