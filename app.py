import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests

# 加载模型
model = joblib.load('risk_prediction_model.pkl')

# 定义分类特征
categorical_features = {
    '兴趣爱好': {
        'labels': ['没有', '一般（拥有1 - 3个兴趣爱好）', '广泛（拥有大于3个兴趣爱好）'],
        'values': [2, 1, 0]
    },
    '性格类型': {
        'labels': ['急躁好胜', '忍气吞声', '焦虑易怒', '成熟稳重'],
        'values': [1, 1, 1, 0]
    },
    '不良饮食习惯': {
        'labels': ['饮食时间不规律', '节食', '暴饮暴食', '偏食挑食', '口味偏咸', '以零食替代正餐', '以上都没有'],
        'values': [1, 1, 1, 1, 1, 1, 0]
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
        'values': [2, 1, 0]
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

# 问题顺序
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

# 中英文变量名映射
chinese_to_english = {
    '年龄': 'Age',
    '户籍类型': 'Household Registration Type',
    '居住环境舒适度': 'Residential Environment Comfort',
    '兴趣爱好': 'Hobbies',
    '性格类型': 'Personality Type',
    '体育锻炼': 'Physical Exercise',
    '睡眠持续时长': 'Sleep Duration',
    '不良饮食习惯': 'Unhealthy Eating Habits',
    '网络使用（非工作学习需要）': 'Internet Use (Non - Work/Study)',
    '近期重大生活事件（近半年）': 'Major Life Events in Recent Half - Year'
}

# 页面标题和介绍
st.write("## 亚健康风险预测工具")
st.write("本工具由南方医科大学许军教授课题组研发，内核采用CatBoost机器学习算法，结合课题组自主研发的SHMS V1.0量表和构建的CURSCS数据库，为您提供科学、精准的亚健康风险评估服务。只需简单输入您的个人信息和生活习惯，即可快速获取个性化的健康风险等级评估，并获得健康建议，助您轻松掌握健康主动权！")

# 获取用户输入
input_features = []
valid_selection = True

for feature_name in question_order:
    if feature_name == '不良饮食习惯':
        selected_labels = st.multiselect(
            f"请选择 {feature_name}",
            categorical_features[feature_name]['labels']
        )
        if '以上都没有' in selected_labels and len(selected_labels) > 1:
            st.error("不能同时选择“无”和其他不良饮食习惯选项，请重新选择。")
            valid_selection = False
        elif '以上都没有' in selected_labels:
            feature_value = 0
        elif selected_labels:
            feature_value = 1
        else:
            st.error("请选择至少一个不良饮食习惯选项或者选择“以上都没有”。")
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

# 进行风险预测和分析
if valid_selection and st.button("风险预测"):
    try:
        input_features = np.array(input_features).reshape(1, -1)
        y_pred_proba = model.predict_proba(input_features)[:, 1]
        optimal_threshold = 0.95
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        if y_pred_proba[0] < optimal_threshold:
            relative_prob = (y_pred_proba[0] / optimal_threshold) * 0.5
        else:
            relative_prob = 0.5 + ((y_pred_proba[0] - optimal_threshold) / (1 - optimal_threshold)) * 0.5

        st.write(f"**您的亚健康发生风险预测概率（相对值）: {relative_prob * 100:.2f}%**")
        if relative_prob < 0.5:
            risk_level = "低风险"
            st.write("根据我们的模型，**您的亚健康风险较低**，健康状况良好！继续保持健康的生活习惯，定期关注自身健康状况，预防胜于治疗！")
        elif 0.5 <= relative_prob < 0.6:
            risk_level = "中风险"
            st.write("根据我们的模型，**您的亚健康风险处于中等水平**，可能存在一些健康隐患。建议您更加关注健康风险，适当调整生活习惯，避免过度疲劳和压力。定期进行健康检查，及时了解身体状况，必要时可咨询健康专家获取建议。")
        else:
            risk_level = "高风险"
            st.write("根据我们的模型，**您的亚健康风险较高**。请高度重视您的健康状况，及时调整生活方式，避免不良习惯。建议尽快咨询医疗保健提供者，进行进一步评估和干预，确保健康风险得到有效控制。")

        st.write("### 健康风险分析图")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_features)

        # 将中文特征名转换为英文
        english_question_order = [chinese_to_english[feature] for feature in question_order]

        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_features[0],
                feature_names=english_question_order
            ),
            show=False
        )
        st.pyplot(fig)

        explanation_text_1 = "这张图展示了影响您健康风险的主要因素及其贡献程度。图中，红色部分表示这些因素增加了您的健康风险，而蓝色部分则表示这些因素降低了您的健康风险。条形的长度代表了该因素对您健康风险的影响大小，条形越长，影响越大。通过这张图，您可以直观地看到哪些因素对您的健康风险影响最显著。"
        explanation_text_2 = "例如，如果 “睡眠时长” 的条形为红色且较长，说明睡眠不足可能显著增加了您的健康风险；如果 “体育锻炼” 的条形为蓝色且较长，则说明规律运动对降低您的健康风险有积极作用。"

        st.markdown(f"<small>{explanation_text_1}</small>", unsafe_allow_html=True)
        st.markdown(f"<small>{explanation_text_2}</small>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"预测或分析过程中出现错误：{e}")

# 声明信息
declaration_text = "声明：以上内容仅代表基于规模人群数据的规律性结果，可能与个人实际情况存在差异，因此仅作为一般性参考，不可替代专业医疗建议。如需了解个人健康状况，请咨询专业医生或健康管理师，以获得针对性的指导和建议。本工具由许军教授课题组硕士研究生刘琛主要负责开发，目前处于测试阶段，其所有权归属于许军教授课题组。该工具现仅作为刘琛本人硕士毕业论文展示使用，未经许可，任何个人或组织不得将其用于商业用途或公开发布。本工具所涉及的代码、算法、模型等相关内容均受版权法保护，未经授权，禁止任何形式的复制、修改、传播或用于其他目的。如有任何疑问或需要进一步了解，请联系liuchen_scires@sina.cn。"
st.markdown(f"<span style='color:red;'><small>{declaration_text}</small></span>", unsafe_allow_html=True)
