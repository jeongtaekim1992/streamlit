import streamlit as st
import os
import base64
import pandas as pd
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder

# 상단 UI
st.markdown("""
    <style>
        .topbar {
            background-color: #4CAF50;
            padding: 20px;
            text-align: center;
            font-size: 28px;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 2px solid #a1c4fd;
            margin-bottom:30px;
        }
        .topbar span {
            font-family: 'Arial', sans-serif;
        }
    </style>
    <div class="topbar">
        <span>음성인식 기술을 활용한 서비스 기획</span>
    </div>
""", unsafe_allow_html=True)

# CSV 데이터 로드를 위한 단일 함수 정의
def load_data(path):
    return pd.read_csv(path)

# CSV 파일 로드
normalization_data = load_data("normalization.csv")
sample_data = load_data("sample_eval.csv")
average_data = load_data("avg_cers.csv")
combined_transcription_data = load_data("combined_transcription.csv")
faster_whisper_data = load_data("faster_whisper_compare.csv")
segments_data = load_data("segments.csv")

# txt 파일 로드
with open('summary.txt', 'r', encoding='utf-8') as file:
    summary_data = file.read().strip().replace('\r\n', '\n').replace('\r', '\n')

# "평가" 데이터를 정렬
def sort_company(company):
    return (company == "한국어", company)
sorted_data = average_data.sort_values(
    by=["Company", "Average_CER_Without_Punct_And_Space"],
    key=lambda col: col.map(sort_company) if col.name == "Company" else col,
    ascending=[True, True]
)

# 사이드바 메뉴
st.sidebar.title("목차")
dropdown_option = st.sidebar.selectbox("선택하세요:", 
    ["개요", "음성모델", "데이터셋", "평가", "최적화", "요약모델"])


# ─── 개요 ──────────────────────────────────────────────
if dropdown_option == "개요":
    st.markdown("""
        <div style="background-color:rgba(240,248,255,0.8); padding:25px; border-radius:15px; 
                    margin-top:30px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15); border: 2px solid #a1c4fd;">
            <h2 style="font-family: 'Verdana', sans-serif; font-size:24px; color:#1e2a47; text-align:center; 
                    margin-bottom:20px; font-weight:bold;">
                <span style="color:#5c6bc0;">📖 프로젝트 개요</span>
            </h2>
            <div style="font-family: 'Helvetica Neue', sans-serif; font-size:18px; color:#3a3a3a; line-height:1.8;">
                <p style="text-align:center; font-weight:500; margin-bottom:30px;">
                    많은 교회들이 매주 강단 메세지 녹취록을 작성합니다.<br>
                    제가 다니는 교회도 타이핑을 통해 녹취록을 전사합니다.<br>
                    <a href="http://www.dongboo.tv/main/sub.html?pageCode=50" target="_blank" style="text-decoration:none; color:#007bff; font-weight:bold;">Click here to visit the website</a>
                </p>
                <p style="text-align:center; font-weight:500; margin-bottom:30px;">
                    본 프로젝트는 유튜브 설교 영상을 자동 전사&요약을 위해 기획했습니다.<br>
                    성도들은 긴 영상 시청 시간을 효율적으로 관리하며,<br>
                    <span style="color:#d32f2f; font-weight:bold;">음성전사 및 요약</span>을 통해 성경 공부에 도움을 제공합니다.<br>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# <p style="text-align:center; font-weight:500; margin-bottom:30px;">
#     이 프로젝트의 첫 번째 단계로,<br> 
#     음성인식 모델들의 <span style="color:#d32f2f; font-weight:bold;">CER(오류율)</span>과 
#     <span style="color:#d32f2f; font-weight:bold;">전사 속도</span>를 비교하여,<br> 
#     최적화된 모델(예: Faster-Whisper)을 선정합니다.
# </p>
# <p style="font-size:16px; color:#5c6bc0; text-align:center;">
#     최종 음성인식 결과를 바탕으로 <span style="color:#5c6bc0;">요약, 질의응답, 번역</span> 등<br> 
#     다양한 부가 서비스와의 연계를 기대합니다.
# </p>



# ─── 모델선정 ──────────────────────────────────────────────
elif dropdown_option == "음성모델":
    st.markdown("""
        <div style="text-align: center; margin-bottom: 25px;">
            <h2 style="font-family: Arial, sans-serif; color: rgba(255,255,255,0.9);
                margin-bottom: 25px; font-size: 24px; font-weight: bold;">✨ 한국어 음성 인식 모델 소개</h2>
        </div>
        """, unsafe_allow_html=True)
    sections = [
        {"title": "Amazon", "description": "클라우드 선두주자 Amazon AWS의 음성 인식 서비스 Amazon Transcribe 소개합니다."},
        {"title": "Microsoft", "description": "Microsoft의 클라우드 기반 Azure Speech 음성 인식 서비스를 소개합니다."},
        {"title": "OpenAI", "description": "OpenAI에서 개발한 높은 정확도의 최신 음성 인식 모델 whisper를 소개합니다."},
        {"title": "META", "description": "META 한국어가 학습된 MMS 음성인식 모델과 SeamlessM4T 멀티모달을 소개합니다."},
        {"title": "Google", "description": "Google의 클라우드 기반 음성인식 서비스 Speech-to-Text의 버전 v1와 버전 v2를 소개합니다."},
        {"title": "ReturnZero", "description": "국내 음성 인식 스타트업에서 가장 빠르고 정확한 음성 인식을 자랑하는 리턴제로의 VITO를 소개합니다."},
        {"title": "ETRI", "description": "국내 인공 지능 R&D 산업을 지원하는 ETRI(한국전자통신연구원) 음성인식 기술을 소개합니다."},
        {"title": "NAVER", "description": "국내 1위 클라우드 네이버의 음성인식 기술 CLOVA Speech Recognition을 소개합니다."},
    ]
    for i in range(0, len(sections), 4):
        cols = st.columns(4)
        for col, sec in zip(cols, sections[i:i+4]):
            with col:
                st.markdown(f"""
                    <div style="padding: 10px; background-color: #d3d3d3;
                        border-radius: 15px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
                        height: 180px; margin: 0px;">
                        <h3 style="color: #333; font-size: 16px; font-weight: bold;
                            text-align: center; padding-top: 15px;">{sec['title']}</h3>
                        <p style="font-size: 14px; font-weight: bold; color: #666; text-align: left;">
                            {sec['description']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

# ─── 데이터셋 ───────────────────────────────────────────────
elif dropdown_option == "데이터셋":
    tab1, tab2, tab3, tab4 = st.tabs(["데이터 준비", "데이터 정규화", "데이터 평균값", "분포 시각화"])
    normalization_data = normalization_data.rename(columns={
        "cleand_text_char_count": "char_count", 
        "cleand_text_audio_length": "audio_length"
    })

    # 오디오 재생 HTML 생성 함수
    def get_audio_html(audio_path):
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return f'<audio controls style="width:100%; margin-top:10px;"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">Your browser does not support the audio element.</audio>'

    def tab1_info():
        audio_folder = "audio"
        audio_files = sorted(
            [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".mp3")],
            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x
        )
        audio_html = "".join(get_audio_html(audio) for audio in audio_files)
        container_html = f"""
        <div style="background-color:rgba(240,248,255,0.8); padding:25px; border-radius:15px;
                    margin-top:30px; box-shadow: 0px 4px 10px rgba(0,0,0,0.15); border: 2px solid #a1c4fd;">
            <h2 style="font-family: 'Verdana', sans-serif; font-size:24px; color:#1e2a47;
                text-align:center; margin-bottom:20px; font-weight:bold;">
                <span style="color:#5c6bc0;">📂 데이터셋 출처</span>
            </h2>
            <div style="margin-top:20px; text-align:center;">
                <div style="display:inline-block; width:99%; padding:12px; background-color:#fff; 
                    border-radius:30px; box-shadow:0 4px 8px rgba(0,0,0,0.1); border:1px solid #dcdcdc;">
                    <p style="margin:0; font-family: 'Helvetica Neue', sans-serif; font-size:18px;
                        color:#3a3a3a; text-align:center; line-height:1.8; width:100%;">
                        AIHUB 한국인 대화음성
                        <a href="https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=130" 
                           target="_blank" style="text-decoration:none; color:#007bff; font-weight:bold; margin-left:12px;">
                           🔗 데이터셋 확인
                        </a>
                    </p>
                </div>
                <div style="margin-top:10px;">{audio_html}</div>
            </div>
        </div>
        """
        st.markdown(container_html, unsafe_allow_html=True)

    def tab2_info():
        st.markdown("""
            <div style="background-color:#E6E6FA; border-radius:5px; box-shadow: 0px 2px 5px rgba(0,0,0,0.2); border:1px solid #888;">
                <h5 style="font-family:'Arial'; font-size:14px; color:#111; margin-top:25px; margin-left:25px; font-weight: bold;">데이터 설명</h5>
                <ul style="font-family:'Arial'; font-size:14px; line-height:1.4; color:#222; margin-left:25px; margin-bottom:25px; font-weight: bold;">
                    <li><strong>orginal_text</strong>: "(철자전사)/(발음전사)"</li>
                    <li><strong>cleand_text</strong>: 특수문자 제거 + 철자전사 정규화</li>
                    <li><strong>difference</strong>: 변경된 텍스트만 표시</li>
                    <li><strong>char_count</strong>: 문자 수</li>
                    <li><strong>audio_length</strong>: 오디오 길이</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab1:
        tab1_info()

    with tab2:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        norm_df = normalization_data.drop(
            columns=["audio_filepath", "remove_punct_and_space", "restore_punct_and_space"]
        ).reset_index()
        grid_options = GridOptionsBuilder.from_dataframe(norm_df)
        grid_options.configure_default_column(minWidth=350, maxWidth=500, wrapText=False, resizable=True,
                                              cellStyle={"fontSize": "12px"})
        for col in ["index", "difference", "char_count", "audio_length"]:
            grid_options.configure_column(col, maxWidth=115,
                                          cellStyle={"textAlign": "left"},
                                          headerClass={"textAlign": "left"})
        AgGrid(norm_df, gridOptions=grid_options.build(), height=300)
        tab2_info()

    with tab3:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        avg_char = normalization_data["char_count"].mean()
        avg_audio = normalization_data["audio_length"].mean()
        avg_table = pd.DataFrame({
            "Metric": ["평균 문자 수", "평균 오디오 길이"],
            "Value": [f"{avg_char:.2f} 글자", f"{avg_audio:.2f} 초"]
        }).reset_index()
        grid_options = GridOptionsBuilder.from_dataframe(avg_table)
        grid_options.configure_default_column(minWidth=350, maxWidth=500, wrapText=False, resizable=True,
                                              cellStyle={"fontSize": "12px"})
        grid_options.configure_column("index", minWidth=75, maxWidth=100,
                                      cellStyle={"textAlign": "left"},
                                      headerClass={"textAlign": "left"})
        AgGrid(avg_table, gridOptions=grid_options.build(), height=105)

    with tab4:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        def create_line_chart(data, field, bin_step, chart_color, x_title, chart_title):
            return alt.Chart(data).transform_bin(
                f"binned_{field}", field=field, bin=alt.Bin(step=bin_step)
            ).transform_aggregate(
                count="count()", groupby=[f"binned_{field}"]
            ).mark_line(color=chart_color, point=True).encode(
                x=alt.X(f"binned_{field}:Q", title=x_title),
                y=alt.Y("count:Q", title="빈도", axis=alt.Axis(titleAngle=0, titlePadding=30)),
                tooltip=[alt.Tooltip(f"binned_{field}:Q", title=x_title),
                         alt.Tooltip("count:Q", title="빈도")]
            ).properties(width=700, height=300, title=chart_title)\
             .configure_axis(labelFontSize=12, titleFontSize=12)\
             .configure_title(fontSize=14)
        st.altair_chart(create_line_chart(normalization_data, "char_count", 1, "darkorange",
                                            "문자 수(글자)", "문자 수 분포"), use_container_width=True)
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.altair_chart(create_line_chart(normalization_data, "audio_length", 0.5, "coral",
                                            "오디오 길이(초)", "오디오 길이 분포"), use_container_width=True)

# ─── 평가 ─────────────────────────────────────────────────
elif dropdown_option == "평가":
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["베스트 모델 선정", "평가 지표 선정", "평가 비교", "개선율 비교", "실제 전사 결과"])
    
    def tab1_info():
        st.markdown(
            """
            <div style="background-color:rgba(240, 248, 255, 0.8); padding:25px; border-radius:15px; 
                        margin-top:30px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15); border: 2px solid #a1c4fd;">
                <h2 style="font-family: 'Verdana', sans-serif; font-size:24px; color:#1e2a47; text-align:center; 
                        margin-bottom:20px; font-weight:bold;">
                    <span style="color:#5c6bc0;">🏆 베스트 모델</span>
                </h2>
                <div style="font-family: 'Helvetica Neue', sans-serif; font-size:18px; color:#3a3a3a; line-height:1.8;">
                    <p style="text-align:center; font-weight:500; margin-bottom:30px;">
                        Whisper-turbo<br>
                        Whisper-large-v3-turbo<br>
                        <span style="font-size:18px; color:#7a7a7a;">구두점 포함된 평균 CER :</span>
                        <span style="font-size:20px; color:#d32f2f;">3%</span><br>
                        <span style="font-size:18px; color:#7a7a7a;">구두점과 띄어쓰기가 제거된 평균 CER :</span>
                        <span style="font-size:20px; color:#d32f2f;">1%</span>
                    </p>
                    <p style="font-size:16px; color:#5c6bc0; text-align:center;">
                        <strong>
                            Whisper-turbo 모델은 Whisper-large-v2의 경량 버전이며,<br>
                            Whisper-large-v3-turbo는 가장 최신 경량 모델입니다.
                        </strong>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    def tab2_info():
        st.markdown("""
            <div style="background-color:rgba(190,210,220,0.95); border-radius:5px;
                        0box-shadow: 0px 2px 5px rgba(0,0,0,0.2); border:1px solid #888;">
                <h5 style="font-family:'Arial'; font-size:14px; color:#111; margin-top:25px; margin-left:25px; font-weight: bold;">데이터 설명</h5>
                <ul style="font-family:'Arial'; font-size:14px; line-height:1.4; color:#222; margin-left:25px; margin-bottom:25px; font-weight: bold;">
                    <li>WER: 단어 단위 오류율.</li>
                    <li>CER: 문자 단위 오류율.</li>
                    <li>띄어쓰기와 철자 오류와 같은 사소한 차이는 WER에 큰 영향을 주지만, CER은 비교적 안정적입니다.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    def tab3_info():
        st.markdown("""
            <div style="background-color:rgba(190,210,220,0.95); border-radius:5px;
                        0box-shadow: 0px 2px 5px rgba(0,0,0,0.1); border:1px solid #999;">
                <h5 style="font-family:'Arial'; font-size:14px; color:#111; margin-top:25px; margin-left:25px; font-weight: bold;">CER 차트 설명</h5>
                <ul style="font-family:'Arial'; font-size:14px; line-height:1.4; color:#222; margin-left:25px; margin-bottom:25px; font-weight: bold;">
                    <li><strong>Average CER</strong>: 구두점 및 띄어쓰기를 포함한 오류율</li>
                    <li><strong>Average CER Without Punct And Space</strong>: 구두점 및 띄어쓰기를 제거한 오류율</li>
                    <li><strong>Model with lowest CER</strong>: whisper_turbo 및 Whisper_large-v3-turbo 모델 (1% 오류율)</li>
                    <li><strong>Tool-Tip</strong>: 마우스를 차트에 가져가면 추가 정보 확인 가능</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    def tab4_info_1():
        st.markdown("""
            <div style="background-color:rgba(190,210,220,0.95); border-radius:5px;
                        0box-shadow: 0px 2px 5px rgba(0,0,0,0.1); border:1px solid #999;">
                <h5 style="font-family:'Arial'; font-size:14px; color:#111; margin-top:25px; margin-left:25px; font-weight: bold;">CER 차트 설명</h5>
                <ul style="font-family:'Arial'; font-size:14px; line-height:1.4; color:#222; margin-left:25px; margin-bottom:25px; font-weight: bold;">
                    <li><strong>Average CER Improvement</strong>: 개선율 (띄어쓰기 및 구두점을 제거해서 나타난 결과)</li>
                    <li>띄어쓰기와 구두점 제거는 라벨과 전사 결과 모두에 적용하여 비교</li>
                    <li>전반적으로 약 2%~7% 개선</li>
                    <li><strong>Tool-Tip</strong>: 마우스를 차트에 가져가면 추가 정보 확인 가능</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    def tab4_info_2():
        st.markdown("""
            <div style="background-color:rgba(190,210,220,0.95); border-radius:5px;
                        0box-shadow: 0px 2px 5px rgba(0,0,0,0.1); border:1px solid #999;">
                <h5 style="font-family:'Arial'; font-size:14px; color:#111; margin-top:25px; margin-left:25px; font-weight: bold;">구두점과 띄어쓰기 제거의 장점</h5>
                <ul style="font-family:'Arial'; font-size:14px; line-height:1.4; color:#222; margin-left:25px; margin-bottom:25px; font-weight: bold;">
                    <li>음성인식 모델의 한국어 띄어쓰기와 구두점 표현의 한계를 보완합니다.</li>
                    <li>오직 텍스트에 대한 원초적인 평가만을 강조합니다</li>
                    <li>불필요한 오류율 증가를 방지합니다</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    model_order = sorted_data["Model"].tolist()
    combined_transcription_data = combined_transcription_data.loc[:, ["cleand_text"] + model_order]
    
    with tab1:
        tab1_info()
    
    with tab2:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        sample_df = sample_data.reset_index()
        grid_options = GridOptionsBuilder.from_dataframe(sample_df)
        grid_options.configure_default_column(minWidth=250, maxWidth=350, wrapText=False, resizable=True,
                                              cellStyle={"fontSize": "12px"})
        grid_options.configure_column("index", minWidth=70, maxWidth=70,
                                      cellStyle={"textAlign": "left"},
                                      headerClass={"textAlign": "left"})
        for col in ["WER", "CER"]:
            grid_options.configure_column(col, minWidth=70, maxWidth=70,
                                          cellStyle={"textAlign": "left"},
                                          headerClass={"textAlign": "left"})
        AgGrid(sample_df, gridOptions=grid_options.build(), height=160)
        tab2_info()

    with tab3:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state.page = 1 
        def next_page():
            st.session_state.page += 1 
        def prev_page():
            st.session_state.page -= 1 

        if st.session_state.page == 1:
            st.button("다음 페이지", on_click=next_page)

            chart_avg_cer = alt.Chart(sorted_data).mark_bar(opacity=0.7, cornerRadiusEnd=5).encode(
                x=alt.X("Average_CER:Q", title=None,
                        axis=alt.Axis(format=".1%", labelFontSize=12)),
                y=alt.Y("Model:N", sort=model_order, title=None, axis=alt.Axis(labelFontSize=12)),
                color=alt.Color("Company:N", scale=alt.Scale(scheme="category10"),
                                legend=alt.Legend(title="기업명", titleFontSize=14, labelFontSize=12,
                                                orient="top", offset=10, titlePadding=10, labelPadding=25)),
                tooltip=["Company", "Model", "Average_CER"]
            ).properties(width=500, height=85)
            facet_chart = chart_avg_cer.facet(
                row=alt.Row("Company:N", header=alt.Header(title="모델명", titleFontSize=14,
                                                            titleAngle=0, labels=False, titlePadding=-30)),
                spacing=10
            ).resolve_scale(x='shared', y='independent').properties(
                title="<구두점과 띄어쓰기가 포함된 평균 CER>"
            ).configure_title(
                font="Helvetica Neue",
                fontSize=18,
                fontWeight="bold",
                anchor="middle",
                offset=35
            )
            st.altair_chart(facet_chart, use_container_width=True)
            tab3_info()

        elif st.session_state.page == 2:
            st.button("이전 페이지", on_click=prev_page)
            tab2_data = sorted_data.copy()
            tab2_data["Sticker"] = tab2_data["Model"].map({"whisper_turbo": "★ Model with lowest CER"}).fillna("")
            tab2_data = tab2_data.set_index("Model").loc[model_order].reset_index()
            chart_avg_cer_np = alt.Chart(tab2_data).mark_bar(opacity=0.7, cornerRadiusEnd=5).encode(
                x=alt.X("Average_CER_Without_Punct_And_Space:Q", 
                        title=None,
                        axis=alt.Axis(format=".1%", labelFontSize=12)),
                y=alt.Y("Model:N", sort=model_order, title=None, axis=alt.Axis(labelFontSize=12)),
                color=alt.Color("Company:N", scale=alt.Scale(scheme="category10"),
                                legend=alt.Legend(title="기업명", titleFontSize=14, labelFontSize=12,
                                                  orient="top", offset=10, titlePadding=10, labelPadding=25)),
                tooltip=["Company", "Model", "Average_CER_Without_Punct_And_Space"]
            ).properties(width=500, height=85)
            sticker_layer = alt.Chart(tab2_data).mark_text(fontSize=12, color="white",
                                                             dx=25, align="left", baseline="middle").encode(
                y=alt.Y("Model:N", sort=model_order, title=None),
                x=alt.value(0),
                text=alt.Text("Sticker:N")
            )
            combined_chart = alt.layer(chart_avg_cer_np, sticker_layer)
            facet_chart = combined_chart.facet(
                row=alt.Row("Company:N", header=alt.Header(title="모델명", titleFontSize=14,
                                                            titleAngle=0, labels=False, titlePadding=-30)),
                spacing=10
            ).resolve_scale(x='shared', y='independent').properties(
                title="<구두점과 띄어쓰기가 제거된 평균 CER>"
            ).configure_title(
                font="Helvetica Neue",
                fontSize=18,
                fontWeight="bold",
                anchor="middle",
                offset=35
            )
            st.altair_chart(facet_chart, use_container_width=True)
            tab3_info()

    with tab4:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state.page = 1 

        def next_page():
            st.session_state.page += 1 

        def prev_page():
            st.session_state.page -= 1 

        if st.session_state.page == 1:
            st.button("다음 페이지", on_click=next_page, key="next_tab4")

            chart_cer_improve = alt.Chart(sorted_data).mark_circle(size=200).encode(
                x=alt.X("Average_CER_Improvement:Q", title=None,
                        axis=alt.Axis(labelFontSize=12, tickCount=9, format=".1%", grid=True),
                        scale=alt.Scale(domain=[0.01, 0.08])),
                y=alt.Y("Model:N", title="모델명", sort=model_order,
                        axis=alt.Axis(title=None, labelFontSize=12, grid=True)),
                color=alt.Color("Company:N", scale=alt.Scale(scheme="category10"),
                                legend=alt.Legend(title="기업명", titleFontSize=14, labelFontSize=12,
                                                orient="top", offset=10, titlePadding=10, labelPadding=25)),
                tooltip=["Company", "Model", "Average_CER_Improvement"]
            ).properties(width=500, height=85)
            facet_chart = chart_cer_improve.facet(
                row=alt.Row("Company:N", header=alt.Header(title="모델명", titleFontSize=14,
                                                            titleAngle=0, labels=False, titlePadding=-30)),
                spacing=10
            ).resolve_scale(x='shared', y='independent').properties(
                title="<구두점과 띄어쓰기를 제거했을 때 평균 개선율>"
            ).configure_title(
                font="Helvetica Neue",
                fontSize=18,
                fontWeight="bold",
                anchor="middle",
                offset=35
            )
            st.altair_chart(facet_chart, use_container_width=True)
            tab4_info_1()

        elif st.session_state.page == 2:
            st.button("이전 페이지", on_click=prev_page, key="prev_tab4")

            chart_cer_improve = alt.Chart(sorted_data).mark_circle(size=200).encode(
                x=alt.X("Average_CER_Improvement:Q", title=None,
                        axis=alt.Axis(labelFontSize=12, tickCount=9, format=".1%", grid=True),
                        scale=alt.Scale(domain=[0.01, 0.08])),
                y=alt.Y("Model:N", title="모델명", sort=model_order,
                        axis=alt.Axis(title=None, labelFontSize=12, grid=True)),
                color=alt.Color("Company:N", scale=alt.Scale(scheme="category10"),
                                legend=alt.Legend(title="기업명", titleFontSize=14, labelFontSize=12,
                                                orient="top", offset=10, titlePadding=10, labelPadding=25)),
                tooltip=["Company", "Model", "Average_CER_Improvement"]
            ).properties(width=500, height=85)
            facet_chart = chart_cer_improve.facet(
                row=alt.Row("Company:N", header=alt.Header(title="모델명", titleFontSize=14,
                                                            titleAngle=0, labels=False, titlePadding=-30)),
                spacing=10
            ).resolve_scale(x='shared', y='independent').properties(
                title="<구두점과 띄어쓰기를 제거했을 때 평균 개선율>"
            ).configure_title(
                font="Helvetica Neue",
                fontSize=18,
                fontWeight="bold",
                anchor="middle",
                offset=35
            )
            st.altair_chart(facet_chart, use_container_width=True)
            tab4_info_2()
            
    with tab5:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        trans_df = combined_transcription_data.reset_index()
        grid_options = GridOptionsBuilder.from_dataframe(trans_df)
        grid_options.configure_default_column(minWidth=350, maxWidth=500, wrapText=False, resizable=True,
                                              cellStyle={"fontSize": "12px"})
        grid_options.configure_column("index", minWidth=75, maxWidth=100,
                                      cellStyle={"textAlign": "left"},
                                      headerClass={"textAlign": "left"})
        grid_options.configure_column("cleand_text", cellStyle={"backgroundColor": "rgba(169, 169, 169, 0.1)"})
        AgGrid(trans_df, gridOptions=grid_options.build(), height=400)

# ─── 최적화 ───────────────────────────────────────────────
elif dropdown_option == "최적화":
    # tab1, tab2, tab3, tab4, tab5 = st.tabs(["최적화 모델 소개", "테스트 환경", "추론 속도 비교", "실제 전사 결과", "기술연계"])
    tab1, tab2, tab3, tab4 = st.tabs(["최적화 모델 소개", "테스트 환경", "추론 속도 비교", "실제 전사 결과"])
    faster_whisper_data = faster_whisper_data.melt(
        id_vars=['model', 'type', 'batch'],
        value_vars=['A100', 'T4×2'],
        var_name='gpu_type', value_name='performance(sec)'
    )
    faster_whisper_data['performance(sec)'] = pd.to_numeric(faster_whisper_data['performance(sec)'], errors='coerce')
    faster_whisper_data = faster_whisper_data.dropna(subset=['performance(sec)'])
    faster_whisper_data['performance(sec)'] = faster_whisper_data['performance(sec)'].apply(round)
    faster_whisper_data['gpu_model'] = faster_whisper_data['gpu_type'] + ' - ' + faster_whisper_data['model']
    
    test_env = {
        "Test Environment": ["Google Colab", "Kaggle"],
        "Model Size": ["large-v3", "large-v3-turbo"],
        "GPU": ["NVIDIA A100", "NVIDIA T4 × 2"],
        "Batch Size": [8, 16],
        "Data Type": ["FP16", "FP32"]
    }
    test_env = pd.DataFrame(test_env)

    def tab1_info():
        st.markdown(
            """
            <div style="background-color:rgba(240, 248, 255, 0.8); padding:25px; border-radius:15px; 
                        margin-top:30px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15); border: 2px solid #a1c4fd;">
                <h2 style="font-family: 'Verdana', sans-serif; font-size:24px; color:#1e2a47; text-align:center; 
                        margin-bottom:20px; font-weight:bold;">
                    <span style="color:#5c6bc0;">🚀 Faster Whisper</span>
                </h2>
                <div style="font-family: 'Helvetica Neue', sans-serif; font-size:18px; color:#3a3a3a; line-height:1.8;">
                    <p style="text-align:center; font-weight:500; margin-bottom:30px;">
                        Whisper 모델을 기반으로 최적화 엔진 CTranslate2를 사용합니다.<br>
                        빠른 추론 속도와 경량화로 Openai/Whisper 대비 4배 더 빠른 속도를 자랑합니다.
                    </p>
                    <p style="font-size:16px; color:#5c6bc0; text-align:center;">
                        <strong>Faster Whisper 모델은 약 34분 오디오 데이터를 
                            <span style="font-size:20px; color:#d32f2f;">6초</span> 만에 전사했습니다.
                        </strong>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with tab1:
        tab1_info()
    with tab2:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        grid_options = GridOptionsBuilder.from_dataframe(test_env)
        grid_options.configure_default_column(minWidth=150, maxWidth=150, wrapText=False, resizable=True,
                                              cellStyle={"fontSize": "12px"})

        grid_options.configure_column("Test Environment", flex=1, cellStyle={"textAlign": "left"},
                                    headerClass={"textAlign": "left"})
        grid_options.configure_column("Model Size", flex=1, cellStyle={"textAlign": "left"},
                                    headerClass={"textAlign": "left"})
        grid_options.configure_column("GPU", flex=1, cellStyle={"textAlign": "left"},
                                    headerClass={"textAlign": "left"})
        grid_options.configure_column("Batch Size", flex=1, cellStyle={"textAlign": "left"},
                                    headerClass={"textAlign": "left"})
        grid_options.configure_column("Data Type", flex=1, cellStyle={"textAlign": "left"},
                                    headerClass={"textAlign": "left"})

        AgGrid(test_env, gridOptions=grid_options.build(), height=105)
    with tab3:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        chart = alt.Chart(faster_whisper_data).mark_bar(opacity=0.7, cornerRadiusEnd=5).encode(
            x=alt.X('gpu_model:N', title=None,
                    sort=alt.EncodingSortField(field='performance(sec)', op='mean', order='ascending'),
                    axis=alt.Axis(labelFontSize=12)),
            y=alt.Y('performance(sec):Q', title=None,
                    axis=alt.Axis(labelFontSize=12),
                    scale=alt.Scale(type='linear', domain=[0, 300], clamp=True)),
            color=alt.Color('gpu_type:N', scale=alt.Scale(scheme='category10'),
                            legend=alt.Legend(title="GPU", titleFontSize=14, labelFontSize=12,
                                              titlePadding=5, offset=15)),
            tooltip=['model', 'type', 'batch', 'gpu_type', 'performance(sec)']
        ).properties(width=250, height=250)
        facet_chart = chart.facet(
            row=alt.Row('type:N', header=alt.Header(title=None, labelFontSize=14, labelAngle=0, labelPadding=-15)),
            column=alt.Column('batch:N', header=alt.Header(title=None, labelFontSize=14, labelPadding=15)),
            spacing=20
        )
        st.altair_chart(facet_chart, use_container_width=False)
    with tab4:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.video("https://www.youtube.com/watch?v=3kzLZjKAvzQ")
        grid_options = GridOptionsBuilder.from_dataframe(segments_data)
        grid_options.configure_default_column(wrapText=False, resizable=True, cellStyle={"fontSize": "12px"})
        for col in ["start", "end"]:
            grid_options.configure_column(col, minWidth=70, maxWidth=70,
                                          cellStyle={"textAlign": "left"},
                                          headerClass={"textAlign": "left"})
        grid_options.configure_column("text", flex=1,
                                      cellStyle={"textAlign": "left"},
                                      headerClass={"textAlign": "left"})
        AgGrid(segments_data, gridOptions=grid_options.build(), height=160, fit_columns_on_grid_load=True)
  

elif dropdown_option == "요약모델":
    tab1, tab2 = st.tabs(["모델 소개", "요약 결과 비교"])

    with tab1:
        st.markdown(
            """
            <div style="background-color:rgba(240, 248, 255, 0.8); padding:25px; border-radius:15px; 
                        margin-top:30px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15); border: 2px solid #a1c4fd;">
                <h2 style="font-family: 'Verdana', sans-serif; font-size:24px; color:#1e2a47; text-align:center; 
                        margin-bottom:20px; font-weight:bold;">
                    <span style="color:#5c6bc0;">🤖 Gemma</span>
                </h2>
                <div style="font-family: 'Helvetica Neue', sans-serif; font-size:18px; color:#3a3a3a; line-height:1.8;">
                    <p style="text-align:center; font-weight:500; margin-bottom:30px;">
                        리턴제로에서 Google LLM Gemma를 기반으로<br>
                        요약 서비스를 위한 한국어 파인튜닝 모델을 발표했습니다 <br>
                        스크립트 요약에 <span style="color:#d32f2f;">"rtzr/ko-gemma-2-9b-it"</span>를 사용합니다.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab2:        
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        
        def create_text_column(title, text, char_count):
            st.markdown(f"""
                <div style="background-color:rgba(240, 248, 255, 0.8); padding:15px; border-radius:10px; 
                            margin-bottom:15px; border: 1px solid #a1c4fd;">
                    <h3 style="font-family: 'Arial', sans-serif; font-size:18px; color:#1e2a47; 
                            text-align:center; margin:0; font-weight:bold;">
                        {title}
                    </h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(
                f"""
                <div style="background-color: white; padding: 15px; border-radius: 5px; 
                            border: 1px solid #ddd; margin-bottom: 10px;">
                    <div style="font-family: 'Arial', sans-serif; font-size: 14px; color: #666;">
                        총 문자 수: {char_count}자
                    </div>
                </div>
                <div style="background-color: white; padding: 15px; border-radius: 5px; 
                            border: 1px solid #ddd; height: 400px; overflow-y: auto;">
                    <p style="font-family: 'Arial', sans-serif; font-size: 14px; 
                            line-height: 1.6; color: rgb(33, 33, 33);">
                        {text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            full_text = ' '.join(segments_data['text'].tolist())
            create_text_column("원본 텍스트", full_text, len(full_text))

        with col2:
            create_text_column("요약 결과", summary_data, len(summary_data))

        