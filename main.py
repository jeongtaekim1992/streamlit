import streamlit as st
import pandas as pd
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder

st.markdown(
    """
    <style>
        .topbar {
            background-color: #4CAF50;
            padding: 10px;
            text-align: center;
            font-size: 24px;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
        .chart-container {
            margin-top: 50px;  
        }
    </style>
    <div class="topbar">
        <span>음성인식 모델 CER 비교</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="chart-container"></div>', unsafe_allow_html=True)

#######################################################################################################
#######################################################################################################

# 데이터셋 데이터 로드
path = "normalization.csv"
def load_data(path):
    return pd.read_csv(path)
normalization_data = load_data(path)

# 평가지표 데이터 로드
path = "sample_eval.csv"
def load_data(path):
    return pd.read_csv(path)
sample_data = load_data(path)

# 평가 데이터 로드
path = "avg_cers.csv"
def load_data(path):
    return pd.read_csv(path)
average_data = load_data(path)

path = "combined_transcription.csv"
def load_data(path):
    return pd.read_csv(path)
combined_transcription_data = load_data(path)

# 평가 데이터 정렬
# sorted_data = average_data.sort_values(by=["Company", "Average_CER_Without_Punct_And_Space"], ascending=[True, True])
def sort_company(company):
    return (company == "한국어", company)  # "한국어"는 True로 반환되어 가장 뒤로 이동
sorted_data = average_data.sort_values(
    by=["Company", "Average_CER_Without_Punct_And_Space"],
    key=lambda col: col.map(sort_company) if col.name == "Company" else col,
    ascending=[True, True]
)

model_order = sorted_data["Model"].tolist()
combined_transcription_data = combined_transcription_data.loc[:, ["cleand_text"] + model_order]

# 경량화 데이터 로드
path = "faster_whisper_compare.csv"
def load_data(path):
    return pd.read_csv(path)
faster_whisper_data = load_data(path)

# 'T4*2' 컬럼명을 안전한 이름으로 변경
faster_whisper_data = faster_whisper_data.rename(columns={'T4*2': 'T4_2'})

# 데이터 전처리
faster_whisper_data = faster_whisper_data.melt(id_vars=["model", "type", "batch"], value_vars=["A100", "T4_2"], var_name="GPU", value_name="Value")
faster_whisper_data['Value'] = pd.to_numeric(faster_whisper_data['Value'], errors='coerce')  # 'out of memory' 값 처리

#######################################################################################################
#######################################################################################################

# 사이드바 추가
st.sidebar.title("목차")
dropdown_option = st.sidebar.selectbox("선택하세요:", ["모델선정", "데이터셋", "평가지표", "평가"])

# if dropdown_option == "개요":

#     tab1 = st.tabs(["개요"])

if dropdown_option == "모델선정":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 25px;">
            <h2 style="font-family: Arial, sans-serif; color: rgba(255, 255, 255, 0.9); margin-bottom: 25px; font-size: 24px; font-weight: bold;">✨ 한국어 음성 인식 모델 소개</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    sections = [
        {"title": "Amazon", "description": "클라우드 선두주자 Amazon AWS의 음성 인식 서비스 Amazon Transcribe 소개합니다."},
        {"title": "Microsoft", "description": "Microsoft의 클라우드 기반 Azure Speech 음성 인식 서비스를 소개합니다."},
        {"title": "OpenAI", "description": "OpenAI에서 개발한 높은 정확도의 최신 음성 인식 모델 whisper를 소개합니다."},
        {"title": "META", "description": "META 한국어가 학습된 MMS 음성인식 모델과 SeamlessM4T 멀티모달을 소개합니다."},
        {"title": "Google", "description": "Google의 클라우드 기반 음성인식 서비스 Speech-to-Text의 버전 v1와 버전 v2를 소개합니다."},
        {"title": "ReturnZero", "description": "국내 음성 인식 스타트업에서 가장 빠르고 정확한 음성 인식을 자랑하는 리턴제로의 VITO를 소개합니다."},
        {"title": "ETRI", "description": "국내 인공 지능 R&D 산업을 지원하는 ETRI(한국전자통신연구원) 음성인식 기술을 소개합니다."},
        {"title": "NAVER", "description": "국내 1위 클라우드 네이버의 음성인식 기술 CLOVA Speech Recognition을 소개합니다."},
        # {"title": "example", "description": "준비중입니다."}
    ]

    # 섹션을 4개씩 나누기
    for i in range(0, len(sections), 4):
        cols = st.columns(4)
        for col, section in zip(cols, sections[i:i+4]):
            with col:
                st.markdown(
                    f"""
                    <div style="
                        padding: 10px; 
                        background-color: #d3d3d3; 
                        border-radius: 15px; 
                        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); 
                        height: 180px;
                        box-sizing: border-box;
                        margin: 0px;  /* 각 박스 사이 간격 유지 */
                    ">
                        <h3 style="
                            color: #333; 
                            font-size: 16px; 
                            font-weight: bold;                        
                            text-align: center; 
                            padding-top: 15px;  
                        ">
                            {section['title']}
                        </h3>
                        <p style=
                            "font-size: 14px; 
                            font-weight: bold; 
                            color: #666; 
                            text-align: left;">
                            {section['description']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        # 각 행 사이 간격 조정
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

elif dropdown_option == "데이터셋":
    
    tab1, tab2, tab3, tab4 = st.tabs(["데이터셋", "평균값", "분포 시각화", "데이터셋 출처"])

    def tab1_info():
        st.markdown(
            """
            <div style="background-color:rgba(190, 210, 220, 0.95); padding:15px; border-radius:10px; 
                        margin-top:20px; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2); border: 1px solid #888;">
                <h5 style="font-family:Arial; font-size:14px; color:#111; margin-top:15px;">데이터 설명</h5>
                <ul style="font-size:13px; line-height:1.4; color:#222; margin-left:15px;">
                    <li><strong>orginal_text</strong> : 라벨.</li>
                    <li><strong>cleand_text</strong> : 특수 문자 제거, 이중전사 (철자전사)/(발음전사) 구성 중, 철자전사로 정규화.</li>
                    <li><strong>difference</strong> : 변경된 cleand_text 체크.</li>
                    <li><strong>cleand_text_char_count</strong> : 텍스트 문자 수.</li>
                    <li><strong>cleand_text_audio_length</strong> : 오디오 파일 길이.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab1:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )

        filtered_data = normalization_data.drop(columns=["audio_filepath", "remove_punct_and_space", "restore_punct_and_space"])
        combined_transcription_data = filtered_data.reset_index()
        grid_options = GridOptionsBuilder.from_dataframe(combined_transcription_data)

        grid_options.configure_default_column(
            minWidth=350,  # 최소 너비
            maxWidth=500,  # 최대 너비
            wrapText=False,  # 텍스트 줄바꿈 비활성화
            resizable=True,   # 컬럼 크기 조정 가능
            cellStyle={"fontSize": "12px"}
        )

        columns_to_style = ["index", "difference", "cleand_text_char_count", "cleand_text_audio_length"]

        for column in columns_to_style:
            grid_options.configure_column(
                column,
                minWidth=75,
                maxWidth=100,
                cellStyle={"textAlign": "left"},
                headerClass={"textAlign": "left"},
            )

        grid_response = AgGrid(
            combined_transcription_data,
            gridOptions=grid_options.build(),
            height=400
        )

        tab1_info()

        # def highlight_rows(row):
        #     if row["difference"]: 
        #         return ["background-color: rgba(169, 169, 169, 0.1)"] * len(row) 
        #     else:
        #         return [""] * len(row)
    
        # styled_data = filtered_data.style.apply(highlight_rows, axis=1).format({"cleand_text_audio_length": "{:.2f}"})

        # st.write(styled_data)

        # st.markdown("""
        #     #### 데이터셋 정규화 설명
        #     - 특수문자 제거
        #     - 이중전사 (철자전사)/(발음전사) 구성 중, 철자전사로 정규화

        #     #### 테이블 설명
        #     - **difference**: 전처리 후 변경된 데이터 여부.
        #     - **cleand_text_char_count**: 텍스트의 문자 수.
        #     - **cleand_text_audio_length**: 오디오 파일의 길이(초 단위).
        # """, unsafe_allow_html=True)

    with tab2:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )

        avg_char_count = normalization_data["cleand_text_char_count"].mean()
        avg_audio_length = normalization_data["cleand_text_audio_length"].mean()

        avg_table = pd.DataFrame({
            "Metric": ["평균 문자 수", "평균 오디오 길이"],
            "Value": [f"{avg_char_count:.2f} 글자", f"{avg_audio_length:.2f} 초"]
        })

        combined_transcription_data = avg_table.reset_index()
        grid_options = GridOptionsBuilder.from_dataframe(combined_transcription_data)
        

        grid_options.configure_default_column(
            minWidth=350,  # 최소 너비
            maxWidth=500,  # 최대 너비
            wrapText=False,  # 텍스트 줄바꿈 비활성화
            resizable=True,   # 컬럼 크기 조정 가능
            cellStyle={"fontSize": "12px"}
        )

        grid_options.configure_column(
            "index",
            minWidth=75,
            maxWidth=100,
            cellStyle={"textAlign": "left"},
            headerClass={"textAlign": "left"},
        )

        grid_response = AgGrid(
            combined_transcription_data,
            gridOptions=grid_options.build(),
            height=105
        )

    with tab3:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )

        def create_line_chart(data, field, bin_step, chart_color, x_title, chart_title):
            return alt.Chart(data).transform_bin(
                f"binned_{field}", field=field, bin=alt.Bin(step=bin_step)
            ).transform_aggregate(
                count="count()", groupby=[f"binned_{field}"]
            ).mark_line(color=chart_color, point=True).encode(
                x=alt.X(f"binned_{field}:Q", title=x_title),
                y=alt.Y("count:Q", title="빈도", axis=alt.Axis(titleAngle=0, titlePadding=30)),
                tooltip=[
                    alt.Tooltip(f"binned_{field}:Q", title=x_title), 
                    alt.Tooltip("count:Q", title="빈도")
                ]
            ).properties(
                width=700, height=300, title=chart_title
            )

        # 문자 수 분포 차트
        char_line = create_line_chart(
            data=normalization_data,
            field="cleand_text_char_count",
            bin_step=1,
            chart_color="darkorange",
            x_title="문자 수(글자)",
            chart_title="문자 수 분포"
        ).configure_axis(
            labelFontSize=12,  # 축 레이블 폰트 크기
            titleFontSize=12   # 축 제목 폰트 크기
        ).configure_title(
            fontSize=14  # 차트 제목 폰트 크기
        )
        st.altair_chart(char_line, use_container_width=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # 오디오 길이 분포 차트
        audio_line = create_line_chart(
            data=normalization_data,
            field="cleand_text_audio_length",
            bin_step=0.5,
            chart_color="coral",
            x_title="오디오 길이(초)",
            chart_title="오디오 길이 분포"
        ).configure_axis(
            labelFontSize=12,  # 축 레이블 폰트 크기
            titleFontSize=12   # 축 제목 폰트 크기
        ).configure_title(
            fontSize=14  # 차트 제목 폰트 크기
        )

        st.altair_chart(audio_line, use_container_width=True)

    with tab4:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="background-color: #dcdcdc; padding: 15px; border-radius: 10px; 
                        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); border: 1px solid #aaa;">
                <h5 style="font-family: Arial, sans-serif; color: #333; font-size: 24px;">📂 데이터셋 출처</h5>
                <div style="margin-top: 15px;">
                    <div style="margin-bottom: 10px; padding: 10px; background-color: #e6e6e6; border-radius: 8px; 
                                box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.1); border: 1px solid #bbb;">
                        <h5 style="margin: 0; font-size: 14px; color: #222;">AIHUB 한국인 대화음성</h5>
                        <p style="margin: 5px 0; font-size: 14px; color: #555;"> 
                            <a href="https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=130" target="_blank" 
                            style="text-decoration: none; color: #007bff;">click</a>
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

elif dropdown_option == "평가지표":

    tab1, tab2 = st.tabs(["평가 지표 선정", " "])

    def tab1_info():
        st.markdown(
            """
            <div style="background-color:rgba(190, 210, 220, 0.95); padding:15px; border-radius:10px; 
                        margin-top:20px; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2); border: 1px solid #888;">
                <h5 style="font-family:Arial; font-size:14px; color:#111; margin-top:15px;">데이터 설명</h5>
                <ul style="font-size:13px; line-height:1.4; color:#222; margin-left:15px;">
                    <li><strong>WER(Word Error Rate)</strong> : 단어 단위 오류율.</li>
                    <li><strong>CER(Character Error Rate)</strong> : 문자 단위 오류율.</li>
                </ul>
                <ul style="font-size:13px; color:#333; margin-top:10px; line-height:1.4; margin-left:15px;">
                    <li><strong>띄어쓰기 차이와 철자 오류 등이 WER에 큰 영향을 주는 반면, CER은 안정적으로 평가됩니다.</strong></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab1:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )

        combined_transcription_data = sample_data.reset_index() # 인덱스 추가
        grid_options = GridOptionsBuilder.from_dataframe(combined_transcription_data)

        grid_options.configure_default_column(
            minWidth=250,  # 최소 너비
            maxWidth=350,  # 최대 너비
            wrapText=False,  # 텍스트 줄바꿈 비활성화
            resizable=True,   # 컬럼 크기 조정 가능
            cellStyle={"fontSize": "12px"}
        )

        grid_options.configure_column(
            "index",  # 인덱스 컬럼 셀렉
            minWidth=70,  # 별도 최소 너비 설정
            maxWidth=70,  # 별도 최대 너비 설정
            cellStyle={"textAlign": "left"}, # 셀 좌정렬
            headerClass={"textAlign": "left"},  # 헤더 우정렬
        )

        grid_options.configure_column(
            "WER",  # 인덱스 컬럼 셀렉
            minWidth=70,  # 별도 최소 너비 설정
            maxWidth=70,  # 별도 최대 너비 설정
            cellStyle={"textAlign": "left"}, # 셀 좌정렬
            headerClass={"textAlign": "left"},  # 헤더 우정렬
        )

        grid_options.configure_column(
            "CER",  # 인덱스 컬럼 셀렉
            minWidth=70,  # 별도 최소 너비 설정
            maxWidth=70,  # 별도 최대 너비 설정
            cellStyle={"textAlign": "left"}, # 셀 좌정렬
            headerClass={"textAlign": "left"},  # 헤더 우정렬
        )

        grid_response = AgGrid(
            combined_transcription_data,
            gridOptions=grid_options.build(),
            height=160
        )

        tab1_info()

elif dropdown_option == "평가":

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["베스트 모델 선정","구두점 포함된 평균 CER", "구두점 제거된 평균 CER", "CER 개선도 비교 (구두점 포함 vs 제거)", "실제 전사 결과"])

    def tab1_and_tab2_info():
        st.markdown(
            """
            <div style="background-color:rgba(200, 200, 240, 0.9); padding:15px; border-radius:10px; 
                        margin-top:20px; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); border: 1px solid #999;">
                <h5 style="font-family:Arial; font-size:14px; color:#222; margin-top:15px;">CER 차트 설명</h5>
                <ul style="font-size:13px; line-height:1.6; color:#333; margin-left:20px;">
                    <li>
                        <strong>Average CER</strong> : 
                        구두점을 포함한 문자 오류율.
                    </li>
                    <li>
                        <strong>Average CER Without Punct And Space</strong> : 
                        구두점과 띄어쓰기를 제외한 문자 오류율.
                    </li>
                    <li>
                        <strong>Model with lowest CER</strong> : 
                        구두점과 띄어쓰기가 제외된 Whisper_turbo 모델은 <strong>1%</strong> 오류율을 보여줍니다.
                    </li>
                    <li>
                        <strong>Tool-Tip</strong> : 
                        마우스를 차트 위로 가져가 보세요. 추가적인 데이터를 확인할 수 있습니다.
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    def tab3_info():
        if "page" not in st.session_state:
            st.session_state.page = 1

        # 페이지 내용 함수
        def render_page(page):
            if page == 1:
                st.markdown(
                    """
                    <div style="background-color:rgba(200, 200, 240, 0.9); padding:15px; border-radius:10px; 
                                margin-top:20px; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); border: 1px solid #999;">
                        <h5 style="font-family:Arial; font-size:14px; color:#222; margin-top:15px;">CER 차트 설명</h5>
                        <ul style="font-size:13px; line-height:1.6; color:#333; margin-left:20px;">
                            <li>
                                <strong>Average CER Improvement</strong> : 
                                구두점 및 띄어쓰기 제거 전/후의 평균 문자 오류율 간 차이를 나타냅니다.   
                            </li>
                            <li>
                                라벨과와 음성인식 전사 결과 모두에서 구두점과 띄어쓰기 제거를 적용했습니다.  
                            </li>
                            <li>
                                약 <strong>2% ~ 5%</strong> 정도의 오류율 개선이 관찰되었습니다.
                            </li>
                            <li><strong>Tool-Tip</strong> :  
                                차트에 마우스를 올리면 각 데이터 포인트에 대한 추가 정보를 확인할 수 있습니다. 
                            </li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif page == 2:
                st.markdown(
                    """
                    <div style="background-color:rgba(200, 200, 240, 0.9); padding:15px; border-radius:10px; 
                                margin-top:20px; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); border: 1px solid #999;">
                        <h5 style="font-family:Arial; font-size:14px; color:#222; margin-top:15px;">구두점과 띄어쓰기 제거의 장점</h5>
                        <ul style="font-size:13px; line-height:1.6; color:#333; margin-left:20px;">
                            <li>
                                음성인식 모델의 구두점과 띄어쓰기 처리에는 한계가 있습니다.
                            </li>
                            <li>
                                구두점과 띄어쓰기를 부수적인 요소로 판단 할 필요가 있습니다.
                            </li>
                            <li>
                                불필요한 오류로 인한 CER 평가 감소를 방지합니다.
                            </li>
                            <li>
                                모델의 본질적인 텍스트 인식 능력만을 평가할 수 있습니다.
                            </li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        def update_page():
            st.session_state.page = 1 if st.session_state.page == 2 else 2

        render_page(st.session_state.page)
        
        st.markdown("<div style='margin-top: 25x;'></div>", unsafe_allow_html=True)

        st.button("Next Page", key="next", help="다음 페이지로 이동", on_click=update_page)

    with tab0:
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
                        <strong style="font-size:22px; color:#1d3557;">Whisper-turbo</strong><br>
                        <strong style="font-size:22px; color:#1d3557;">Whisper-large-v3-turbo</strong><br>
                        <span style="font-size:18px; color:#7a7a7a;">구두점 포함된 평균 CER :</span>
                        <strong style="font-size:20px; color:#388e3c;">3%</strong><br>
                        <span style="font-size:18px; color:#7a7a7a;">구두점과 띄어쓰기가 제거된 평균 CER :</span>
                        <strong style="font-size:20px; color:#388e3c;">1%</strong>
                    </p>
                    <p style="font-size:16px; color:#5c6bc0; text-align:center;">
                        Whisper-turbo 모델은 Whisper-large-v2의 경량화 버전이며,<br>
                        Whisper-large-v3-turbo는 가장 최신 경량 모델입니다.
                    </p>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with tab1:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )
        chart_average_cer = alt.Chart(sorted_data).mark_bar(opacity=0.7, cornerRadiusEnd=5).encode(
            x=alt.X(
                "Average_CER:Q", 
                title="Average CER (%)",
                axis=alt.Axis(
                    titlePadding=25,
                    format=".1%",
                    titleFontSize=14,
                    labelFontSize=12
                )
            ),
            y=alt.Y(
                "Model:N", 
                sort=model_order,
                title=None,
                axis=alt.Axis(
                    labelFontSize=12,
                )
            ),
            color=alt.Color(
                "Company:N",
                scale=alt.Scale(scheme="category10"), 
                legend=alt.Legend(
                    title="기업명",
                    titleFontSize=14,
                    labelFontSize=12,
                    orient="top",
                    offset=10,
                    titlePadding=15,
                    labelPadding=25 
                )
            ),
            tooltip=["Company", "Model", "Average_CER"]
        ).properties(width=500, height=85)

        facet_chart = chart_average_cer.facet(
            row=alt.Row(
                "Company:N", 
                header=alt.Header(
                    title="모델명", 
                    titleFontSize=14,
                    titleAngle=0,
                    labels=False,
                    titlePadding=-30
                )
            ),
            spacing=10  
        ).resolve_scale(
            x='shared', 
            y='independent' 
        )

        st.altair_chart(facet_chart, use_container_width=True)
        tab1_and_tab2_info()

    with tab2:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )

        # 정렬 순서 유지
        model_order = sorted_data["Model"].tolist()
        tab2_data = sorted_data.copy()
        tab2_data["Sticker"] = tab2_data["Model"].map({"whisper_turbo": "★ Model with lowest CER"}).fillna("")
        tab2_data = tab2_data.set_index("Model").loc[model_order].reset_index()

        # 기본 차트
        chart_average_cer_without_punct = alt.Chart(tab2_data).mark_bar(opacity=0.7, cornerRadiusEnd=5).encode(
            x=alt.X(
                "Average_CER_Without_Punct_And_Space:Q", 
                title="Average CER Without Punct And Space (%)",
                axis=alt.Axis(
                    titlePadding=25,
                    format=".1%",
                    titleFontSize=14,
                    labelFontSize=12
                )
            ),
            y=alt.Y(
                "Model:N", 
                sort=model_order,
                title=None,
                axis=alt.Axis(
                    labelFontSize=12,
                )
            ),
            color=alt.Color(
                "Company:N",
                scale=alt.Scale(scheme="category10"), 
                legend=alt.Legend(
                    title="기업명",
                    titleFontSize=14,
                    labelFontSize=12,
                    orient="top",
                    offset=10,
                    titlePadding=15,
                    labelPadding=25 
                )
            ),
            tooltip=["Company", "Model", "Average_CER_Without_Punct_And_Space"]
        ).properties(width=500, height=85)

        sticker_layer = alt.Chart(tab2_data).mark_text(
            fontSize=12, 
            color="white",
            dx=25,
            align="left",
            baseline="middle"
        ).encode(
            y=alt.Y("Model:N", sort=model_order, title=None),
            x=alt.value(0),
            text=alt.Text("Sticker:N")
        )

        combined_chart = alt.layer(chart_average_cer_without_punct, sticker_layer)

        facet_chart = combined_chart.facet(
            row=alt.Row(
                "Company:N",
                header=alt.Header(
                    title="모델명", 
                    titleFontSize=14,
                    titleAngle=0,
                    labels=False,
                    titlePadding=-30
                )
            ),
            spacing=10  
        ).resolve_scale(
            x='shared', 
            y='independent' 
        )

        st.altair_chart(facet_chart, use_container_width=True)
        tab1_and_tab2_info()

    with tab3:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )
        
        chart_average_cer_improvement = alt.Chart(sorted_data).mark_circle(
            size=200  # 모든 점의 크기를 100으로 설정
        ).encode(
            x=alt.X(
                "Average_CER_Improvement:Q",
                title="Average CER Improvement (%)",
                axis=alt.Axis(
                    titleFontSize=14,
                    titlePadding=25,
                    labelFontSize=12,
                    tickCount=9,  # 눈금을 4개로 설정
                    format=".1%",  # 퍼센트 포맷
                    grid=True  # x축 그리드 추가
                ),
                scale=alt.Scale(domain=[0.01, 0.08])  # x축 범위 설정
            ),
            y=alt.Y(
                "Model:N", 
                title="모델명", 
                sort=model_order,
                axis=alt.Axis(
                    title=None,
                    labelFontSize=12, 
                    grid=True  # y축 그리드 추가
                )
            ),
            color=alt.Color(
                "Company:N",
                scale=alt.Scale(scheme="category10"), 
                legend=alt.Legend(
                    title="기업명",
                    titleFontSize=14,
                    labelFontSize=12,
                    orient="top",
                    offset=10,
                    titlePadding=15,
                    labelPadding=25 
                )
            ),
            tooltip=["Company", "Model", "Average_CER_Improvement"]
        ).properties(width=500, height=85)

        facet_chart = chart_average_cer_improvement.facet(
            row=alt.Row(
                "Company:N", 
                header=alt.Header(
                    title="모델명", 
                    titleFontSize=14,
                    titleAngle=0,
                    labels=False,
                    titlePadding=-30
                )
            ),
            spacing=10  
        ).resolve_scale(
            x='shared', 
            y='independent' 
        )

        st.altair_chart(facet_chart, use_container_width=True)
        tab3_info()

    with tab4:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )

        combined_transcription_data = combined_transcription_data.reset_index() # 인덱스 추가
        grid_options = GridOptionsBuilder.from_dataframe(combined_transcription_data)

        grid_options.configure_default_column(
            minWidth=350,  # 최소 너비
            maxWidth=500,  # 최대 너비
            wrapText=False,  # 텍스트 줄바꿈 비활성화
            resizable=True,   # 컬럼 크기 조정 가능
            cellStyle={"fontSize": "12px"}
        )
    
        grid_options.configure_column(
            "index",  # 인덱스 컬럼 셀렉
            minWidth=75,  # 별도 최소 너비 설정
            maxWidth=100,  # 별도 최대 너비 설정
            cellStyle={"textAlign": "left"}, # 셀 좌정렬
            headerClass={"textAlign": "left"},  # 헤더 우정렬
        )

        grid_options.configure_column(
            "cleand_text",  # 강조할 열 이름
            cellStyle={"backgroundColor": "rgba(169, 169, 169, 0.1)"}
        )

        grid_response = AgGrid(
            combined_transcription_data,
            gridOptions=grid_options.build(),
            height=400
        )









        # # 특정 열 강조 함수
        # def highlight_columns(column):
        #     if column.name == "cleand_text": 
        #         return ["background-color: rgba(169, 169, 169, 0.1)"] * len(column)
        #     else:
        #         return [""] * len(column)

        # # 스타일링된 데이터 생성
        # styled_data = combined_transcription_data.style.apply(
        #     highlight_columns, axis=0
        # ).set_table_styles(
        #     [
        #         {'selector': 'th', 'props': [('font-size', '12px'), ('white-space', 'nowrap')]},
        #         {'selector': 'td', 'props': [('font-size', '12px'), ('white-space', 'nowrap')]},
        #     ]
        # )

        # # HTML로 변환
        # html = styled_data.to_html()

        # # 테이블 HTML 출력
        # st.markdown(f"""
        #     <div style="max-width: 800px; max-height: 400px; overflow: auto; border: 1px solid #ddd; border-radius: 5px;">
        #         {html}
        #     </div>
        # """, unsafe_allow_html=True)
    
# elif dropdown_option == "경량화":

    # tab1, tab2 = st.tabs(["경량화 모델 속도 측정", " "])

    # with tab1:

    #     st.markdown(
    #         """
    #         <div style="margin-top: 20px;"></div>
    #         """, 
    #         unsafe_allow_html=True
    #     )


    #     chart = alt.Chart(faster_whisper_data).mark_bar(opacity=0.7, cornerRadiusEnd=5).encode(
    #         x=alt.X('GPU:N', title=None, axis=alt.Axis(labelAngle=0)),
    #         y=alt.Y('sum(Value):Q', title='sec', 
    #                 scale=alt.Scale(type='log', domain=[5, 1000]), 
    #                 axis=alt.Axis(ticks=True, labels=True, grid=False, titleAngle=0, titlePadding=40)),
    #         color='GPU:N',
    #         column='model:N'
    #     ).properties(
    #         width=130,  # 너비 조정
    #         height=250  # 높이 조정
    #     )

    #     st.altair_chart(chart, use_container_width=False) 