import streamlit as st
import os
import base64
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

# "데이터셋" 데이터 로드
path = "normalization.csv"
def load_data(path):
    return pd.read_csv(path)
normalization_data = load_data(path)

# "평가지표" 데이터 로드
path = "sample_eval.csv"
def load_data(path):
    return pd.read_csv(path)
sample_data = load_data(path)

# "평가" 데이터 로드
path = "avg_cers.csv"
def load_data(path):
    return pd.read_csv(path)
average_data = load_data(path)

path = "combined_transcription.csv"
def load_data(path):
    return pd.read_csv(path)
combined_transcription_data = load_data(path)

# "평가" 데이터 정렬
def sort_company(company):
    return (company == "한국어", company) 

sorted_data = average_data.sort_values( # sort_values()는 Pandas에서 데이터를 정렬하는 함수
    by=["Company", "Average_CER_Without_Punct_And_Space"], # by : 정렬하고자 하는 "열"
    key=lambda col: col.map(sort_company) if col.name == "Company" else col, # key : 열을 정렬하기 전에 각 열에 대해 적용할 변환 함수를 지정. 조건문은 열 이름(col.name)이 "Company"일 경우에만 특정 함수를 적용하겠다는 의미. "Company" 열에 대해서만 sort_company라는 사용자 정의 함수(map)를 적용하고, else col 통해 다른 열들은 그대로 두겠다는 것. map()은 Pandas 객체에서 각 값을 다른 값으로 변환할 수 있는 메서드. 여기서는 sort_company라는 함수가 “한국어” 값이 우선적으로 정렬.
    ascending=[True, True] # ascending 오름차순 혹은 내림차순, Ture의 경우 오름차순
)

# "최적화" 데이터 로드
path = "faster_whisper_compare.csv"
def load_data(path):
    return pd.read_csv(path)
faster_whisper_data = load_data(path)

path = "segments.csv"
def load_data(path):
    return pd.read_csv(path)
segments_data = load_data(path)

# 여기까지
#######################################################################################################

# 사이드바
st.sidebar.title("목차")
dropdown_option = st.sidebar.selectbox("선택하세요:", ["모델선정", "데이터셋", "평가지표", "평가", "최적화", "기대효과"])

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
    ]

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
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

elif dropdown_option == "데이터셋":
    
    tab1, tab2, tab3, tab4 = st.tabs(["데이터 준비", "데이터 정규화", "데이터 평균값", "분포 시각화"])
    
    normalization_data = normalization_data.rename(columns={"cleand_text_char_count": "char_count", "cleand_text_audio_length": "audio_length"})

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


            # <div style="background-color:rgba(240, 248, 255, 0.8); padding:25px; border-radius:15px; 
            #             margin-top:30px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15); border: 2px solid #a1c4fd;">
            #     <h2 style="font-family: 'Verdana', sans-serif; font-size:24px; color:#1e2a47; text-align:center; 
            #             margin-bottom:20px; font-weight:bold;">
            #         <span style="color:#5c6bc0;">🚀 Faster Whisper</span>
            #     </h2>
            #     <div style="font-family: 'Helvetica Neue', sans-serif; font-size:18px; color:#3a3a3a; line-height:1.8;">
            #         <p style="text-align:center; font-weight:500; margin-bottom:30px;">
            #             Whisper 모델을 기반으로 최적화 엔진 CTranslate2를 사용합니다.<br>
            #             빠른 추론 속도와 경량화로 Openai/Whisper 대비 4배 더 빠른 속도를 자랑합니다.
            #         </p>
            #         <p style="font-size:16px; color:#5c6bc0; text-align:center;">
            #             <strong>Faster Whisper 모델은 약 34분 오디오 데이터를 <span style="font-size:20px; color:#d32f2f;">6초</span> 만에 전사했습니다.</strong>
            #         </p>
            #     </div>
            # </div>

        container_html = f"""
        <div style="background-color:rgba(240, 248, 255, 0.8); padding:25px; border-radius:15px;
                    margin-top:30px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15); border: 2px solid #a1c4fd;">
            <h2 style="font-family: 'Verdana', sans-serif; font-size:24px; color:#1e2a47; text-align:center;
                    margin-bottom:20px; font-weight:bold;">
                <span style="color:#5c6bc0;">📂 데이터셋 출처</span>
            </h2>
            <div style="margin-top: 20px; text-align:center;">
                <div style="
                    display:inline-block; 
                    width:99%; 
                    padding:12px; 
                    background-color:#fff; 
                    border-radius:30px;
                    box-shadow:0 4px 8px rgba(0,0,0,0.1); 
                    border:1px solid #dcdcdc;
                ">
                    <p style="
                        margin:0; 
                        font-family: 'Helvetica Neue', sans-serif;
                        font-size:18px; 
                        color:#3a3a3a; 
                        text-align:center; 
                        line-height:1.8;
                        width:100%;
                    ">
                        AIHUB 한국인 대화음성
                        <a href="https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=130" 
                        target="_blank" 
                        style="text-decoration:none; color:#007bff; font-weight:bold; margin-left:12px;">
                            🔗 데이터셋 확인
                        </a>
                    </p>
                </div>
                <div style="margin-top: 10px;">
                    {audio_html}
                </div>
            </div>
        </div>
        """
        st.markdown(container_html, unsafe_allow_html=True)

    def tab2_info():
        st.markdown(
            """
            <div style="background-color:#FFDAB9; padding:15px; border-radius:10px; border: 1px solid #888;">
                <h5 style="font-family:Arial; font-size:14px; color:#111; margin-top:15px;">데이터 설명</h5>
                <ul style="font-size:13px; line-height:1.4; color:#222; margin-left:15px;">
                    <li><strong>orginal_text</strong> : 이중 전사 라벨로 "(철자전사)/(발음전사)" 구성</li>
                    <li><strong>cleand_text</strong> : 특수 문자 제거 + 철자전사로 정규화</li>
                    <li><strong>difference</strong> : 변경된 "cleand_text"만 표시</li>
                    <li><strong>cleand_text_char_count</strong> : 텍스트 문자 수</li>
                    <li><strong>cleand_text_audio_length</strong> : 오디오 파일 길이</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab1:
        tab1_info()

    with tab2:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )

        normalization_data = normalization_data.drop(columns=["audio_filepath", "remove_punct_and_space", "restore_punct_and_space"])
        normalization_data = normalization_data.reset_index()
        grid_options = GridOptionsBuilder.from_dataframe(normalization_data)

        grid_options.configure_default_column(
            minWidth=350,  # 최소 너비
            maxWidth=500,  # 최대 너비
            wrapText=False,  # 텍스트 줄바꿈 비활성화
            resizable=True,   # 컬럼 크기 조정 가능
            cellStyle={"fontSize": "12px"}
        )

        fixed_columns = ["index", "difference", "char_count", "audio_length"]

        for column in fixed_columns:
            grid_options.configure_column(
                column,
                maxWidth=115,
                cellStyle={"textAlign": "left"},
                headerClass={"textAlign": "left"},
            )

        grid_response = AgGrid(
            normalization_data,
            gridOptions=grid_options.build(),
            height=300
        )

        tab2_info()

    with tab3:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )

        avg_char_count = normalization_data["char_count"].mean()
        avg_audio_length = normalization_data["audio_length"].mean()

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

    with tab4:
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
            field="char_count",
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
            field="audio_length",
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["베스트 모델 선정","구두점 포함된 평균 CER", "구두점 제거된 평균 CER", "개선도 비교(구두점 포함 vs 제거)", "실제 전사 결과"])

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
                        <span style="font-size:18px;">구두점 포함된 평균 CER :</span>
                        <strong style="font-size:20px; color:#388e3c;">3%</strong><br>
                        <span style="font-size:18px;">구두점과 띄어쓰기가 제거된 평균 CER :</span>
                        <strong style="font-size:20px; color:#388e3c;">1%</strong>
                    </p>
                    <p style="font-size:16px; color:#5c6bc0; text-align:center;">
                        <strong>Whisper-turbo 모델은 Whisper-large-v2의 경량 버전이며,<br>
                        Whisper-large-v3-turbo는 가장 최신 경량 모델입니다.</strong>
                    </p>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    def tab2_and_tab3_info():
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

    def tab4_info():
        if "page" not in st.session_state:
            st.session_state.page = 1

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
                                약 <strong>2% ~ 7%</strong> 정도의 오류율 개선이 관찰되었습니다.
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

    with tab1:
        tab1_info()

    model_order = sorted_data["Model"].tolist() # 판다스 객체를 파이썬 리스트 형태로 변환
    combined_transcription_data = combined_transcription_data.loc[:, ["cleand_text"] + model_order] # loc[]는 Pandas에서 특정 행과 열을 선택하는 인덱싱 방식.  

    with tab2:
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
        tab2_and_tab3_info()

    with tab3:
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
        tab2_and_tab3_info()

    with tab4:
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
        tab4_info()

    with tab5:
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
    
elif dropdown_option == "최적화":

    tab1, tab2, tab3, tab4 = st.tabs(["최적화 모델 소개", "테스트 환경", "추론 속도 비교", "실제 전사 결과"])

    # "최적화" 데이터 전처리
    faster_whisper_data = faster_whisper_data.melt(id_vars=['model', 'type', 'batch'], value_vars=['A100', 'T4×2'], var_name='gpu_type', value_name='performance(sec)') # 기본 문법은 다음과 같습니다 : df.melt(id_vars=['고정할 열'], value_vars=['변환할 열들'], var_name='새로운 변수명', value_name='새로운 값명')
    faster_whisper_data['performance(sec)'] = pd.to_numeric(faster_whisper_data['performance(sec)'], errors='coerce')  # 'out of memory' 값 처리
    faster_whisper_data = faster_whisper_data.dropna(subset=['performance(sec)']) # NaN 값 배제
    faster_whisper_data['performance(sec)'] = faster_whisper_data['performance(sec)'].apply(lambda x: round(x)) # 반올림
    faster_whisper_data['gpu_model'] = faster_whisper_data['gpu_type'] + ' - ' + faster_whisper_data['model'] # 결합

    def tab1_info():
        # if "page" not in st.session_state:
        #     st.session_state.page = 1
        # def render_page(page):
            # if page == 1:
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
                        <strong>Faster Whisper 모델은 약 34분 오디오 데이터를 <span style="font-size:20px; color:#d32f2f;">6초</span> 만에 전사했습니다.</strong>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
            # elif page == 2:
            #     st.markdown(
            #         """
            #         <div style="background-color:rgba(240, 248, 255, 0.8); padding:25px; border-radius:15px; 
            #                     margin-top:30px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15); border: 2px solid #a1c4fd;">
            #             <h2 style="font-family: 'Verdana', sans-serif; font-size:24px; color:#1e2a47; text-align:center; 
            #                     margin-bottom:20px; font-weight:bold;">
            #                 <span style="color:#5c6bc0;">🚀 Faster Whisper</span>
            #             </h2>
            #             <div style="font-family: 'Helvetica Neue', sans-serif; font-size:16px; color:#3a3a3a; line-height:1.8;">
            #                 <p style="margin-bottom:15px;">
            #                     <strong>Whisper-large-v3-turbo</strong> 크기에 맞춰 최적화된 프로젝트는 
            #                     <strong>Faster-Whisper</strong>가 대표적입니다.
            #                 </p>
            #                 <ul style="margin-left:20px; margin-bottom:15px;">
            #                     <li>
            #                         <strong>Whisper JAX</strong>: TPU/GPU 환경에서 최적화되었지만, <em>v3-turbo</em>에 지원은 미비합니다.
            #                         <br>
            #                         [<a href="https://github.com/sanchit-gandhi/whisper-jax" target="_blank"
            #                         style="text-decoration:none; color:#007bff;">출처: GitHub</a>]
            #                     </li>
            #                     <li>
            #                         <strong>Insanely Fast Whisper</strong>: GPU 활용을 극대화했으나, <em>v3-turbo</em> 지원 여부는 제한적입니다.
            #                         <br>
            #                         [<a href="https://github.com/instructsub/insanely-fast-whisper" target="_blank"
            #                         style="text-decoration:none; color:#007bff;">출처: GitHub</a>]
            #                     </li>
            #                     <li>
            #                         <strong>whisper.cpp</strong>: CPU 환경에서 메모리 사용을 줄인 C++ 구현으로, 모델 자체를 경량화하지만, 
            #                         <em>v3-turbo</em> 변형은 공식적으로 제공되지 않습니다.
            #                         <br>
            #                         [<a href="https://github.com/ggerganov/whisper.cpp" target="_blank"
            #                         style="text-decoration:none; color:#007bff;">출처: GitHub</a>]
            #                     </li>
            #                 </ul>
            #                 <p style="margin-bottom:0;">
            #                     <strong>결론</strong><br>
            #                     &bull; 현재까지는 <strong>Faster-Whisper</strong>가 
            #                     <strong>Whisper-large-v3-turbo</strong> 기반 경량화 모델로 가장 활발히 사용되고 있습니다.<br>
            #                     &bull; 다른 프로젝트들은 주로 기본 <strong>Whisper(v1, v2, large 등)</strong> 버전을 최적화하며, 
            #                     <em>v3-turbo</em>를 공식 지원하는 사례는 아직 드뭅니다.
            #                 </p>
            #             </div>
            #         </div>
            #         """,
            #         unsafe_allow_html=True
            #     )
        # def update_page():
        #     st.session_state.page = 1 if st.session_state.page == 2 else 2
        # render_page(st.session_state.page)
        # st.markdown("<div style='margin-top: 25x;'></div>", unsafe_allow_html=True)
        # st.button("Next Page", key="next", help="다음 페이지로 이동", on_click=update_page)

    def tab2_info():
        st.markdown(
            """
            <div style="background-color:#A7C7E7; padding:15px; border-radius:10px; 
                        margin-top:20px; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); border: 1px solid #999;">
                <h5 style="font-family:Arial; font-size:14px; color:#222; margin-top:20px;">
                    ⚙️ Faster Whisper 테스트 환경
                </h5>
                <ul style="font-size:14px; line-height:1.8; color:#333; margin-left:20px; font-weight: bold;">
                    <li>테스트 환경: Google Colab, Kaggle</li>
                    <li>모델 사이즈: large-v3, large-v3-turbo</li>
                    <li>GPU: NVIDIA A100, T4 × 2</li>
                    <li>배치 크기: 8(기본값), 16</li>
                    <li>데이터 타입: FP16, FP32</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab1:
        tab1_info()

    with tab2:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        ) 
    
        tab2_info()
    
    with tab3:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        )     
        chart = alt.Chart(faster_whisper_data).mark_bar(opacity=0.7, cornerRadiusEnd=5).encode(
        x=alt.X('gpu_model:N', 
                title=None,
                sort=alt.EncodingSortField(field='performance(sec)', op='mean', order='ascending'),
                axis=alt.Axis(labelFontSize=12)), 

        y=alt.Y('performance(sec):Q',  
                title=None,
                axis=alt.Axis(labelFontSize=12),
                scale=alt.Scale(type='linear', domain=[0, 300], clamp=True)),

        color=alt.Color('gpu_type:N', scale=alt.Scale(scheme='category10'), legend=alt.Legend(
                    title="GPU",
                    titleFontSize=14,
                    labelFontSize=12,
                    titlePadding=5,
                    offset=15,
                )),

        tooltip=['model', 'type', 'batch', 'gpu_type', 'performance(sec)']
        ).properties(width=250, height=250)

        facet_chart = chart.facet(
            row=alt.Row(
                'type:N',
                header=alt.Header(
                    title=None,
                    labelFontSize=14, 
                    labelAngle=0,
                    labelPadding=-15
                )), 
            column=alt.Column(
                'batch:N',
                header=alt.Header(
                    title=None,
                    labelFontSize=14,
                    labelPadding=15
                )
            ),
            spacing=20  
        )

        st.altair_chart(facet_chart, use_container_width=False)

    with tab4:
        st.markdown(
            """
            <div style="margin-top: 20px;"></div>
            """, 
            unsafe_allow_html=True
        ) 

        st.video("https://www.youtube.com/watch?v=3kzLZjKAvzQ")

        grid_options = GridOptionsBuilder.from_dataframe(segments_data)

        grid_options.configure_default_column(
            wrapText=False,  # 텍스트 줄바꿈 비활성화
            resizable=True,  # 컬럼 크기 조정 가능
            cellStyle={"fontSize": "12px"}
        )

        # 고정된 크기의 컬럼 설정 (start, end)
        fixed_columns = ["start", "end"]
        for column in fixed_columns:
            grid_options.configure_column(
                column,
                minWidth=70, maxWidth=70,  # 고정된 너비 유지
                cellStyle={"textAlign": "left"},
                headerClass={"textAlign": "left"},
            )

        # text 컬럼을 남은 공간 전체 차지하도록 설정
        grid_options.configure_column(
            "text",
            flex=1,  # 남은 공간을 모두 차지
            cellStyle={"textAlign": "left"},
            headerClass={"textAlign": "left"},
        )

        # 전체 테이블 크기 자동 조정
        grid_response = AgGrid(
            segments_data,
            gridOptions=grid_options.build(),
            height=160,  # 전체 테이블 높이
            fit_columns_on_grid_load=True,  # 전체 컬럼 크기 맞추기
        )

elif dropdown_option == "기대효과":
    tab1 = st.tabs(["기대효과"])
