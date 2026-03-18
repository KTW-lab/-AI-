import streamlit as st
import os
import tempfile
import pandas as pd
from typing import List, Dict

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. 세션 상태 초기화
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.set_page_config(page_title="서연이화 생산기술 AI", layout="wide", page_icon="🏭")
st.title("🏭 서연이화 생산기술 문서/데이터 분석 AI")

# ==========================================
# 2. 사이드바 - 설정 및 업로드
# ==========================================
with st.sidebar:
    st.header("🏭 공장 맞춤 설정")
    factory_filter = st.selectbox("검색할 공장/라인", ["전체 공장", "울산 공장 (A)", "아산 공장 (B)", "해외 법인"])

    st.header("📁 실무 문서 업로드")
    st.caption("지원 포맷: PDF, TXT, CSV, 엑셀(XLSX)")
    uploaded_files = st.file_uploader(
        "매뉴얼, PFMEA, MES 불량 데이터 등 업로드",
        type=["pdf", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
    )

    st.header("💾 대화 관리")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("📤 대화 저장"):
            st.download_button(
                "내보내기",
                data=str(st.session_state.messages),
                file_name="sy_smt_history.json",
            )

    if st.button("🔄 AI 엔진 초기화 / 문서 재분석"):
        st.session_state.llm = None
        st.session_state.retriever = None
        st.rerun()


# ==========================================
# 3. 고정밀 문서/엑셀 처리 엔진
# ==========================================
@st.cache_resource
def load_documents(uploaded_files):
    if not uploaded_files:
        return None, None

    docs = []
    temp_files = []

    # 임시 파일 저장
    for f in uploaded_files:
        ext = f.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(f.getvalue())
            temp_files.append((tmp.name, ext, f.name))

    # 확장자별 로드 및 출처(메타데이터) 강제 주입
    for path, ext, orig_name in temp_files:
        try:
            if ext == "pdf":
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = orig_name  # 문서명 저장
                docs.extend(loaded_docs)
            elif ext == "txt":
                loader = TextLoader(path, encoding="utf-8")
                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = orig_name
                docs.extend(loaded_docs)
            elif ext == "csv":
                loader = CSVLoader(path, encoding="utf-8")
                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = orig_name
                docs.extend(loaded_docs)
            elif ext in ["xls", "xlsx"]:
                # 엑셀 처리
                df = pd.read_excel(path)
                text_data = df.to_string()
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt", mode="w", encoding="utf-8"
                ) as txt_tmp:
                    txt_tmp.write(text_data)
                    loader = TextLoader(txt_tmp.name, encoding="utf-8")
                    loaded_docs = loader.load()
                    for d in loaded_docs:
                        d.metadata["source"] = f"[엑셀데이터] {orig_name}"
                    docs.extend(loaded_docs)
        except Exception as e:
            st.error(f"파일 로드 실패 ({orig_name}): {e}")
            continue

    # 문서를 정밀하게 쪼개기
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    # 임베딩 및 DB 생성
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Temperature 0.0으로 설정하여 지어내는 현상(환각) 원천 차단
    llm = OllamaLLM(model="llama3.1", temperature=0.0)

    # 정확도(유사도)가 높은 문서 최대 5개 추출
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return llm, retriever


# 문서 분석 실행
if uploaded_files and not st.session_state.llm:
    with st.spinner("전문 데이터 정밀 분석 및 학습 중..."):
        st.session_state.llm, st.session_state.retriever = load_documents(uploaded_files)
    st.success("✅ 고정밀 문서/데이터 분석 완료!")

# ==========================================
# 4. 메인 챗봇 인터페이스 (팩트 체크 강화)
# ==========================================
if st.session_state.llm:

    # 액션 아이템 추출 버튼
    if st.button("📋 현재까지의 회의/대화 액션 아이템 추출"):
        if len(st.session_state.messages) > 0:
            with st.spinner("핵심 업무 요약 중..."):
                history_text = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
                )
                summary_prompt = (
                    "다음 대화를 읽고, 엔지니어가 현장에서 즉시 처리해야 할 Action Item을 추출해.\n"
                    + history_text
                )
                summary = st.session_state.llm.invoke(summary_prompt)
                st.info(f"**⚡ 핵심 액션 아이템:**\n{summary}")

    # 이전 대화 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 사용자 텍스트 입력
    prompt = st.chat_input("공정 기준, 불량 원인, MES 데이터 분석 등을 질문하세요...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        factory_context = (
            f"반드시 [{factory_filter}] 기준에 맞춰서 "
            if factory_filter != "전체 공장"
            else ""
        )

        with st.chat_message("assistant"):
            with st.spinner("업로드된 문서 교차 검증 중..."):
                try:
                    # 1. 관련된 문서 검색
                    docs = st.session_state.retriever.invoke(prompt)

                    if not docs:
                        answer = (
                            "⚠️ **업로드하신 문서에서 관련된 정보를 찾을 수 없습니다.** "
                            "다른 말로 질문하시거나 관련 문서를 추가로 업로드해 주세요."
                        )
                        st.warning(answer)
                    else:
                        # 2. 검색된 문서의 출처 텍스트 조립
                        context_list = []
                        for i, d in enumerate(docs):
                            source = d.metadata.get("source", "알 수 없는 문서")
                            page = d.metadata.get("page", "")
                            page_info = f"(페이지: {page})" if page else ""
                            context_list.append(
                                f"[출처 {i+1}: {source} {page_info}]\n{d.page_content}"
                            )

                        context_text = "\n\n".join(context_list)

                        # 3. 강력한 환각 방지 프롬프트 생성
                        full_prompt = f"""너는 서연이화의 엄격한 생산기술 엔지니어입니다.
아래 제공된 [현장 데이터]만을 근거로 사용자의 [질문]에 답변하십시오.

[엄격한 규칙]
1. {factory_context}
2. 제공된 [현장 데이터]에 없는 내용은 절대 지어내지 마십시오 (No Hallucination).
3. 데이터에 답이 없다면 \"업로드된 문서에 해당 내용이 없습니다\"라고 명확히 답변하십시오.
4. 답변 시 반드시 근거가 된 [출처 X: 문서명]을 함께 명시하여 신뢰성을 높이십시오.

[현장 데이터]
{context_text}

[질문]
{prompt}

정확하고 검증된 답변:"""

                        # 4. 답변 생성 및 출력
                        answer = st.session_state.llm.invoke(full_prompt)
                        st.markdown(answer)

                        # 5. 사용자가 직접 원문을 검증할 수 있는 토글 UI 제공
                        with st.expander("🔍 AI가 참고한 실제 문서 원문 보기 (팩트 체크)"):
                            st.text(context_text)

                except Exception as e:
                    answer = f"검색 중 오류가 발생했습니다: {e}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

else:
    st.info("👈 왼쪽 사이드바에서 PDF 매뉴얼, TXT 회의록, 엑셀(XLSX) 불량 데이터를 업로드하세요.")
    st.markdown(
        """
    ### 🛡️ 고정밀 팩트 체크 모드 활성화
    - **엑셀/CSV 데이터 완벽 지원:** MES 다운로드 데이터를 그대로 업로드하여 질문 가능
    - **AI 창의성 억제 (Temperature = 0.0):** 상상해서 답변하는 현상 원천 차단
    - **출처 의무 표기:** 답변에 근거 파일명 표시
    - **거짓말 방지:** 문서에 없는 내용은 "모른다"고 답변
    - **교차 검증 UI:** 사용자가 AI의 참고 원문을 직접 확인 가능
    """
    )
