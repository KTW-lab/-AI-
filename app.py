import streamlit as st
import tempfile
import pandas as pd
import sys
import concurrent.futures
import requests

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional dependencies for Office docs (installed as needed)
# - Word: pip install python-docx
# - PowerPoint: pip install python-pptx
try:
    from docx import Document  # python-docx
except Exception:
    Document = None

try:
    from pptx import Presentation  # python-pptx
except Exception:
    Presentation = None

# Optional dependencies for OCR (installed as needed)
# - OCR: pip install pytesseract pillow pymupdf
# - Also install Tesseract OCR binary (Windows): https://github.com/UB-Mannheim/tesseract/wiki
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# =========================
# 0) 설정값 (모델명 등)
# =========================
LLM_MODEL_NAME = "llama3.1"
EMBEDDING_MODEL_NAME = "nomic-embed-text"

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TIMEOUT_SECONDS = 120

# OCR 설정
ENABLE_OCR_FOR_PDF = True
OCR_MAX_PAGES = 5  # 너무 큰 PDF는 OCR 시간이 길어져서 제한
OCR_LANGUAGE = "kor+eng"  # 한글+영문 (Tesseract 설치 필요)


# ==========================================
# 1. 세션 상태 초기화 (중요: 항상 맨 위에서 보장)
# ==========================================
# Streamlit은 사용자가 입력/클릭할 때마다 파일 전체를 위에서 아래로 "재실행"합니다.
# 따라서 session_state 키를 항상 먼저 만들어줘야 에러가 안 납니다.
st.session_state.setdefault("messages", [])
st.session_state.setdefault("llm", None)
st.session_state.setdefault("retriever", None)


def get_llm():
    """항상 안전하게 LLM 객체를 가져오는 헬퍼."""
    return st.session_state.get("llm", None)


def get_retriever():
    """항상 안전하게 Retriever 객체를 가져오는 헬퍼."""
    return st.session_state.get("retriever", None)


def model_exists(models: list[str], base_name: str) -> bool:
    """Ollama 모델 리스트에서 llama3.1 / llama3.1:latest 둘 다 True 처리."""
    return any(m == base_name or m.startswith(base_name + ":") for m in models)


# ==========================================
# 2. 페이지 기본 UI
# ==========================================
st.set_page_config(page_title="🏭 서연이화 AI 시스템", layout="wide", page_icon="🏭")
st.title("🏭 서연이화 AI 시스템")


# ==========================================
# 3. 사이드바 (설정/업로드/디버그)
# ==========================================
with st.sidebar:
    st.header("🏭 공장 맞춤 설정")
    factory_filter = st.selectbox(
        "검색할 공장/라인", ["전체 공장", "울산 공장 (A)", "아산 공장 (B)", "해외 법인"]
    )

    st.header("📁 실무 문서 업로드")
    st.caption("지원 포맷: PDF, TXT, CSV, 엑셀(XLSX/XLS), Word(DOCX), PowerPoint(PPTX)")
    uploaded_files = st.file_uploader(
        "매뉴얼, PFMEA, MES 불량 데이터 등 업로드",
        type=["pdf", "txt", "csv", "xlsx", "xls", "docx", "ppt", "pptx"],
        accept_multiple_files=True,
    )

    st.header("💾 대화 관리")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 대화 초기화"):
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        if st.button("📤 대화 저장"):
            st.download_button(
                "내보내기",
                data=str(st.session_state.get("messages", [])),
                file_name="sy_smt_history.json",
            )

    if st.button("🔄 AI 엔진 초기화 / 문서 재분석"):
        # 문서 인덱스와 LLM을 초기화해서 다시 학습(인덱스 생성)하게 함
        st.session_state["llm"] = None
        st.session_state["retriever"] = None
        st.rerun()

    st.divider()
    st.caption(f"🤖 LLM: {LLM_MODEL_NAME} | Embedding: {EMBEDDING_MODEL_NAME}")

    # Ollama 연결/모델 상태 표시
    st.caption("🧪 Ollama 상태 체크")
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if r.ok:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            st.caption(f"✅ Ollama 연결됨 (모델 {len(models)}개)")
            st.caption(f"- LLM 존재: {model_exists(models, LLM_MODEL_NAME)}")
            st.caption(f"- Embedding 존재: {model_exists(models, EMBEDDING_MODEL_NAME)}")
        else:
            st.caption(f"⚠️ Ollama 응답 오류: HTTP {r.status_code}")
    except Exception as e:
        st.caption(f"❌ Ollama 연결 실패: {e}")
        st.caption("→ 터미널에서 `ollama serve` 실행 + `ollama pull`로 모델 설치 확인")

    with st.expander("⚙️ 실행 환경(디버그)"):
        st.write(
            {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "pandas": getattr(pd, "__version__", "unknown"),
                "ocr": {
                    "fitz_pymupdf": fitz is not None,
                    "pytesseract": pytesseract is not None,
                    "PIL": Image is not None,
                    "enabled": ENABLE_OCR_FOR_PDF,
                    "max_pages": OCR_MAX_PAGES,
                    "lang": OCR_LANGUAGE,
                },
            }
        )


def ocr_pdf_first_pages(pdf_path: str, max_pages: int = OCR_MAX_PAGES) -> str:
    """스캔 PDF처럼 텍스트가 없는 경우를 위해 OCR로 텍스트 추출.

    요구사항:
    - PyMuPDF(fitz)
    - Pillow
    - pytesseract
    - OS에 Tesseract OCR 바이너리 설치
    """

    if fitz is None or pytesseract is None or Image is None:
        raise RuntimeError(
            "OCR 실행에 필요한 패키지가 없습니다. "
            "pip install pytesseract pillow pymupdf 를 설치하고, "
            "Windows라면 Tesseract OCR 프로그램도 설치하세요."
        )

    doc = fitz.open(pdf_path)
    pages = min(len(doc), max_pages)
    out_lines: list[str] = []

    for i in range(pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang=OCR_LANGUAGE)
        text = (text or "").strip()
        if text:
            out_lines.append(f"[OCR_PAGE_{i+1}]\n{text}")

    doc.close()
    return "\n\n".join(out_lines)


# ==========================================
# 4. 문서 로딩/인덱싱 함수
# ==========================================
@st.cache_resource
def load_documents(_uploaded_files):
    """업로드된 파일들에서 텍스트를 뽑아 -> chunk로 나누고 -> 임베딩 -> FAISS 인덱스 생성."""

    if not _uploaded_files:
        return None, None

    docs = []
    temp_files = []

    # 1) Streamlit 업로드 파일을 임시 파일로 저장
    for f in _uploaded_files:
        parts = f.name.rsplit(".", 1)
        ext = parts[-1].lower() if len(parts) == 2 else ""

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(f.getvalue())
            temp_files.append((tmp.name, ext, f.name))

    # 2) 확장자별 텍스트 로딩
    for path, ext, orig_name in temp_files:
        try:
            if ext == "pdf":
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = orig_name
                docs.extend(loaded_docs)

                # PDF에서 텍스트가 거의 없으면 OCR 시도
                if ENABLE_OCR_FOR_PDF:
                    total_chars = sum(len((d.page_content or "").strip()) for d in loaded_docs)
                    if total_chars < 50:
                        try:
                            ocr_text = ocr_pdf_first_pages(path)
                            if ocr_text.strip():
                                # OCR 텍스트를 추가 문서로 주입
                                from langchain_core.documents import Document as LCDocument

                                docs.append(
                                    LCDocument(
                                        page_content=ocr_text,
                                        metadata={
                                            "source": f"[OCR]{orig_name}",
                                            "page": "",
                                        },
                                    )
                                )
                            else:
                                st.warning(
                                    f"PDF 텍스트가 거의 없어 OCR을 시도했지만 추출 실패했습니다: {orig_name}"
                                )
                        except Exception as e:
                            st.warning(
                                f"PDF 텍스트가 거의 없어 OCR을 시도했지만 실패했습니다 ({orig_name}): {e}"
                            )

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
                # 엑셀 로딩: openpyxl 필요
                try:
                    df = pd.read_excel(path)
                except ImportError as ie:
                    st.error(
                        "엑셀(.xlsx/.xls) 파일을 읽으려면 추가 패키지가 필요합니다.\n"
                        "CMD에서 다음을 실행하세요: pip install openpyxl"
                    )
                    st.caption(f"상세 오류: {ie}")
                    continue
                except Exception as e:
                    st.error(f"엑셀 로드 실패 ({orig_name}): {e}")
                    continue

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

            elif ext == "docx":
                if Document is None:
                    st.error(
                        "Word(.docx) 파일을 읽으려면 추가 패키지가 필요합니다.\n"
                        "CMD에서 다음을 실행하세요: pip install python-docx"
                    )
                    continue

                try:
                    doc = Document(path)
                    text_data = "\n".join([p.text for p in doc.paragraphs if p.text])
                except Exception as e:
                    st.error(f"Word 로드 실패 ({orig_name}): {e}")
                    continue

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt", mode="w", encoding="utf-8"
                ) as txt_tmp:
                    txt_tmp.write(text_data)
                    loader = TextLoader(txt_tmp.name, encoding="utf-8")
                    loaded_docs = loader.load()
                    for d in loaded_docs:
                        d.metadata["source"] = f"[Word] {orig_name}"
                    docs.extend(loaded_docs)

            elif ext in ["ppt", "pptx"]:
                if ext == "ppt":
                    st.error(
                        "PowerPoint(.ppt) 형식은 직접 파싱이 어렵습니다.\n"
                        "PowerPoint에서 '다른 이름으로 저장' → .pptx로 저장 후 업로드해 주세요."
                    )
                    continue

                if Presentation is None:
                    st.error(
                        "PowerPoint(.pptx) 파일을 읽으려면 추가 패키지가 필요합니다.\n"
                        "CMD에서 다음을 실행하세요: pip install python-pptx"
                    )
                    continue

                try:
                    pres = Presentation(path)
                    lines = []
                    for slide in pres.slides:
                        for shape in slide.shapes:
                            if hasattr(shape, "text") and shape.text:
                                lines.append(shape.text)
                    text_data = "\n".join(lines)
                except Exception as e:
                    st.error(f"PPTX 로드 실패 ({orig_name}): {e}")
                    continue

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt", mode="w", encoding="utf-8"
                ) as txt_tmp:
                    txt_tmp.write(text_data)
                    loader = TextLoader(txt_tmp.name, encoding="utf-8")
                    loaded_docs = loader.load()
                    for d in loaded_docs:
                        d.metadata["source"] = f"[PPTX] {orig_name}"
                    docs.extend(loaded_docs)

            else:
                st.warning(f"지원하지 않는 확장자입니다: {orig_name}")
                continue

        except Exception as e:
            st.error(f"파일 로드 실패 ({orig_name}): {e}")
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    if len(splits) == 0:
        st.warning(
            "업로드된 파일에서 인덱싱할 텍스트를 추출하지 못했습니다.\n"
            "- 파일이 비어있거나(텍스트 없음)\n"
            "- 모든 파일 로드가 실패했거나\n"
            "- PDF가 스캔본이라 텍스트가 없을 수 있습니다.\n\n"
            "다른 파일로 다시 시도해 주세요."
        )
        return None, None

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(splits, embeddings)

    llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=0.0)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return llm, retriever


# ==========================================
# 5. 업로드 파일이 있으면 인덱싱 실행
# ==========================================
if uploaded_files and get_llm() is None:
    with st.spinner("전문 데이터 정밀 분석 및 학습 중..."):
        llm, retriever = load_documents(uploaded_files)
        st.session_state["llm"] = llm
        st.session_state["retriever"] = retriever

    if get_llm() is not None:
        st.success("✅ 고정밀 문서/데이터 분석 완료!")


# ==========================================
# 6. 메인 챗봇
# ==========================================
if get_llm():

    # 액션 아이템 추출
    if st.button("📋 현재까지의 회의/대화 액션 아이템 추출"):
        if len(st.session_state.get("messages", [])) > 0:
            with st.spinner("핵심 업무 요약 중..."):
                history_text = "\n".join(
                    [
                        f"{m['role']}: {m['content']}"
                        for m in st.session_state.get("messages", [])
                    ]
                )
                summary_prompt = (
                    "다음 대화를 읽고, 엔지니어가 현장에서 즉시 처리해야 할 Action Item을 추출해.\n"
                    + history_text
                )
                llm = get_llm()
                if llm is None:
                    st.error(
                        "LLM이 초기화되지 않았습니다. 왼쪽에서 파일을 다시 업로드하거나 'AI 엔진 초기화' 후 재시도하세요."
                    )
                    st.stop()
                summary = llm.invoke(summary_prompt)
                st.info(f"**⚡ 핵심 액션 아이템:**\n{summary}")

    # 이전 대화 표시
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("공정 기준, 불량 원인, MES 데이터 분석 등을 질문하세요...")

    if prompt:
        st.session_state.setdefault("messages", [])
        st.session_state["messages"].append({"role": "user", "content": prompt})

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
                    retriever = get_retriever()
                    if retriever is None:
                        st.error(
                            "Retriever가 초기화되지 않았습니다. 왼쪽에서 파일을 다시 업로드하거나 'AI 엔진 초기화' 후 재시도하세요."
                        )
                        st.stop()

                    docs = retriever.invoke(prompt)

                    if not docs:
                        answer = (
                            "⚠️ **업로드하신 문서에서 관련된 정보를 찾을 수 없습니다.** "
                            "다른 말로 질문하시거나 관련 문서를 추가로 업로드해 주세요."
                        )
                        st.warning(answer)
                    else:
                        context_list = []
                        for i, d in enumerate(docs):
                            source = d.metadata.get("source", "알 수 없는 문서")
                            page = d.metadata.get("page", "")
                            page_info = f"(페이지: {page})" if page != "" else ""
                            context_list.append(
                                f"[출처 {i+1}: {source} {page_info}]\n{d.page_content}"
                            )
                        context_text = "\n\n".join(context_list)

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

                        def _run_llm():
                            llm = get_llm()
                            if llm is None:
                                raise RuntimeError("LLM is not initialized")
                            return llm.invoke(full_prompt)

                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                            future = ex.submit(_run_llm)
                            try:
                                answer = future.result(timeout=LLM_TIMEOUT_SECONDS)
                            except concurrent.futures.TimeoutError:
                                answer = (
                                    f"⚠️ LLM 응답이 지연되고 있습니다({LLM_TIMEOUT_SECONDS}초 타임아웃).\n\n"
                                    "확인사항:\n"
                                    "1) Ollama가 실행 중인지 (ollama serve)\n"
                                    "2) 모델이 설치되어 있는지 (ollama list / ollama pull)\n"
                                    "3) PC 사양 대비 모델이 무거운지\n"
                                )
                            except Exception as e:
                                answer = f"LLM 호출 중 오류: {e}"

                        st.markdown(answer)

                        with st.expander("🔍 AI가 참고한 실제 문서 원문 보기 (팩트 체크)"):
                            st.text(context_text)

                except Exception as e:
                    answer = f"검색 중 오류가 발생했습니다: {e}"
                    st.error(answer)

        st.session_state.setdefault("messages", [])
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()

else:
    st.info("👈 왼쪽 사이드바에서 PDF/TXT/CSV/XLSX/DOCX/PPTX 파일을 업로드하세요.")
    st.markdown(
        """
### 🛡️ 고정밀 팩트 체크 모드 활성화
- **엑셀/CSV 데이터 지원**
- **AI 창의성 억제 (Temperature = 0.0)**
- **출처 의무 표기**
- **거짓말 방지:** 문서에 없는 내용은 "업로드된 문서에 해당 내용이 없습니다"라고 답변
- **교차 검증 UI:** 사용자가 AI의 참고 원문을 직접 확인 가능

### 📌 스캔 PDF(OCR 필요) 지원
- PDF에서 텍스트가 거의 없으면 자동으로 OCR을 시도합니다.
- OCR을 쓰려면 아래 설치가 필요합니다:
  - pip install pytesseract pillow pymupdf
  - Windows: Tesseract OCR 프로그램 설치 (UB Mannheim 빌드 권장)
"""
    )
