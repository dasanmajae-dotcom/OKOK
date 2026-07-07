import io

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="AI 시황 분석기", page_icon="📈")
st.title("📈 네이버 시가총액 AI 시황 분석기")
st.write("네이버 금융에서 시가총액 상위 종목을 크롤링하고, GPT가 오늘의 시장 흐름을 요약해드립니다.")


def get_naver_market_sum_data():
    """네이버 금융 시가총액 페이지에서 상위 종목 데이터를 크롤링하는 함수"""
    url = "https://finance.naver.com/sise/sise_market_sum.naver"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = "cp949"

        tables = pd.read_html(io.StringIO(response.text))

        df = None
        for table in tables:
            if "종목명" in table.columns:
                df = table
                break

        if df is None:
            return None, "시가총액 테이블을 찾을 수 없습니다."

        df = df.dropna(subset=["종목명"])
        df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

        top_stocks = df.head(15)

        stock_strings = []
        for _, row in top_stocks.iterrows():
            stock_strings.append(
                f"순위: {int(row['N'])} | 종목명: {row['종목명']} | 현재가: {row['현재가']}원 | "
                f"등락률: {row['등락률']} | 거래량: {row['거래량']} | 시가총액: {row['시가총액']}억"
            )

        scraped_text = "=== 국내 시가총액 상위 15개 종목 현황 ===\n" + "\n".join(stock_strings)
        return top_stocks, scraped_text

    except Exception as e:
        return None, f"데이터 크롤링 및 파싱 중 오류 발생: {e}"


def summarize_market_sum(market_data):
    """LangChain과 OpenAI LLM을 이용하여 시가총액 데이터를 분석 및 요약하는 함수"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 20년 경력의 국내 주식 시장 전문 애널리스트입니다. 주어진 시가총액 상위 "
                "종목 리스트를 바탕으로 오늘의 증시 흐름과 특징을 investor가 이해하기 쉽게 "
                "한글로 3~5문단 분량의 리포트 형식으로 작성해주세요.",
            ),
            ("user", "다음 시가총액 데이터를 분석해줘:\n\n{market_data}"),
        ]
    )

    # gpt-5.5는 존재하지 않는 모델명이므로 실제 사용 가능한 모델 'gpt-4o'로 대체
    model = ChatOpenAI(model="gpt-4o")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser
    result = chain.invoke({"market_data": market_data})
    return result


if st.button("시황 분석 시작"):
    with st.spinner("1. 네이버 금융에서 시가총액 상위 종목 데이터를 수집하는 중..."):
        top_stocks, market_text_data = get_naver_market_sum_data()

    if top_stocks is None:
        st.error(market_text_data)
    else:
        st.subheader("수집된 원본 데이터")
        st.dataframe(top_stocks)

        with st.spinner("2. 수집된 데이터를 바탕으로 AI 시황 리포트를 생성하는 중..."):
            ai_summary = summarize_market_sum(market_text_data)

        st.subheader("AI 시황 분석 리포트")
        st.write(ai_summary)
