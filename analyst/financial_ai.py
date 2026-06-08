"""AI Financial Analyst."""
from langchain_google_vertexai import ChatVertexAI
from langchain_community.document_loaders import PyPDFLoader
import yfinance as yf
import pandas as pd
from typing import dict as Dict

class AIFinancialAnalyst:
    def __init__(self):
        self.llm = ChatVertexAI(model_name="gemini-1.5-pro-002")

    def get_financials(self, ticker: str) -> Dict:
        stock = yf.Ticker(ticker)
        return {"info": stock.info, "financials": stock.financials.to_dict(),
                "balance_sheet": stock.balance_sheet.to_dict(), "cashflow": stock.cashflow.to_dict()}

    def calculate_ratios(self, financials: Dict) -> Dict:
        info = financials["info"]
        return {"pe_ratio": info.get("trailingPE"), "pb_ratio": info.get("priceToBook"),
                "ev_ebitda": info.get("enterpriseToEbitda"), "roe": info.get("returnOnEquity"),
                "debt_equity": info.get("debtToEquity"), "gross_margin": info.get("grossMargins"),
                "fcf_yield": info.get("freeCashflow", 0) / max(info.get("marketCap", 1), 1)}

    def generate_thesis(self, ticker: str) -> str:
        financials = self.get_financials(ticker)
        ratios = self.calculate_ratios(financials)
        prompt = f"""Analyze {ticker} and generate an investment thesis:
Financials: {financials["info"].get("longName")}
Key Ratios: {ratios}
Provide: 1) Business overview 2) Key strengths 3) Key risks 4) Valuation 5) Buy/Hold/Sell verdict"""
        return self.llm.invoke(prompt).content
