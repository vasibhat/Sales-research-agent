"""Wires all the parts together. The only place that knows which real tools we use."""

from __future__ import annotations

from functools import partial

from .agent import SalesResearchAgent
from .config import Settings
from .llm import LLM, OpenAILLM
from .memory import Memory
from .rag.store import VectorStore
from .tools import Tool, ToolRegistry, make_http_client
from .tools.company import SearchCompanyArgs, search_company
from .tools.internal_docs import SearchDocsArgs, search_internal_docs
from .tools.news import GetNewsArgs, get_news
from .tools.qualify import ScoreLeadArgs, load_icp, score_lead
from .tools.website import FetchWebsiteArgs, fetch_website


def build_tools(settings: Settings, llm: LLM) -> ToolRegistry:
    http = make_http_client()
    web = make_http_client(follow_redirects=False)  # website tool checks each redirect itself
    store = VectorStore.load(settings.index_path)
    icp = load_icp(settings.icp_path)
    seller = settings.seller_name
    return ToolRegistry([
        Tool(
            "search_company",
            "Get a public profile of a company: what it does, industry, size, history. Source: Wikipedia.",
            SearchCompanyArgs,
            partial(search_company, client=http),
        ),
        Tool("get_news", "Get the latest news headlines about a company.", GetNewsArgs, partial(get_news, client=http)),
        Tool(
            "fetch_website",
            "Read the text of a web page, such as the company's home page or 'about us' page.",
            FetchWebsiteArgs,
            partial(fetch_website, client=web, max_chars=settings.max_tool_output_chars),
        ),
        Tool(
            "search_internal_docs",
            f"Search {seller}'s own documents: products, pricing, customer case studies, objection handling.",
            SearchDocsArgs,
            partial(search_internal_docs, store=store, llm=llm),
        ),
        Tool(
            "score_lead",
            "Score how well the company fits our ideal customer (0-100, Hot/Warm/Cold). "
            "Call it once you know the industry and rough size.",
            ScoreLeadArgs,
            partial(score_lead, icp=icp),
        ),
    ])


def build_agent(settings: Settings | None = None, llm: LLM | None = None) -> SalesResearchAgent:
    settings = settings or Settings.from_env()
    llm = llm or OpenAILLM(settings.model, settings.embedding_model)
    return SalesResearchAgent(llm, build_tools(settings, llm), Memory(settings.memory_path), settings)
