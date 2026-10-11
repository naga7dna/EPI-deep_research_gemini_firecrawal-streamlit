import streamlit as st
from typing import Dict, Any, List, Iterator
import time
import ollama
from firecrawl import FirecrawlApp


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ollama Deep Research Agent",
    page_icon="📘",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = ""


# ============================================================
# OLLAMA CLIENT
# ============================================================

@st.cache_resource
def get_ollama_client(host: str):

    return ollama.Client(
        host=host,
        timeout=3600.0
    )


# ============================================================
# AVAILABLE OLLAMA MODELS
# ============================================================

def get_available_models(host: str) -> List[str]:

    try:

        client = get_ollama_client(host)

        response = client.list()

        names = []

        for model in getattr(response, "models", None) or []:

            name = (
                getattr(model, "model", None)
                or getattr(model, "name", None)
            )

            if name:
                names.append(name)

        return names

    except Exception:

        return []


# ============================================================
# OLLAMA GENERATION (STREAMING)
# ============================================================

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def partial_tag_length(text: str, tag: str) -> int:

    # Longest suffix of text that is a prefix of tag.

    for length in range(min(len(text), len(tag) - 1), 0, -1):

        if text.endswith(tag[:length]):
            return length

    return 0


def stream_without_thinking(
    chunks
) -> Iterator[str]:

    # Takes a stream of content strings and strips
    # any <think>...</think> blocks some local
    # models emit inline even when thinking is
    # disabled.

    inside_think = False

    pending = ""

    for piece in chunks:

        if not piece:
            continue

        pending += piece

        emit = ""

        while pending:

            if inside_think:

                end = pending.find(THINK_CLOSE)

                if end == -1:

                    keep = partial_tag_length(
                        pending,
                        THINK_CLOSE
                    )

                    pending = pending[
                        len(pending) - keep:
                    ]

                    break

                pending = pending[
                    end + len(THINK_CLOSE):
                ]

                inside_think = False

            else:

                start = pending.find(THINK_OPEN)

                if start != -1:

                    emit += pending[:start]

                    pending = pending[
                        start + len(THINK_OPEN):
                    ]

                    inside_think = True

                else:

                    keep = partial_tag_length(
                        pending,
                        THINK_OPEN
                    )

                    emit += pending[:len(pending) - keep]

                    pending = pending[len(pending) - keep:]

                    break

        if emit:
            yield emit

    if pending and not inside_think:
        yield pending


def describe_hardware() -> str:

    # Reports whether Ollama runs the model on
    # GPU or CPU, so slow generation on CPU is
    # understandable in the UI.

    try:

        client = get_ollama_client(ollama_host)

        loaded = client.ps()

        models = (
            getattr(loaded, "models", None)
            or []
        )

        if not models:

            return (
                "no model loaded yet "
                "(loads on the first request)"
            )

        on_gpu = any(
            (getattr(model, "size_vram", 0) or 0) > 0
            for model in models
        )

        if on_gpu:
            return "GPU acceleration detected"

        return (
            "CPU only — no GPU detected. "
            "Generation is much slower."
        )

    except Exception:

        return "unknown"


def ollama_generate_stream(
    prompt: str,
    stats: Dict[str, Any]
) -> Iterator[str]:

    # Streams model output while recording live
    # statistics into `stats` for the progress
    # panel.

    client = get_ollama_client(ollama_host)

    stream = client.chat(
        model=ollama_model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True,
        think=False,
        options={
            "num_ctx": num_ctx,
            "num_predict": 2048
        }
    )

    for chunk in stream:

        stats["chunks"] = (
            stats.get("chunks", 0) + 1
        )

        message = chunk.message

        thinking = (
            getattr(message, "thinking", "")
            or ""
        )

        content = message.content or ""

        if thinking:

            stats["thinking_chars"] = (
                stats.get("thinking_chars", 0)
                + len(thinking)
            )

        if content:

            stats["content_chars"] = (
                stats.get("content_chars", 0)
                + len(content)
            )

            yield content

        elif not thinking:

            stats["empty_chunks"] = (
                stats.get("empty_chunks", 0) + 1
            )

        if getattr(chunk, "done", False):

            stats["done_reason"] = (
                getattr(chunk, "done_reason", "")
            )

            stats["prompt_tokens"] = (
                getattr(
                    chunk,
                    "prompt_eval_count",
                    0
                ) or 0
            )

            stats["output_tokens"] = (
                getattr(
                    chunk,
                    "eval_count",
                    0
                ) or 0
            )

            stats["load_seconds"] = (
                getattr(
                    chunk,
                    "load_duration",
                    0
                ) or 0
            ) / 1e9

            stats["prompt_eval_seconds"] = (
                getattr(
                    chunk,
                    "prompt_eval_duration",
                    0
                ) or 0
            ) / 1e9

            stats["eval_seconds"] = (
                getattr(
                    chunk,
                    "eval_duration",
                    0
                ) or 0
            ) / 1e9


def generate_report_stream(
    prompt: str,
    output_slot
) -> str:

    # Runs the model with a live status panel so
    # every stage of local generation is visible
    # in the app.

    start_time = time.time()

    stats: Dict[str, Any] = {}

    status = st.status(
        "Waiting for first token — Ollama is loading "
        "the model and processing the whole prompt "
        "before any text appears. On CPU this runs "
        "at roughly 5–10 prompt tokens/s, so a "
        "3,000-token prompt takes several minutes. "
        "Leave this page open; progress appears "
        "below as soon as generation starts.",
        expanded=True
    )

    with status:

        st.markdown(
            f"**Host:** {ollama_host}  \n"
            f"**Model:** `{ollama_model}`  \n"
            f"**Hardware:** {describe_hardware()}  \n"
            f"**Context window:** {num_ctx:,} tokens  \n"
            f"**Prompt size:** {len(prompt):,} characters "
            f"(~{len(prompt) // 4:,} tokens)  \n"
            f"**Output limit:** 2048 tokens"
        )

        stats_box = st.empty()

        try:

            parts: List[str] = []

            last_render = 0.0

            for piece in stream_without_thinking(
                ollama_generate_stream(
                    prompt,
                    stats
                )
            ):

                parts.append(piece)

                now = time.time()

                if now - last_render >= 0.5:

                    output_slot.markdown(
                        "".join(parts) + " ▌"
                    )

                    stats_box.markdown(
                        f"⏱️ **{now - start_time:,.0f}s** "
                        f"elapsed · "
                        f"{stats.get('chunks', 0):,} chunks · "
                        f"{len(''.join(parts)):,} characters"
                    )

                    last_render = now

            final_text = "".join(parts)

            output_slot.markdown(
                final_text
                or "*The model returned no visible text.*"
            )

            if not final_text:

                raise ValueError(
                    "The model returned no visible text "
                    f"after {stats.get('chunks', 0)} chunks "
                    f"({stats.get('thinking_chars', 0)} thinking "
                    "characters were stripped). "
                    "Try another model from the sidebar."
                )

            elapsed = time.time() - start_time

            summary = (
                f"✅ Done in {elapsed:,.0f}s · "
                f"{stats.get('output_tokens', 0):,} output "
                f"tokens · "
                f"{stats.get('prompt_tokens', 0):,} prompt "
                f"tokens"
            )

            if stats.get("eval_seconds"):

                tok_s = (
                    stats["output_tokens"]
                    / stats["eval_seconds"]
                )

                summary += (
                    f" · {tok_s:.1f} tokens/s generation"
                )

            if stats.get("prompt_eval_seconds"):

                summary += (
                    f" · prompt processing took "
                    f"{stats['prompt_eval_seconds']:,.0f}s"
                )

            if stats.get("done_reason") == "length":

                summary += (
                    " · ⚠️ stopped at the output "
                    "token limit"
                )

            stats_box.markdown(summary)

            status.update(
                label=(
                    f"Generated "
                    f"{stats.get('output_tokens', 0):,} tokens "
                    f"in {elapsed:,.0f}s with "
                    f"{ollama_model}"
                ),
                state="complete",
                expanded=False
            )

            return final_text

        except Exception:

            elapsed = time.time() - start_time

            stats_box.markdown(
                f"❌ Failed after "
                f"{elapsed:,.0f}s · "
                f"{stats.get('chunks', 0):,} chunks "
                f"received · "
                f"{stats.get('content_chars', 0):,} "
                f"characters generated · "
                f"{stats.get('thinking_chars', 0):,} "
                f"thinking characters stripped"
            )

            status.update(
                label="Generation failed",
                state="error"
            )

            raise


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Configuration")

    st.caption(
        "Research analysis runs on a local model "
        "served by Ollama."
    )

    ollama_host = st.text_input(
        "Ollama Host",
        value="http://localhost:11434",
        help="Address of your Ollama server"
    )

    available_models = get_available_models(
        ollama_host
    )

    if available_models:

        default_model = (
            "qwen3:latest"
            if "qwen3:latest" in available_models
            else available_models[0]
        )

        ollama_model = st.selectbox(
            "Ollama Model",
            options=available_models,
            index=available_models.index(
                default_model
            ),
            help=(
                "Local model used for analysis "
                "and report writing"
            )
        )

    else:

        st.warning(
            "Could not reach Ollama. "
            "Is the server running?"
        )

        ollama_model = "qwen3:latest"

    num_ctx = st.select_slider(
        "Context Window (tokens)",
        options=[4096, 8192, 16384, 32768],
        value=8192,
        help=(
            "On CPU, smaller windows process the "
            "prompt much faster. 8192 is a good "
            "default."
        )
    )

    st.caption(describe_hardware())

    st.markdown("---")

    run_enhancement = st.checkbox(
        "Run Step 4 enhancement pass",
        value=False,
        help=(
            "Re-writes the whole report with the "
            "model for extra polish. Doubles the "
            "generation time."
        )
    )

    st.markdown("---")

    firecrawl_api_key = st.text_input(
        "Firecrawl API Key",
        value=st.session_state.firecrawl_api_key,
        type="password",
        help="Firecrawl API key"
    )

    if firecrawl_api_key:
        st.session_state.firecrawl_api_key = firecrawl_api_key

    st.markdown("---")

    st.markdown(
        """
        ### API Usage

        **Ollama (local)**
        - Research analysis
        - Evidence synthesis
        - Report generation
        - Report enhancement

        **Firecrawl**
        - Web search
        - Web page extraction
        - Source collection
        """
    )


# ============================================================
# MAIN UI
# ============================================================

st.title("📘 Ollama Deep Research Agent")

st.markdown(
    """
    Perform web research with **Firecrawl** and analyze the
    collected evidence using a **local Ollama model** — no
    model API key needed.
    """
)

st.markdown("---")


research_topic = st.text_input(
    "🔎 Enter your research topic",
    placeholder=(
        "Example: "
        "Latest developments in AI drug discovery"
    )
)


# ============================================================
# FIRECRAWL CLIENT
# ============================================================

def get_firecrawl_client():

    return FirecrawlApp(
        api_key=st.session_state.firecrawl_api_key
    )


# ============================================================
# FIRECRAWL SEARCH
# ============================================================

def firecrawl_search(
    query: str,
    max_results: int = 10
) -> Dict[str, Any]:

    try:

        app = get_firecrawl_client()

        results = app.search(
            query,
            limit=max_results
        )

        return {
            "success": True,
            "results": results
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# NORMALIZE FIRECRAWL SEARCH RESULTS
# ============================================================

def extract_search_results(
    results: Any
) -> List[Dict[str, str]]:

    extracted = []

    if results is None:
        return extracted

    # --------------------------------------------------------
    # Object-style response
    # --------------------------------------------------------

    if hasattr(results, "web"):

        items = results.web

        if items is None:
            return extracted

        for item in items:

            extracted.append(
                {
                    "title": getattr(
                        item,
                        "title",
                        ""
                    ),
                    "url": getattr(
                        item,
                        "url",
                        ""
                    ),
                    "description": getattr(
                        item,
                        "description",
                        ""
                    )
                }
            )

        return extracted

    # --------------------------------------------------------
    # Dictionary response
    # --------------------------------------------------------

    if isinstance(results, dict):

        items = (
            results.get("web")
            or results.get("results")
            or results.get("data")
            or []
        )

        if isinstance(items, dict):

            items = (
                items.get("web")
                or items.get("results")
                or []
            )

        for item in items:

            if not isinstance(item, dict):
                continue

            extracted.append(
                {
                    "title": item.get(
                        "title",
                        ""
                    ),
                    "url": item.get(
                        "url",
                        ""
                    ),
                    "description": item.get(
                        "description",
                        item.get(
                            "snippet",
                            ""
                        )
                    )
                }
            )

    return extracted


# ============================================================
# SCRAPE WEB PAGE
# ============================================================

def scrape_page(
    app: FirecrawlApp,
    url: str
) -> str:

    try:

        result = app.scrape(
            url,
            formats=["markdown"]
        )

        # ----------------------------------------------------
        # Object response
        # ----------------------------------------------------

        if hasattr(result, "markdown"):

            return result.markdown or ""

        # ----------------------------------------------------
        # Dictionary response
        # ----------------------------------------------------

        if isinstance(result, dict):

            markdown = result.get(
                "markdown"
            )

            if markdown:
                return markdown

            data = result.get(
                "data"
            )

            if isinstance(data, dict):

                return data.get(
                    "markdown",
                    ""
                )

        return str(result)

    except Exception as e:

        return (
            f"[Could not scrape this page: {e}]"
        )


# ============================================================
# COLLECT WEB SOURCES
# ============================================================

def collect_web_research(
    query: str,
    max_results: int = 10
) -> Dict[str, Any]:

    search_result = firecrawl_search(
        query=query,
        max_results=max_results
    )

    if not search_result["success"]:

        return search_result

    search_items = extract_search_results(
        search_result["results"]
    )

    if not search_items:

        return {
            "success": False,
            "error": (
                "Firecrawl returned no search results."
            )
        }

    app = get_firecrawl_client()

    collected_sources = []

    progress = st.progress(0)

    total = len(search_items)

    for index, item in enumerate(
        search_items
    ):

        title = item.get(
            "title",
            "Untitled"
        )

        url = item.get(
            "url",
            ""
        )

        description = item.get(
            "description",
            ""
        )

        st.write(
            f"📄 Reading source "
            f"{index + 1}/{total}: "
            f"{title}"
        )

        content = ""

        if url:

            content = scrape_page(
                app,
                url
            )

        collected_sources.append(
            {
                "title": title,
                "url": url,
                "description": description,
                "content": content
            }
        )

        progress.progress(
            int(
                ((index + 1) / total) * 100
            )
        )

    progress.empty()

    return {
        "success": True,
        "sources": collected_sources
    }


# ============================================================
# PREPARE RESEARCH FOR THE LOCAL MODEL
# ============================================================

def prepare_research(
    sources: List[Dict[str, Any]]
) -> str:

    output = []

    for index, source in enumerate(
        sources,
        start=1
    ):

        title = source.get(
            "title",
            "Untitled"
        )

        url = source.get(
            "url",
            ""
        )

        description = source.get(
            "description",
            ""
        )

        content = source.get(
            "content",
            ""
        )

        # Keep each source small enough for the
        # local model's context window — on CPU,
        # every extra token of prompt is slow.

        if len(content) > 2000:

            content = content[:2000]

            content += (
                "\n\n[Content truncated]"
            )

        output.append(
            f"""
============================================================
SOURCE {index}
============================================================

TITLE:
{title}

URL:
{url}

DESCRIPTION:
{description}

CONTENT:
{content}

============================================================
END SOURCE {index}
============================================================
"""
        )

    return "\n".join(output)


# ============================================================
# PROMPT — INITIAL ANALYSIS
# ============================================================

def build_initial_report_prompt(
    topic: str,
    research_data: str
) -> str:

    prompt = f"""
You are an expert scientific and technical
research analyst.

Your task is to create a comprehensive research
report from web sources collected by a separate
web-research system.

============================================================
RESEARCH TOPIC
============================================================

{topic}

============================================================
WEB RESEARCH
============================================================

{research_data}

============================================================
ANALYSIS REQUIREMENTS
============================================================

Analyze the evidence carefully.

Do NOT invent:

- papers
- authors
- datasets
- statistics
- experimental results
- URLs
- organizations
- citations

Use only information supported by the supplied
research.

If information is missing, explicitly state that
the available sources do not provide sufficient
evidence.

Identify disagreements between sources.

Distinguish:

- established facts
- reported findings
- interpretations
- predictions

============================================================
REPORT STRUCTURE
============================================================

# Research Title

## Executive Summary

Provide a concise summary of the most important
findings.

## Background

Explain the topic and its importance.

## Key Findings

List the major findings from the collected
sources.

## Detailed Analysis

Provide a detailed synthesis of the evidence.

## Technical Concepts

Explain important technical concepts required
to understand the topic.

## Evidence From Sources

Connect major claims to the relevant sources.

Use:

[Source 1]

[Source 2]

etc.

## Practical Applications

Explain real-world applications.

## Limitations

Discuss limitations in the available evidence.

## Research Gaps

Identify unanswered questions and missing evidence.

## Future Directions

Discuss reasonable future research directions.

## Conclusion

Provide a concise conclusion.

## Sources

List each source with:

- title
- URL
- relevance

Return the complete report in Markdown.
"""

    return prompt


# ============================================================
# PROMPT — REPORT ENHANCEMENT
# ============================================================

def build_enhancement_prompt(
    topic: str,
    report: str
) -> str:

    prompt = f"""
You are a senior scientific editor.

Improve the following research report.

============================================================
RESEARCH TOPIC
============================================================

{topic}

============================================================
INITIAL REPORT
============================================================

{report}

============================================================
EDITORIAL REQUIREMENTS
============================================================

Improve:

1. Scientific depth
2. Technical accuracy
3. Logical organization
4. Clarity
5. Evidence interpretation
6. Explanation of complex concepts
7. Practical implications
8. Research gaps
9. Limitations
10. Future directions

Preserve factual information.

Do NOT invent:

- papers
- authors
- statistics
- experimental results
- datasets
- URLs
- citations

Do not introduce unsupported claims.

If the initial report contains a claim that is
not supported by the supplied evidence, qualify
or remove the claim.

Keep all valid source URLs.

============================================================
FINAL STRUCTURE
============================================================

# Research Title

## Executive Summary

## Background

## Key Findings

## Detailed Analysis

## Technical Concepts

## Evidence From Sources

## Practical Applications

## Limitations

## Research Gaps

## Future Directions

## Conclusion

## Sources

Return ONLY the final Markdown report.
"""

    return prompt


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(
    sources: List[Dict[str, Any]]
):

    for index, source in enumerate(
        sources,
        start=1
    ):

        title = source.get(
            "title",
            "Untitled"
        )

        url = source.get(
            "url",
            ""
        )

        description = source.get(
            "description",
            ""
        )

        st.markdown(
            f"""
### {index}. {title}

**URL:** {url}

{description}
"""
        )


# ============================================================
# RUN RESEARCH
# ============================================================

def run_research(
    topic: str
):

    # ========================================================
    # STEP 1 — FIRECRAWL
    # ========================================================

    st.header(
        "🔎 Step 1 — Web Research"
    )

    with st.spinner(
        "Searching the web with Firecrawl..."
    ):

            research_result = (
                collect_web_research(
                    query=topic,
                    max_results=6
                )
            )

    if not research_result["success"]:

        st.error(
            "Firecrawl research failed:\n\n"
            + research_result["error"]
        )

        return None

    sources = research_result["sources"]

    st.success(
        f"Collected {len(sources)} sources."
    )

    # ========================================================
    # SHOW SOURCES
    # ========================================================

    with st.expander(
        "🔗 View Collected Sources"
    ):

        display_sources(
            sources
        )

    # ========================================================
    # STEP 2 — PREPARE RESEARCH
    # ========================================================

    st.header(
        "📚 Step 2 — Preparing Evidence"
    )

    research_data = prepare_research(
        sources
    )

    st.success(
        "Web research prepared for the local model "
        f"— {len(research_data):,} characters "
        f"(~{len(research_data) // 4:,} tokens)."
    )

    # ========================================================
    # STEP 3 — LOCAL MODEL ANALYSIS
    # ========================================================

    st.header(
        "🧠 Step 3 — Local Model Analysis"
    )

    prompt = build_initial_report_prompt(
        topic,
        research_data
    )

    with st.expander(
        "📄 Initial Report",
        expanded=True
    ):

        output_slot = st.empty()

        try:

            initial_report = generate_report_stream(
                prompt,
                output_slot
            )

        except Exception as e:

            st.error(
                "Local model analysis failed:\n\n"
                + str(e)
            )

            return None

    # ========================================================
    # STEP 4 — LOCAL MODEL ENHANCEMENT
    # ========================================================

    st.header(
        "✨ Step 4 — Report Enhancement"
    )

    st.markdown("---")

    st.header(
        "📘 Final Research Report"
    )

    if run_enhancement:

        output_slot = st.empty()

        try:

            final_report = generate_report_stream(
                build_enhancement_prompt(
                    topic,
                    initial_report
                ),
                output_slot
            )

        except Exception as e:

            st.error(
                "Report enhancement failed:\n\n"
                + str(e)
            )

            return None

    else:

        final_report = initial_report

        st.markdown(final_report)

        st.caption(
            "Enhancement pass skipped — enable "
            "\"Run Step 4 enhancement pass\" in the "
            "sidebar to polish the report with a "
            "second model pass."
        )

    # ========================================================
    # SAFE FILE NAME
    # ========================================================

    safe_name = (
        topic
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("?", "")
    )

    if not safe_name:

        safe_name = "research"

    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.download_button(
        label="⬇️ Download Research Report",
        data=final_report,
        file_name=(
            f"{safe_name}_report.md"
        ),
        mime="text/markdown"
    )

    # ========================================================
    # SOURCE FILE
    # ========================================================

    source_text = ""

    for index, source in enumerate(
        sources,
        start=1
    ):

        source_text += (
            f"{index}. "
            f"{source.get('title', 'Untitled')}\n"
        )

        source_text += (
            f"URL: "
            f"{source.get('url', '')}\n"
        )

        source_text += (
            f"{source.get('description', '')}\n\n"
        )

    st.download_button(
        label="⬇️ Download Sources",
        data=source_text,
        file_name=(
            f"{safe_name}_sources.txt"
        ),
        mime="text/plain"
    )

    return final_report


# ============================================================
# START BUTTON
# ============================================================

ready = (
    bool(firecrawl_api_key)
    and bool(research_topic.strip())
)


if st.button(
    "🚀 Start Deep Research",
    type="primary",
    disabled=not ready
):

    if not firecrawl_api_key:

        st.warning(
            "Please enter your Firecrawl API key."
        )

    elif not research_topic.strip():

        st.warning(
            "Please enter a research topic."
        )

    else:

        try:

            run_research(
                research_topic.strip()
            )

        except Exception as e:

            st.error(
                "Unexpected error:\n\n"
                + str(e)
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Powered by local Ollama models + Firecrawl"
)