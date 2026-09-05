#!/usr/bin/env python3
"""Ollama を利用して小説の続きを執筆するスクリプト。

入力テキストの世界観とプロットを抽出し、指定したシーン数だけ本編を書き足す。

使用例:
  python storywriter.py novel.txt
  python storywriter.py novel.txt -n 6
  python storywriter.py novel.txt --url http://localhost:11434 --model llama3
  python storywriter.py novel.txt --plot-only
  python storywriter.py novel.txt -q
  python storywriter.py novel.txt --no-shift
  python storywriter.py novel.txt --ending
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# 設定（CLI 引数で上書き可能）
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = (
    "hf.co/bartowski/Ateron_Gemma-4-Novelist-Eclipse-31B-GGUF:Q4_K_M"
)
DEFAULT_SCENES = 3
DEFAULT_TIMEOUT = 1800
CHAR_WINDOW = 2000
WORLDVIEW_TARGET_CHARS = 1000
PLOT_SCENE_CHARS = 100
SCENE_TARGET_CHARS = 1000
SCENE_CLIMAX_CHARS = 1500
PAST_PLOT_COUNT = 3
PAST_SCENES_PER_WINDOW = 3
PREVIOUS_CONTEXT_SCENES = 5
FUTURE_PLOT_COUNT = 3
WORLDVIEW_UPDATE_INTERVAL = 5
WORLDVIEW_REVIEW_CHARS = 6000
NEWS_ITEM_LIMIT = 8
NEWS_FETCH_TIMEOUT = 20
NEWS_FEEDS = (
    "https://assets.wor.jp/rss/rdf/maidona/new.rdf",
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "output"

DONE_MARK = "[済]"
PENDING_MARK = "[未]"
SOURCE_MARK = "[既出]"

SYSTEM_WRITER = (
    "あなたは日本語の小説家です。指定された形式を厳守し、"
    "前置き・後書き・メタ解説は一切出力せず、求められた内容だけを書いてください。"
    "思考過程、英語のメモ、<|channel> や <channel|> などのタグは絶対に出力しないでください。"
)

_CHANNEL_START = re.compile(
    r"<\|channel\|?>?(?:thought|analysis|commentary)?",
    re.IGNORECASE,
)
_CHANNEL_END = re.compile(
    r"(?:<channel\|>|<\|channel\|>|<\|end\|>)",
    re.IGNORECASE,
)
_THINK_BLOCK = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
_CHANNEL_BLOCK = re.compile(
    r"<\|channel\|?>?(?:thought|analysis|commentary)?.*?"
    r"(?:<channel\|>|<\|channel\|>|<\|end\|>)",
    re.DOTALL | re.IGNORECASE,
)
_CHANNEL_SPLIT = re.compile(
    r"<\|channel\|?>?(?:thought|analysis|commentary)?|<channel\|>|<\|end\|>",
    re.IGNORECASE,
)
_JAPANESE_CHAR = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_PARTIAL_TAG_SUFFIXES = tuple(
    sorted(
        {
            "<",
            "<|",
            "<|c",
            "<|ch",
            "<|cha",
            "<|chan",
            "<|chann",
            "<|channe",
            "<|channel",
            "<|channel|",
            "<c",
            "<ch",
            "<cha",
            "<chan",
            "<chann",
            "<channe",
            "<channel",
            "<channel|",
        },
        key=len,
        reverse=True,
    )
)


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------
@dataclass
class PlotScene:
    text: str
    done: bool
    from_source: bool = False

    def to_line(self) -> str:
        if self.from_source:
            mark = SOURCE_MARK
        elif self.done:
            mark = DONE_MARK
        else:
            mark = PENDING_MARK
        return f"{mark} {self.text}"


@dataclass
class NewsItem:
    title: str
    summary: str = ""


@dataclass
class RunStats:
    worldview_reused: bool = False
    plot_reused: bool = False
    scenes_written: int = 0
    scene_chars: int = 0
    worldview_updates: int = 0
    plot_extensions: int = 0
    news_enabled: bool = False
    news_count: int = 0
    plot_only: bool = False
    no_shift: bool = False
    ending: bool = False
    started_at: datetime = field(default_factory=datetime.now)
    started_mono: float = field(default_factory=time.monotonic)

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_mono)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def first_chars(text: str, n: int = CHAR_WINDOW) -> str:
    return text[:n] if len(text) > n else text


def last_chars(text: str, n: int = CHAR_WINDOW) -> str:
    return text[-n:] if len(text) > n else text


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
        "\r", "\n"
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def append_scene(path: Path, scene: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    block = scene.strip()
    if existing.strip():
        content = existing.rstrip() + "\n\n" + block + "\n"
    else:
        content = block + "\n"
    path.write_text(content, encoding="utf-8")


def combined_body(base_text: str, honpen_path: Path) -> str:
    honpen = read_text(honpen_path) if honpen_path.exists() else ""
    if honpen.strip():
        return base_text.rstrip() + "\n" + honpen.lstrip("\n")
    return base_text


def split_honpen_scenes(text: str) -> list[str]:
    if not text.strip():
        return []
    return [
        part.strip()
        for part in re.split(r"\n\s*\n", text.strip())
        if part.strip()
    ]


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def looks_japanese(text: str) -> bool:
    return bool(_JAPANESE_CHAR.search(text))


def _partial_tag_len(text: str) -> int:
    for suffix in _PARTIAL_TAG_SUFFIXES:
        if text.endswith(suffix):
            return len(suffix)
    return 0


def push_visible_stream(buf: str, in_thought: bool) -> tuple[str, str, bool]:
    """ストリーム中の channel 思考ブロックを隠す。戻り値は表示分、残りバッファ、思考中フラグ。"""
    visible: list[str] = []
    while buf:
        if in_thought:
            match = _CHANNEL_END.search(buf)
            if not match:
                keep = _partial_tag_len(buf)
                buf = buf[-keep:] if keep else buf[-48:]
                break
            buf = buf[match.end():]
            in_thought = False
            continue
        match = _CHANNEL_START.search(buf)
        if not match:
            keep = _partial_tag_len(buf)
            if keep:
                visible.append(buf[:-keep])
                buf = buf[-keep:]
            else:
                visible.append(buf)
                buf = ""
            break
        if match.start():
            visible.append(buf[:match.start()])
        buf = buf[match.end():]
        in_thought = True
    return "".join(visible), buf, in_thought


def strip_model_thought(text: str) -> str:
    if not text:
        return text
    text = _THINK_BLOCK.sub("", text)
    prev = None
    while prev != text:
        prev = text
        text = _CHANNEL_BLOCK.sub("", text)
    pieces = [
        _CHANNEL_START.sub("", part).strip()
        for part in _CHANNEL_SPLIT.split(text)
    ]
    kept = []
    for piece in pieces:
        piece = _CHANNEL_END.sub("", piece).strip()
        if not piece or piece.lower() == "thought":
            continue
        if looks_japanese(piece):
            kept.append(piece)
    if kept:
        return "\n\n".join(kept).strip()
    text = _CHANNEL_START.sub("", text)
    text = _CHANNEL_END.sub("", text)
    return text.strip()


def clean_llm_text(text: str) -> str:
    text = strip_code_fences(text)
    text = strip_model_thought(text)
    text = re.sub(
        r"^(はい[、。]?|了解しました[。]?|以下が.+です[。:]?|わかりました[。]?)\s*",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text.strip()


ANSI_GRAY = "\033[90m"
ANSI_RESET = "\033[0m"
_quiet = False


def set_quiet(enabled: bool) -> None:
    global _quiet
    _quiet = enabled


def use_log_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.platform.startswith("linux"):
        return False
    return bool(sys.stderr.isatty())


def log(message: str, *, always: bool = False) -> None:
    if _quiet and not always:
        return
    text = message
    if not always and use_log_color():
        text = f"{ANSI_GRAY}{message}{ANSI_RESET}"
    print(text, file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}時間{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def read_stats_field(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    prefix = f"{key}:"
    for line in read_text(path).splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


class PromptLogger:
    """AI に渡したプロンプトを prompt/ へ 1 件 1 ファイルで保存する。"""

    def __init__(self, prompt_dir: Path, model: str) -> None:
        self.prompt_dir = prompt_dir
        self.model = model
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self.index = self._next_index()
        self.saved = 0

    def _next_index(self) -> int:
        numbers = []
        for path in self.prompt_dir.glob("*.txt"):
            match = re.match(r"^(\d+)_", path.name)
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def save(self, kind: str, prompt: str, system: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_kind = (
            re.sub(r"[^\w一-龥ぁ-んァ-ンー]+", "_", kind).strip("_") or "prompt"
        )
        path = self.prompt_dir / f"{self.index:03d}_{stamp}_{safe_kind}.txt"
        self.index += 1
        self.saved += 1
        body = (
            f"種別: {kind}\n"
            f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"モデル: {self.model}\n"
            f"\n"
            f"-------- system --------\n"
            f"{system.strip()}\n"
            f"\n"
            f"-------- prompt --------\n"
            f"{prompt.rstrip()}\n"
        )
        write_text(path, body)
        return path


# ---------------------------------------------------------------------------
# Ollama クライアント
# ---------------------------------------------------------------------------
class OllamaClient:
    def __init__(
        self,
        url: str,
        model: str,
        timeout: int,
        prompt_logger: PromptLogger | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.prompt_logger = prompt_logger

    def generate(
        self,
        prompt: str,
        *,
        system: str = SYSTEM_WRITER,
        temperature: float = 0.8,
        num_predict: int = 2048,
        num_ctx: int = 8192,
        show_stream: bool | None = None,
        label: str = "generate",
    ) -> str:
        if show_stream is None:
            show_stream = not _quiet
        if self.prompt_logger is not None:
            self.prompt_logger.save(label, prompt, system)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.08,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
        request = urllib.request.Request(
            f"{self.url}/api/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        chunks: list[str] = []
        hold = ""
        in_thought = False
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(f"Ollama エラー: {data['error']}")
                    token = data.get("response", "")
                    chunks.append(token)
                    visible, hold, in_thought = push_visible_stream(
                        hold + token, in_thought
                    )
                    if show_stream and visible:
                        print(visible, end="", file=sys.stderr, flush=True)
                    if data.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama に接続できませんでした ({self.url})\n"
                f"Ollama が起動しているか確認してください: ollama serve\n"
                f"詳細: {exc.reason}"
            ) from exc
        if show_stream and not in_thought and hold:
            print(hold, end="", file=sys.stderr, flush=True)
        if show_stream:
            print(file=sys.stderr, flush=True)
        return clean_llm_text("".join(chunks))


# ---------------------------------------------------------------------------
# プロット入出力
# ---------------------------------------------------------------------------
_LINE_PREFIX = re.compile(
    r"^(?:[-*・]|\[済\]|\[未\]|\[既出\])?\s*"
    r"(?:\d+[\.．、:：)]\s*)?"
)


def parse_plot_file(text: str) -> list[PlotScene]:
    scenes: list[PlotScene] = []
    saw_mark = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"[-ー=〜~＊*]+", line):
            continue
        if re.fullmatch(r"(これまでのシーン|これからのシーン|プロット)[:：]?", line):
            continue
        from_source = line.startswith(SOURCE_MARK)
        done_mark = from_source or line.startswith(DONE_MARK)
        pending_mark = line.startswith(PENDING_MARK)
        if done_mark or pending_mark or from_source:
            saw_mark = True
        body = _LINE_PREFIX.sub("", line).strip()
        if not body:
            continue
        scenes.append(
            PlotScene(text=body, done=done_mark, from_source=from_source)
        )
    if scenes and not saw_mark:
        for scene in scenes[:PAST_PLOT_COUNT]:
            scene.done = True
            scene.from_source = True
    elif scenes and not any(scene.from_source for scene in scenes):
        for scene in scenes[:PAST_PLOT_COUNT]:
            if scene.done:
                scene.from_source = True
    return scenes


def dump_plot(scenes: Iterable[PlotScene]) -> str:
    return "\n".join(scene.to_line() for scene in scenes) + "\n"


def extract_scene_lines(text: str, expected: int) -> list[str]:
    text = clean_llm_text(text)
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(
            r"(これまでのシーン|これからのシーン|プロット|次のシーン)[:：]?",
            line,
        ):
            continue
        if re.fullmatch(r"[-ー=〜~＊*#]+", line):
            continue
        body = _LINE_PREFIX.sub("", line).strip()
        body = re.sub(r"^【[^】]*】\s*", "", body).strip()
        body = re.sub(r"\s+", " ", body)
        if not body or not looks_japanese(body) or len(body) < 8:
            continue
        lines.append(body)
        if len(lines) >= expected:
            break
    return lines[:expected]


def plot_lines_usable(scenes: Iterable[PlotScene]) -> bool:
    items = list(scenes)
    if len(items) < 2:
        return False
    for scene in items:
        if (
            "\n" in scene.text
            or "<|channel" in scene.text
            or "<channel|" in scene.text
        ):
            return False
        if not looks_japanese(scene.text):
            return False
    return True


def previous_scenes(
    plot: list[PlotScene],
    target_index: int,
    count: int = PREVIOUS_CONTEXT_SCENES,
) -> list[PlotScene]:
    start = max(0, target_index - count)
    return plot[start:target_index]


def written_scene_count(plot: Iterable[PlotScene]) -> int:
    return sum(1 for scene in plot if scene.done and not scene.from_source)


def windows_from_end(
    text: str, size: int = CHAR_WINDOW
) -> list[tuple[int, int, str]]:
    """末尾から size 字ずつ遡った断片。先頭が最も新しい末尾。"""
    windows: list[tuple[int, int, str]] = []
    end = len(text)
    while end > 0:
        start = max(0, end - size)
        excerpt = text[start:end]
        if excerpt.strip():
            windows.append((start, end, excerpt))
        end = start
    return windows


def scenes_for_window(length: int, *, is_latest: bool) -> int:
    if is_latest:
        return PAST_SCENES_PER_WINDOW
    estimated = max(1, round(length / CHAR_WINDOW * PAST_SCENES_PER_WINDOW))
    return max(1, min(PAST_SCENES_PER_WINDOW, estimated))


def _char_ngrams(text: str, size: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[i:i + size] for i in range(len(compact) - size + 1)}


def plot_similarity(left: str, right: str) -> float:
    grams_left = _char_ngrams(left)
    grams_right = _char_ngrams(right)
    if not grams_left or not grams_right:
        return 0.0
    return len(grams_left & grams_right) / len(grams_left | grams_right)


def plot_coverage(summary: str, source: str) -> float:
    """短いシーン説明が、長い本文の言い換えになっていないかを見る。"""
    grams = _char_ngrams(summary)
    source_grams = _char_ngrams(source)
    if not grams:
        return 0.0
    return len(grams & source_grams) / len(grams)


def is_duplicate_plot(
    candidate: str,
    existing: Iterable[PlotScene | str],
    threshold: float = 0.42,
) -> bool:
    texts = [
        item.text if isinstance(item, PlotScene) else item for item in existing
    ]
    for text in texts:
        if not text.strip():
            continue
        if plot_similarity(candidate, text) >= threshold:
            return True
        long_source = len(text) > len(candidate) * 2
        if long_source and plot_coverage(candidate, text) >= 0.5:
            return True
    return False


def future_plots_stale(
    lines: list[str],
    *,
    past_count: int,
    existing: Iterable[PlotScene],
    tail: str,
) -> bool:
    future = lines[past_count:]
    if not future:
        return False
    compared: list[PlotScene | str] = list(existing)
    compared.extend(lines[:past_count])
    if tail.strip():
        compared.append(tail)
    return any(is_duplicate_plot(item, compared) for item in future)


def strip_repeated_source(scene: str, *contexts: str) -> str:
    """本編出力の先頭から、ベースや直前本文の再利用を取り除く。"""
    text = scene.strip()
    combined = "\n".join(part for part in contexts if part and part.strip())
    if not text or not combined:
        return text

    ctx_compact = re.sub(r"\s+", "", combined)
    paragraphs = [
        part.strip() for part in re.split(r"\n{2,}", text) if part.strip()
    ]
    kept: list[str] = []
    skipping = True
    for paragraph in paragraphs:
        compact = re.sub(r"\s+", "", paragraph)
        if skipping and len(compact) >= 30 and compact in ctx_compact:
            continue
        skipping = False
        kept.append(paragraph)
    if kept:
        text = "\n\n".join(kept)

    max_check = min(len(text), 400)
    overlap = 0
    for length in range(max_check, 24, -1):
        prefix = text[:length]
        if prefix in combined or combined.endswith(prefix):
            overlap = length
            break
    if overlap:
        text = text[overlap:].lstrip("\n　 ")
    return text.strip() or scene.strip()


def _xml_local_name(tag: str) -> str:
    return tag.split("}", 1)[-1].lower()


def _parse_rss_items(raw: bytes) -> list[NewsItem]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    items: list[NewsItem] = []
    for node in root.iter():
        if _xml_local_name(node.tag) not in {"item", "entry"}:
            continue
        title = ""
        summary = ""
        for child in list(node):
            name = _xml_local_name(child.tag)
            text = (child.text or "").strip()
            if not text:
                continue
            if name == "title":
                title = html.unescape(text)
            elif name in {"description", "summary"}:
                cleaned = re.sub(r"<[^>]+>", "", text)
                summary = html.unescape(cleaned).strip()
        if title:
            items.append(
                NewsItem(
                    title=title,
                    summary=first_chars(summary, 120),
                )
            )
    return items


def fetch_recent_news(limit: int = NEWS_ITEM_LIMIT) -> list[NewsItem]:
    """ニュースの公開 RSS から最近の見出しを取得する。"""
    collected: list[NewsItem] = []
    seen: set[str] = set()
    for url in NEWS_FEEDS:
        if len(collected) >= limit:
            break
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "storywriter/1.0"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=NEWS_FETCH_TIMEOUT
            ) as response:
                raw = response.read()
        except urllib.error.URLError as exc:
            log(
                f"ニュース取得に失敗しました ({url}): {exc.reason}",
                always=True,
            )
            continue
        for item in _parse_rss_items(raw):
            if item.title in seen:
                continue
            seen.add(item.title)
            collected.append(item)
            if len(collected) >= limit:
                break
    return collected


def format_news_digest(items: Iterable[NewsItem]) -> str:
    lines: list[str] = []
    for index, item in enumerate(items, 1):
        if item.summary:
            lines.append(f"{index}. {item.title} — {item.summary}")
        else:
            lines.append(f"{index}. {item.title}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# プロンプト
# ---------------------------------------------------------------------------
def prompt_worldview(head: str) -> str:
    return f"""次の小説本文の冒頭から、続きを執筆するために必要な世界観情報を抽出してください。

含める情報:
- 登場人物の名前・性格・口調（分かる範囲で）
- 舞台（時代・場所・雰囲気）
- 物語固有の用語・設定・ルール
- 人間関係、対立、主人公の目的など、続きを書くのに必要な情報

制約:
- 約{WORLDVIEW_TARGET_CHARS}文字程度でまとめる
- 箇条書きを交えて簡潔に書く
- 本文に無い情報は推測しすぎない
- 前置きや「以下が世界観です」などの文言は書かない
- 思考過程や特殊タグは出力しない
- 世界観情報のみを日本語で出力する

--- 本文冒頭 ---
{head}
"""


def prompt_extract_past_scenes(
    excerpt: str,
    worldview: str,
    expected: int,
    later_scenes: list[str],
    window_note: str,
) -> str:
    later_block = ""
    if later_scenes:
        later_text = "\n".join(f"- {scene}" for scene in later_scenes)
        later_block = (
            f"\n# これより後の本文ですでに抽出したシーン（重複禁止）\n{later_text}\n"
        )
    return f"""次の本文断片に書かれている出来事だけを、時系列順に{expected}シーンで要約してください。
まだ起きていない続きは書かないでください。

# 世界観
{worldview.strip()}
{later_block}
# 本文断片（{window_note}）
{excerpt}

出力ルール:
- ちょうど{expected}行だけ出力する
- 1行1シーン、各シーン約{PLOT_SCENE_CHARS}字
- この断片の中で実際に起きている出来事だけを、古い順に書く
- 後続シーンと同じ出来事は書かない
- 見出し・番号・前置きは不要
- 思考過程、英語、特殊タグは出力しない
- 各行の中に改行を入れない
"""


def prompt_next_plot(
    tail: str,
    plot: list[PlotScene],
    worldview: str,
    news_text: str = "",
    expected: int = FUTURE_PLOT_COUNT,
    *,
    no_shift: bool = False,
    ending: bool = False,
) -> str:
    plot_text = dump_plot(plot).strip()
    news_block = ""
    news_rules = ""
    if news_text.strip():
        news_block = f"\n# 最近の時事\n{news_text.strip()}\n"
        news_rules = (
            "- 時事の見出しをそのまま写さず、世界観に合う形で核だけを取り込む\n"
            "- 新しいシーンの少なくとも1つに、時事から着想した要素を入れる\n"
            "- 舞台が現代でない場合は、主題をその世界の事件・噂・制度に翻訳する\n"
            "- 全シーンを報道の再現にしない\n"
        )
    lead = (
        f"既存の世界観・プロット・本文末尾を踏まえ、これから書く次の"
        f"シーンを{expected}つ作成してください。\n"
        "これは長編小説の続きです。"
    )
    if not no_shift:
        lead += (
            "同じ場面を引き伸ばすのではなく、物語全体を前へ進めてください。"
        )
    if ending:
        lead += (
            "物語の締めくくりとして自然に収まる展開にしてください。"
        )
    shift_rules = ""
    if not no_shift:
        shift_rules = (
            "- 本文末尾や既存シーンと同じ出来事・同じ会話・同じ状況の"
            "繰り返しは禁止\n"
            "- 同じ場所や同じやり取りが続きそうなら、時間経過・場所移動・"
            "新たな人物や事件で場面を切り替える\n"
        )
    ending_rules = ""
    if ending:
        ending_rules = (
            "- 長編の締めくくりとしてふさわしい着地へ向かう\n"
            "- 急な打ち切りや未消化の山場を残さず、読後感がよくなるようにする\n"
        )
    return f"""{lead}

# 世界観
{worldview.strip()}

# 既存プロット
{plot_text}

# 本文末尾（すでに書かれている内容）
{tail}
{news_block}
出力ルール:
- {expected}行だけ出力する
- 1行1シーン、各シーン約{PLOT_SCENE_CHARS}字
- 既存プロットの続きとして自然につながる
{shift_rules}{ending_rules}{news_rules}- 見出し・番号・前置きは不要
- 思考過程、英語、特殊タグは出力しない
- 各行の中に改行を入れない
"""


def prompt_update_worldview(worldview: str, honpen_excerpt: str) -> str:
    return f"""現在の世界観メモと、最近書いた本編を読み比べてください。
本編に、続きを執筆するために必要な新しい情報が増えていないか確認し、世界観へ反映してください。

# 現在の世界観
{worldview.strip()}

# 本編（最近のシーン）
{honpen_excerpt.strip()}

反映してよい情報:
- 新しく登場した人物の名前・性格・口調
- 新しい場所、用語、設定、ルール
- 以降の執筆に必要な人間関係や立場の変化

やってはいけないこと:
- 世界観をゼロから書き直さない。既存の正しい情報は残す
- あらすじや各シーンの要約は書かない
- 一度きりの出来事、一時的な心情、今回限りの演出は載せない
- 本編に無い情報を推測して足さない

分量と出力:
- 出力は更新後の世界観メモ全体のみ
- 追記があっても、全体で約{WORLDVIEW_TARGET_CHARS}文字を厳守する
- 文字数が増えそうなら、重要度の低い既存記述を短くして枠を空ける
- 追記が不要なら、現在の世界観を約{WORLDVIEW_TARGET_CHARS}文字のまま出力する
- 「更新なし」などのメタ発言、前置き、後書きは書かない
- 思考過程や特殊タグは出力しない
"""


def prompt_normalize_plot(
    raw: str,
    expected: int,
    tail: str,
    worldview: str,
    past_count: int = 0,
    *,
    no_shift: bool = False,
    ending: bool = False,
) -> str:
    if past_count >= expected:
        split_rule = (
            f"- {expected}行すべてが、この本文断片にすでに書かれている出来事の要約\n"
            f"- まだ起きていない続きは書かない"
        )
    elif past_count > 0:
        split_rule = (
            f"- 最初の{past_count}行は、本文末尾にすでに書かれている出来事の要約\n"
            f"- 残りの{expected - past_count}行は、本文にも既存プロットの既出部分にも無い、新しい続き"
        )
    else:
        split_rule = (
            f"- {expected}行すべてが、本文末尾や既存シーンにまだ書かれていない新しい展開"
        )
    extra = ""
    if past_count < expected and not no_shift:
        extra += (
            "- 新しいシーンは本文末尾の言い換えにしない。"
            "必要なら場所・時間・相手を切り替えて物語を進める\n"
        )
    if past_count < expected and ending:
        extra += (
            "- 新しいシーンは物語の締めくくりとしてふさわしい着地へ向かう\n"
        )
    return f"""次のテキストは小説のシーン説明です。世界観と本文末尾を参照し、指定形式へ加工してください。

形式:
- ちょうど{expected}行だけ出力する
- 1行に1シーンだけ書く
- 各行の中に改行を入れない
- 各行は日本語のみで、約{PLOT_SCENE_CHARS}字のシーン説明にする
- 番号、箇条書き記号、見出し、英語、思考過程、特殊タグは付けない
- 前置きと後書きは禁止
{split_rule}
{extra}
# 世界観
{worldview.strip()}

# 本文末尾（既に書かれている）
{tail}

# 加工するテキスト
{raw}
"""


def prompt_write_scene(
    worldview: str,
    prev: list[PlotScene],
    target: PlotScene,
    tail: str,
    *,
    no_shift: bool = False,
    ending: bool = False,
) -> str:
    if prev:
        prev_text = "\n".join(f"- {scene.text}" for scene in prev)
    else:
        prev_text = "- （冒頭のため直前シーンなし）"
    extra_rules = ""
    if not no_shift:
        extra_rules += (
            "- 同じ会話、同じ場所、同じ心情描写の繰り返しになったら、"
            "時間経過や場所移動で場面を切り替えて先へ進める\n"
            "- 長編の一場面として、そのシーンで状況が少しでも"
            "前に進むように書く\n"
        )
    if ending:
        extra_rules += (
            "- このシーンは小説の締めくくりとしてふさわしい終わり方にする\n"
            "- 末尾は余韻が残るようにきれいに閉じ、読後感を良くする\n"
            "- 続きを急かす中途半端な切れ方や、説明で畳む終わりは避ける\n"
        )
    return f"""以下の情報を踏まえ、指定されたシーンの「新しい本文」だけを執筆してください。

# 世界観
{worldview.strip()}

# これまでのシーン
{prev_text}

# 今回執筆するシーン
- {target.text}

# 直前の本文（参照用。出力に含めない）
{tail}

執筆ルール:
- 出力は新しいシーンの本文のみ。直前の本文やベース本文をコピー・再掲・要約し直して書き始めない
- 直前の最後の文の「次の瞬間」から書き始める
- 今回のシーンで起きるべき出来事を、小説の地の文と会話で描く
- 分量は約{SCENE_TARGET_CHARS}字を基本とする
- 対立、告白、暴力、重要な選択など物語が盛り上がる場面では、必要に応じて約{SCENE_CLIMAX_CHARS}字まで伸ばしてよい
{extra_rules}- 日本語として破綻なく、そのシーンとして完結する形で終わる（文の途中で切らない）
- タイトル、シーン番号、解説、メタ情報、思考過程、特殊タグは出力しない
- 本文のみを日本語で出力する
"""


# ---------------------------------------------------------------------------
# 執筆ワークフロー
# ---------------------------------------------------------------------------
class StoryWriter:
    def __init__(
        self,
        client: OllamaClient,
        work_dir: Path,
        stats: RunStats | None = None,
    ) -> None:
        self.client = client
        self.work_dir = work_dir
        self.stats = stats if stats is not None else RunStats()
        self.news_text = ""
        self.base_path = work_dir / "ベース.txt"
        self.worldview_path = work_dir / "世界観.txt"
        self.plot_path = work_dir / "プロット.txt"
        self.honpen_path = work_dir / "本編.txt"
        self.stats_path = work_dir / "統計.txt"
        self.prompt_dir = work_dir / "prompt"

    def sanitize_outputs(self) -> None:
        for path in (self.worldview_path, self.honpen_path):
            if not path.exists():
                continue
            original = read_text(path)
            cleaned = clean_llm_text(original)
            if cleaned and cleaned != original.strip():
                write_text(path, cleaned)
                log(f"{path.name} から思考タグを除去しました。")

    def ensure_worldview(self, source: str, force: bool) -> str:
        if self.worldview_path.exists() and not force:
            text = clean_llm_text(read_text(self.worldview_path))
            if looks_japanese(text):
                if text != read_text(self.worldview_path).strip():
                    write_text(self.worldview_path, text)
                log("既存の 世界観.txt を再利用します。")
                self.stats.worldview_reused = True
                return text
            log("世界観.txt が不正なため再生成します。")

        head = first_chars(source)
        log(f"世界観を抽出しています（冒頭 {len(head)} 字）...")
        raw = self.client.generate(
            prompt_worldview(head),
            temperature=0.4,
            num_predict=2048,
            label="世界観抽出",
        )
        worldview = clean_llm_text(raw)
        if not worldview:
            raise RuntimeError("世界観の抽出に失敗しました（空の応答）。")
        write_text(self.worldview_path, worldview)
        log(f"世界観.txt を保存しました（{len(worldview)} 字）。")
        return worldview

    def ensure_plot(
        self, source: str, worldview: str, force: bool
    ) -> list[PlotScene]:
        if self.plot_path.exists() and not force:
            scenes = parse_plot_file(read_text(self.plot_path))
            if plot_lines_usable(scenes):
                log("既存の プロット.txt を再利用します。")
                self.stats.plot_reused = True
                return scenes
            log(
                "プロット.txt の形式が不正なため、"
                "本文末尾から作り直します（本編は保持します）。"
            )

        body = combined_body(source, self.honpen_path)
        past = self._extract_source_plot(body, worldview)
        tail = last_chars(body)
        log("これからのシーンを作成しています...")
        future_lines = self._generate_plot_lines(
            prompt_next_plot(
                tail,
                past,
                worldview,
                self.news_text,
                no_shift=self.stats.no_shift,
                ending=self.stats.ending,
            ),
            expected=FUTURE_PLOT_COUNT,
            tail=tail,
            worldview=worldview,
            create_label="プロット作成",
            past_count=0,
            existing=past,
        )
        plot = past + [
            PlotScene(text=line, done=False) for line in future_lines
        ]
        write_text(self.plot_path, dump_plot(plot))
        log(
            f"プロット.txt を保存しました"
            f"（既出 {sum(1 for scene in past if scene.from_source)} シーン"
            f" + これから {FUTURE_PLOT_COUNT} シーン）。"
        )
        return plot

    def _extract_source_plot(
        self, source: str, worldview: str
    ) -> list[PlotScene]:
        windows = windows_from_end(source)
        if not windows:
            return []
        groups: list[list[str]] = []
        later: list[str] = []
        for index, (start, end, excerpt) in enumerate(windows):
            is_latest = index == 0
            expected = scenes_for_window(len(excerpt), is_latest=is_latest)
            note = f"全体 {len(source)} 字のうち {start + 1}〜{end} 字目"
            log(f"既出シーンを抽出しています（{note} / {expected} シーン）...")
            lines = self._generate_plot_lines(
                prompt_extract_past_scenes(
                    excerpt,
                    worldview,
                    expected,
                    later,
                    note,
                ),
                expected=expected,
                tail=excerpt,
                worldview=worldview,
                create_label=f"既出プロット_{index + 1}",
                past_count=expected,
                existing=[
                    PlotScene(text=item, done=True, from_source=True)
                    for item in later
                ],
            )
            groups.append(lines)
            later = lines + later
        chronological = [line for group in reversed(groups) for line in group]
        return [
            PlotScene(text=line, done=True, from_source=True)
            for line in chronological
        ]

    def _generate_plot_lines(
        self,
        prompt: str,
        expected: int,
        tail: str,
        worldview: str,
        create_label: str = "プロット作成",
        past_count: int = 0,
        existing: Iterable[PlotScene] | None = None,
    ) -> list[str]:
        last_error = ""
        existing_list = list(existing or [])
        current_prompt = prompt
        for attempt in range(1, 4):
            retry_label = f"{create_label}_再試行{attempt}"
            raw = self.client.generate(
                current_prompt,
                temperature=0.35 + 0.1 * (attempt - 1),
                num_predict=1024,
                label=create_label if attempt == 1 else retry_label,
            )
            format_label = (
                "プロット整形"
                if attempt == 1
                else f"プロット整形_再試行{attempt}"
            )
            formatted = self.client.generate(
                prompt_normalize_plot(
                    raw,
                    expected,
                    tail,
                    worldview,
                    past_count=past_count,
                    no_shift=self.stats.no_shift,
                    ending=self.stats.ending,
                ),
                temperature=0.2,
                num_predict=1024,
                show_stream=False,
                label=format_label,
            )
            lines = extract_scene_lines(formatted, expected)
            if len(lines) < expected:
                lines = extract_scene_lines(raw, expected)
            if len(lines) >= expected:
                lines = [
                    re.sub(r"\s+", " ", line).strip()
                    for line in lines[:expected]
                ]
                duplicated = existing_list and any(
                    is_duplicate_plot(line, existing_list) for line in lines
                )
                stale_future = future_plots_stale(
                    lines,
                    past_count=past_count,
                    existing=existing_list,
                    tail=tail,
                )
                if duplicated or (past_count < expected and stale_future):
                    if self.stats.no_shift and past_count < expected:
                        return lines
                    last_error = "新しいシーンが既存内容と重複しています"
                    log(f"{last_error}。再試行 {attempt}/3")
                    if past_count >= expected:
                        current_prompt = (
                            prompt
                            + "\n\n【再出力】後続シーンと重複せず、"
                            "この断片に書かれている出来事だけを"
                            "時系列で出力すること。"
                        )
                    else:
                        current_prompt = (
                            prompt
                            + "\n\n【重要】新しいシーンが既存の本文・"
                            "プロットと同じ状況に寄っています。"
                            "場所・時間・相手・目的のいずれかを変えて"
                            "場面転換し、まだ書かれていない展開だけを"
                            "書いてください。"
                        )
                    continue
                return lines
            last_error = f"{len(lines)} 行しか得られませんでした"
            log(f"プロットの行数が足りません（{last_error}）。再試行 {attempt}/3")
            current_prompt = (
                prompt
                + "\n\n【再出力】指定行数だけ、1行1シーンの日本語で"
                "出力すること。"
            )
        raise RuntimeError(f"プロットの生成に失敗しました: {last_error}")

    def refresh_worldview(self, worldview: str) -> str:
        if self.honpen_path.exists():
            honpen = read_text(self.honpen_path)
        else:
            honpen = ""
        recent = split_honpen_scenes(honpen)[-WORLDVIEW_UPDATE_INTERVAL:]
        excerpt = last_chars("\n\n".join(recent), WORLDVIEW_REVIEW_CHARS)
        if not excerpt.strip():
            return worldview

        log("本編を確認し、世界観への追記が必要か調べています...")
        raw = self.client.generate(
            prompt_update_worldview(worldview, excerpt),
            temperature=0.3,
            num_predict=2048,
            num_ctx=16384,
            label="世界観更新",
        )
        updated = clean_llm_text(raw)
        if not updated or re.fullmatch(r"(更新なし|変更なし|追記なし)[。．]?", updated):
            log("世界観の更新応答が空、または変更なしのため、現行を維持します。")
            return worldview

        write_text(self.worldview_path, updated)
        self.stats.worldview_updates += 1
        log(f"世界観.txt を更新しました（{len(updated)} 字）。")
        return updated

    def extend_plot(
        self,
        plot: list[PlotScene],
        tail: str,
        worldview: str,
        count: int = FUTURE_PLOT_COUNT,
    ) -> list[PlotScene]:
        log(f"プロットの続き（次の{count}シーン）を作成しています...")
        lines = self._generate_plot_lines(
            prompt_next_plot(
                tail,
                plot,
                worldview,
                self.news_text,
                expected=count,
                no_shift=self.stats.no_shift,
                ending=self.stats.ending,
            ),
            expected=count,
            tail=tail,
            worldview=worldview,
            create_label="プロット追加",
            past_count=0,
            existing=plot,
        )
        new_scenes = [PlotScene(text=line, done=False) for line in lines]
        plot.extend(new_scenes)
        write_text(self.plot_path, dump_plot(plot))
        self.stats.plot_extensions += 1
        log("プロット.txt に次のシーンを追記しました。")
        return plot

    def add_upcoming_scenes(
        self,
        source: str,
        worldview: str,
        plot: list[PlotScene],
        count: int,
    ) -> list[PlotScene]:
        """本編は書かず、これから書くシーンだけをプロットへ足す。"""
        already = 0
        if not self.stats.plot_reused:
            already = FUTURE_PLOT_COUNT
        remaining = max(0, count - already)
        if remaining == 0:
            log("本編は書かず、プロットの作成のみ行いました。")
            return plot
        log(f"本編は書かず、プロットに {remaining} シーン追加します。")
        while remaining > 0:
            batch = min(FUTURE_PLOT_COUNT, remaining)
            tail = last_chars(combined_body(source, self.honpen_path))
            plot = self.extend_plot(plot, tail, worldview, count=batch)
            remaining -= batch
        return plot

    def write_scenes(
        self,
        source: str,
        worldview: str,
        plot: list[PlotScene],
        count: int,
    ) -> None:
        for i in range(1, count + 1):
            pending_indexes = [
                idx for idx, scene in enumerate(plot) if not scene.done
            ]
            if not pending_indexes:
                tail = last_chars(combined_body(source, self.honpen_path))
                plot = self.extend_plot(plot, tail, worldview)
                pending_indexes = [
                    idx for idx, scene in enumerate(plot) if not scene.done
                ]
                if not pending_indexes:
                    raise RuntimeError("追加プロットを作成できませんでした。")

            target_index = pending_indexes[0]
            target = plot[target_index]
            prev = previous_scenes(plot, target_index)
            tail = last_chars(combined_body(source, self.honpen_path))

            log(f"[{i}/{count}] シーンを執筆しています: {target.text[:40]}...")
            raw = self.client.generate(
                prompt_write_scene(
                    worldview,
                    prev,
                    target,
                    tail,
                    no_shift=self.stats.no_shift,
                    ending=self.stats.ending,
                ),
                temperature=0.85,
                num_predict=4096,
                label=f"シーン執筆_{i}",
            )
            scene_text = strip_repeated_source(
                clean_llm_text(raw),
                source,
                tail,
                read_text(self.honpen_path)
                if self.honpen_path.exists()
                else "",
            )
            if not scene_text:
                raise RuntimeError("シーン本文が空でした。")

            append_scene(self.honpen_path, scene_text)
            target.done = True
            write_text(self.plot_path, dump_plot(plot))
            self.stats.scenes_written += 1
            self.stats.scene_chars += len(scene_text)
            log(f"本編.txt に追記しました（{len(scene_text)} 字）。")

            done_written = written_scene_count(plot)
            interval = WORLDVIEW_UPDATE_INTERVAL
            if done_written > 0 and done_written % interval == 0:
                worldview = self.refresh_worldview(worldview)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ollama を利用して小説の続きを執筆します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="入力テキストファイル",
    )
    parser.add_argument(
        "-n",
        "--scenes",
        type=int,
        default=DEFAULT_SCENES,
        metavar="N",
        help=(
            "本編.txt に作成するシーン数"
            "（--plot-only 時はプロットへ追加するシーン数）"
        ),
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama の API URL",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="使用する Ollama モデル名",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Ollama API のタイムアウト秒数",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="既存の世界観・プロット・本編を破棄して最初から生成する",
    )
    parser.add_argument(
        "--news",
        action="store_true",
        help="ニュースを取得し、新しいプロットに時事ネタを織り込む",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="本編は書かず、プロットへこれから書くシーンだけを追加する",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="生成テキストと進捗メッセージの表示を抑える",
    )
    parser.add_argument(
        "--no-shift",
        action="store_true",
        help=(
            "同じ出来事の禁止と場面切り替えの促しを"
            "プロンプトから外す"
        ),
    )
    parser.add_argument(
        "--ending",
        action="store_true",
        help=(
            "小説の締めくくりとしてシーン末尾をきれいに閉じ、"
            "読後感を良くする"
        ),
    )
    args = parser.parse_args(argv)
    if args.scenes < 1:
        parser.error("シーン数は 1 以上を指定してください。")
    return args


def save_stats(
    path: Path,
    *,
    input_path: Path,
    work_dir: Path,
    args: argparse.Namespace,
    stats: RunStats,
    prompt_logger: PromptLogger,
    ok: bool,
    plot: list[PlotScene] | None = None,
) -> None:
    elapsed = stats.elapsed_seconds()
    prev_scenes = 0
    if not args.fresh:
        raw_scenes = read_stats_field(path, "本編の累計シーン数")
        try:
            prev_scenes = int(raw_scenes)
        except ValueError:
            prev_scenes = 0
    raw_seconds = read_stats_field(path, "累計処理時間秒")
    try:
        prev_seconds = float(raw_seconds)
    except ValueError:
        prev_seconds = 0.0
    plot_total = 0
    if plot:
        plot_total = written_scene_count(plot)
    total_scenes = prev_scenes + stats.scenes_written
    if not args.fresh:
        total_scenes = max(total_scenes, plot_total)
    else:
        total_scenes = stats.scenes_written
    total_seconds = prev_seconds + elapsed

    worldview_status = "再利用" if stats.worldview_reused else "新規作成"
    plot_status = "再利用" if stats.plot_reused else "新規作成"
    if stats.plot_extensions:
        plot_status += f"（続きを{stats.plot_extensions}回追加）"
    if stats.news_enabled and stats.news_count:
        news_status = f"オン（{stats.news_count}件）"
    elif stats.news_enabled:
        news_status = "オン（取得失敗）"
    else:
        news_status = "オフ"

    history = ""
    if path.exists():
        existing = read_text(path)
        mark = "===== 実行履歴 ====="
        idx = existing.find(mark)
        if idx != -1:
            history = existing[idx + len(mark):].strip()
        else:
            history = existing.strip()

    header = (
        f"入力ファイル: {input_path}\n"
        f"作業フォルダ: {work_dir}\n"
        f"使用モデル: {args.model}\n"
        f"Ollama URL: {args.url}\n"
        f"本編の累計シーン数: {total_scenes}\n"
        f"累計処理時間: {format_duration(total_seconds)}\n"
        f"累計処理時間秒: {int(round(total_seconds))}\n"
        f"最終実行: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"時事ネタ: {news_status}\n"
        f"最終結果: {'完了' if ok else 'エラー'}\n"
    )
    run = (
        f"[{stats.started_at.strftime('%Y-%m-%d %H:%M:%S')}]\n"
        f"モデル: {args.model}\n"
        f"URL: {args.url}\n"
        f"今回作成したシーン数: {stats.scenes_written}\n"
        f"今回の本編追記字数: {stats.scene_chars}\n"
        f"本編の累計シーン数: {total_scenes}\n"
        f"処理時間: {format_duration(elapsed)}\n"
        f"世界観: {worldview_status}\n"
        f"プロット: {plot_status}\n"
        f"世界観更新回数: {stats.worldview_updates}\n"
        f"時事ネタ: {news_status}\n"
        f"本編執筆: {'なし（プロットのみ）' if stats.plot_only else 'あり'}\n"
        f"場面転換の促し: {'オフ' if stats.no_shift else 'オン'}\n"
        f"締めくくり: {'オン' if stats.ending else 'オフ'}\n"
        f"保存したプロンプト数: {prompt_logger.saved}\n"
        f"結果: {'完了' if ok else 'エラー'}\n"
    )
    body = header + "\n===== 実行履歴 =====\n\n"
    if history:
        body += history.rstrip() + "\n\n"
    body += run
    write_text(path, body)
    duration = format_duration(elapsed)
    log(
        f"統計.txt を保存しました"
        f"（今回 {stats.scenes_written} シーン / {duration}）。"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_quiet(args.quiet)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        log(f"入力ファイルが見つかりません: {input_path}", always=True)
        return 1

    source = read_text(input_path)
    if source.startswith("\ufeff"):
        source = source[1:]
    source = source.strip("\n")
    if not source.strip():
        log("入力ファイルが空です。", always=True)
        return 1

    work_dir = OUTPUT_ROOT / input_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for name in ("世界観.txt", "プロット.txt", "本編.txt"):
            path = work_dir / name
            if path.exists():
                path.unlink()
                log(f"{name} を削除しました。")

    write_text(work_dir / "ベース.txt", source)
    honpen_path = work_dir / "本編.txt"
    resuming = (not args.fresh) and (
        (work_dir / "世界観.txt").exists()
        or (work_dir / "プロット.txt").exists()
        or honpen_path.exists()
    )
    log(f"作業フォルダ: {work_dir}")
    log(f"モデル: {args.model}")
    if args.plot_only:
        log(f"プロットへ追加するシーン数: {args.scenes}")
    else:
        log(f"作成するシーン数: {args.scenes}")
    if args.news:
        log("時事ネタモード: オン")
    if args.plot_only:
        log("プロットのみモード: 本編は書きません")
    if args.no_shift:
        log("場面転換の促し: オフ")
    if args.ending:
        log("締めくくりモード: オン")
    if resuming:
        if args.plot_only:
            log(
                "同じ作品の続きです。既存の世界観を利用し、"
                "プロット.txt にシーンを追加します。"
            )
        else:
            log(
                "同じ作品の続きです。既存の世界観・プロットを利用し、"
                "本編.txt に追記します。"
            )

    stats = RunStats(
        news_enabled=args.news,
        plot_only=args.plot_only,
        no_shift=args.no_shift,
        ending=args.ending,
    )
    prompt_logger = PromptLogger(work_dir / "prompt", args.model)
    client = OllamaClient(
        url=args.url,
        model=args.model,
        timeout=args.timeout,
        prompt_logger=prompt_logger,
    )
    writer = StoryWriter(client, work_dir, stats)
    if args.news:
        log("最近のニュースを取得しています...")
        news_items = fetch_recent_news()
        stats.news_count = len(news_items)
        if news_items:
            digest = format_news_digest(news_items)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            write_text(
                work_dir / "時事.txt",
                f"取得時刻: {stamp}\n\n{digest}\n",
            )
            writer.news_text = digest
            log(f"時事ネタ {len(news_items)} 件をプロット作成に使います。")
        else:
            log(
                "ニュースを取得できなかったため、時事ネタなしで続行します。",
                always=True,
            )
    writer.sanitize_outputs()

    ok = False
    plot: list[PlotScene] = []
    try:
        worldview = writer.ensure_worldview(source, force=args.fresh)
        plot = writer.ensure_plot(source, worldview, force=args.fresh)
        if args.plot_only:
            plot = writer.add_upcoming_scenes(
                source, worldview, plot, args.scenes
            )
        else:
            writer.write_scenes(source, worldview, plot, args.scenes)
        ok = True
    except RuntimeError as exc:
        log(str(exc), always=True)
    finally:
        save_stats(
            work_dir / "統計.txt",
            input_path=input_path,
            work_dir=work_dir,
            args=args,
            stats=stats,
            prompt_logger=prompt_logger,
            ok=ok,
            plot=plot,
        )

    if ok:
        if args.plot_only:
            log(f"完了しました: {writer.plot_path}", always=True)
        else:
            log(f"完了しました: {writer.honpen_path}", always=True)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
