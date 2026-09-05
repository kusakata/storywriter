#!/usr/bin/env python3
"""Ollama を利用して小説の続きを執筆するスクリプト。

入力テキストの世界観とプロットを抽出し、指定したシーン数だけ本編を書き足す。

使用例:
  python storywriter.py novel.txt
  python storywriter.py novel.txt -n 6
  python storywriter.py novel.txt --url http://localhost:11434 --model llama3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# 設定（CLI 引数で上書き可能）
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "hf.co/bartowski/Ateron_Gemma-4-Novelist-Eclipse-31B-GGUF:Q4_K_M"
DEFAULT_SCENES = 3
DEFAULT_TIMEOUT = 1800
CHAR_WINDOW = 2000
WORLDVIEW_TARGET_CHARS = 1000
PLOT_SCENE_CHARS = 100
SCENE_TARGET_CHARS = 1000
SCENE_CLIMAX_CHARS = 1500
PAST_PLOT_COUNT = 3
FUTURE_PLOT_COUNT = 3
WORLDVIEW_UPDATE_INTERVAL = 5
WORLDVIEW_REVIEW_CHARS = 6000

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "output"

DONE_MARK = "[済]"
PENDING_MARK = "[未]"

SYSTEM_WRITER = (
    "あなたは日本語の小説家です。指定された形式を厳守し、"
    "前置き・後書き・メタ解説は一切出力せず、求められた内容だけを書いてください。"
    "思考過程、英語のメモ、<|channel> や <channel|> などのタグは絶対に出力しないでください。"
)

_CHANNEL_START = re.compile(
    r"<\|channel\|?>?(?:thought|analysis|commentary)?",
    re.IGNORECASE,
)
_CHANNEL_END = re.compile(r"(?:<channel\|>|<\|channel\|>|<\|end\|>)", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CHANNEL_BLOCK = re.compile(
    r"<\|channel\|?>?(?:thought|analysis|commentary)?.*?(?:<channel\|>|<\|channel\|>|<\|end\|>)",
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

    def to_line(self) -> str:
        mark = DONE_MARK if self.done else PENDING_MARK
        return f"{mark} {self.text}"


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def first_chars(text: str, n: int = CHAR_WINDOW) -> str:
    return text[:n] if len(text) > n else text


def last_chars(text: str, n: int = CHAR_WINDOW) -> str:
    return text[-n:] if len(text) > n else text


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


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
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


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
            buf = buf[match.end() :]
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
            visible.append(buf[: match.start()])
        buf = buf[match.end() :]
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
    pieces = [_CHANNEL_START.sub("", p).strip() for p in _CHANNEL_SPLIT.split(text)]
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


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Ollama クライアント
# ---------------------------------------------------------------------------
class OllamaClient:
    def __init__(self, url: str, model: str, timeout: int) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        system: str = SYSTEM_WRITER,
        temperature: float = 0.8,
        num_predict: int = 2048,
        num_ctx: int = 8192,
        show_stream: bool = True,
    ) -> str:
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
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(f"Ollama エラー: {data['error']}")
                    token = data.get("response", "")
                    chunks.append(token)
                    visible, hold, in_thought = push_visible_stream(hold + token, in_thought)
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
    r"^(?:[-*・]|\[済\]|\[未\])?\s*"
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
        done_mark = line.startswith(DONE_MARK)
        pending_mark = line.startswith(PENDING_MARK)
        if done_mark or pending_mark:
            saw_mark = True
        body = _LINE_PREFIX.sub("", line).strip()
        if not body:
            continue
        scenes.append(PlotScene(text=body, done=done_mark))
    if scenes and not saw_mark:
        for scene in scenes[:PAST_PLOT_COUNT]:
            scene.done = True
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
    if len(items) < PAST_PLOT_COUNT:
        return False
    for scene in items:
        if "\n" in scene.text or "<|channel" in scene.text or "<channel|" in scene.text:
            return False
        if not looks_japanese(scene.text):
            return False
    return True


def previous_scenes(plot: list[PlotScene], target_index: int, count: int = PAST_PLOT_COUNT) -> list[PlotScene]:
    start = max(0, target_index - count)
    return plot[start:target_index]


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


def prompt_initial_plot(tail: str) -> str:
    return f"""次の小説本文の末尾を読み、プロットを作成してください。

出力ルール:
- 合計{PAST_PLOT_COUNT + FUTURE_PLOT_COUNT}行だけ出力する
- 1行1シーン、各シーン約{PLOT_SCENE_CHARS}字
- 最初の{PAST_PLOT_COUNT}行は「これまでのシーン」（直近で起きた出来事）
- 残りの{FUTURE_PLOT_COUNT}行は「これからのシーン」（自然な続き）
- 見出し・番号・前置きは不要。シーンの説明文だけを1行ずつ書く
- これからのシーンは、末尾の状況から無理なく続く展開にする
- 思考過程、英語、特殊タグは出力しない
- 各行の中に改行を入れない

--- 本文末尾 ---
{tail}
"""


def prompt_next_plot(tail: str, plot: list[PlotScene]) -> str:
    plot_text = dump_plot(plot).strip()
    return f"""既存のプロットと本文末尾を踏まえ、これから書く次のシーンを{FUTURE_PLOT_COUNT}つ作成してください。

出力ルール:
- {FUTURE_PLOT_COUNT}行だけ出力する
- 1行1シーン、各シーン約{PLOT_SCENE_CHARS}字
- 既存プロットの続きとして自然につながる
- 既存シーンの繰り返しは禁止
- 見出し・番号・前置きは不要
- 思考過程、英語、特殊タグは出力しない
- 各行の中に改行を入れない

--- 既存プロット ---
{plot_text}

--- 本文末尾 ---
{tail}
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


def prompt_normalize_plot(raw: str, expected: int, tail: str) -> str:
    return f"""次のテキストは小説のシーン説明です。内容を保ったまま、指定形式へ加工してください。

形式:
- ちょうど{expected}行だけ出力する
- 1行に1シーンだけ書く
- 各行の中に改行を入れない
- 各行は日本語のみで、約{PLOT_SCENE_CHARS}字のシーン説明にする
- 番号、箇条書き記号、見出し、英語、思考過程、特殊タグは付けない
- 前置きと後書きは禁止
- 元テキストが英語のメモや思考過程でも、参考本文から読み取れる出来事を日本語のシーン説明に直す

--- 参考にする本文末尾 ---
{tail}

--- 加工するテキスト ---
{raw}
"""


def prompt_write_scene(
    worldview: str,
    prev: list[PlotScene],
    target: PlotScene,
    tail: str,
) -> str:
    prev_text = "\n".join(f"- {scene.text}" for scene in prev) if prev else "- （冒頭のため直前シーンなし）"
    return f"""以下の情報を踏まえ、指定されたシーンの本文を執筆してください。

# 世界観
{worldview.strip()}

# これまでのシーン
{prev_text}

# 今回執筆するシーン
- {target.text}

# 直前の本文（末尾）
{tail}

執筆ルール:
- 直前の本文の自然な続きとして書く（本文の繰り返しは禁止）
- 今回のシーンで起きるべき出来事を、小説の地の文と会話で描く
- 分量は約{SCENE_TARGET_CHARS}字を基本とする
- 対立、告白、暴力、重要な選択など物語が盛り上がる場面では、必要に応じて約{SCENE_CLIMAX_CHARS}字まで伸ばしてよい
- 日本語として破綻なく、そのシーンとして完結する形で終わる（文の途中で切らない）
- タイトル、シーン番号、解説、メタ情報、思考過程、特殊タグは出力しない
- 本文のみを日本語で出力する
"""


# ---------------------------------------------------------------------------
# 執筆ワークフロー
# ---------------------------------------------------------------------------
class StoryWriter:
    def __init__(self, client: OllamaClient, work_dir: Path) -> None:
        self.client = client
        self.work_dir = work_dir
        self.base_path = work_dir / "ベース.txt"
        self.worldview_path = work_dir / "世界観.txt"
        self.plot_path = work_dir / "プロット.txt"
        self.honpen_path = work_dir / "本編.txt"

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
                write_text(self.worldview_path, text)
                log("既存の 世界観.txt を再利用します。")
                return text
            log("世界観.txt が不正なため再生成します。")

        head = first_chars(source)
        log(f"世界観を抽出しています（冒頭 {len(head)} 字）...")
        raw = self.client.generate(
            prompt_worldview(head),
            temperature=0.4,
            num_predict=2048,
        )
        worldview = clean_llm_text(raw)
        if not worldview:
            raise RuntimeError("世界観の抽出に失敗しました（空の応答）。")
        write_text(self.worldview_path, worldview)
        log(f"世界観.txt を保存しました（{len(worldview)} 字）。")
        return worldview

    def ensure_plot(self, source: str, force: bool) -> list[PlotScene]:
        if self.plot_path.exists() and not force:
            scenes = parse_plot_file(read_text(self.plot_path))
            if plot_lines_usable(scenes):
                log("既存の プロット.txt を再利用します。")
                return scenes
            log("プロット.txt の形式が不正なため再生成します。")

        tail = last_chars(source)
        log(f"プロットを作成しています（末尾 {len(tail)} 字）...")
        expected = PAST_PLOT_COUNT + FUTURE_PLOT_COUNT
        lines = self._generate_plot_lines(
            prompt_initial_plot(tail),
            expected=expected,
            tail=tail,
        )
        plot = [
            PlotScene(text=line, done=(i < PAST_PLOT_COUNT))
            for i, line in enumerate(lines)
        ]
        write_text(self.plot_path, dump_plot(plot))
        log("プロット.txt を保存しました。")
        return plot

    def _generate_plot_lines(self, prompt: str, expected: int, tail: str) -> list[str]:
        last_error = ""
        for attempt in range(1, 4):
            raw = self.client.generate(
                prompt if attempt == 1 else prompt + "\n\n【再出力】指定行数だけ、1行1シーンの日本語で出力すること。",
                temperature=0.35 + 0.1 * (attempt - 1),
                num_predict=1024,
            )
            formatted = self.client.generate(
                prompt_normalize_plot(raw, expected, tail),
                temperature=0.2,
                num_predict=1024,
            )
            lines = extract_scene_lines(formatted, expected)
            if len(lines) < expected:
                lines = extract_scene_lines(raw, expected)
            if len(lines) >= expected:
                return [re.sub(r"\s+", " ", line).strip() for line in lines[:expected]]
            last_error = f"{len(lines)} 行しか得られませんでした"
            log(f"プロットの行数が足りません（{last_error}）。再試行 {attempt}/3")
        raise RuntimeError(f"プロットの生成に失敗しました: {last_error}")

    def refresh_worldview(self, worldview: str) -> str:
        honpen = read_text(self.honpen_path) if self.honpen_path.exists() else ""
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
        )
        updated = clean_llm_text(raw)
        if not updated or re.fullmatch(r"(更新なし|変更なし|追記なし)[。．]?", updated):
            log("世界観の更新応答が空、または変更なしのため、現行を維持します。")
            return worldview

        write_text(self.worldview_path, updated)
        log(f"世界観.txt を更新しました（{len(updated)} 字）。")
        return updated

    def extend_plot(self, plot: list[PlotScene], tail: str) -> list[PlotScene]:
        log("プロットの続き（次の3シーン）を作成しています...")
        lines = self._generate_plot_lines(
            prompt_next_plot(tail, plot),
            expected=FUTURE_PLOT_COUNT,
            tail=tail,
        )
        new_scenes = [PlotScene(text=line, done=False) for line in lines]
        plot.extend(new_scenes)
        write_text(self.plot_path, dump_plot(plot))
        log("プロット.txt に次のシーンを追記しました。")
        return plot

    def write_scenes(
        self,
        source: str,
        worldview: str,
        plot: list[PlotScene],
        count: int,
    ) -> None:
        for i in range(1, count + 1):
            pending_indexes = [idx for idx, scene in enumerate(plot) if not scene.done]
            if not pending_indexes:
                tail = last_chars(combined_body(source, self.honpen_path))
                plot = self.extend_plot(plot, tail)
                pending_indexes = [idx for idx, scene in enumerate(plot) if not scene.done]
                if not pending_indexes:
                    raise RuntimeError("追加プロットを作成できませんでした。")

            target_index = pending_indexes[0]
            target = plot[target_index]
            prev = previous_scenes(plot, target_index)
            tail = last_chars(combined_body(source, self.honpen_path))

            log(f"[{i}/{count}] シーンを執筆しています: {target.text[:40]}...")
            raw = self.client.generate(
                prompt_write_scene(worldview, prev, target, tail),
                temperature=0.85,
                num_predict=4096,
            )
            scene_text = clean_llm_text(raw)
            if not scene_text:
                raise RuntimeError("シーン本文が空でした。")

            append_scene(self.honpen_path, scene_text)
            target.done = True
            write_text(self.plot_path, dump_plot(plot))
            log(f"本編.txt に追記しました（{len(scene_text)} 字）。")

            honpen_count = len(split_honpen_scenes(read_text(self.honpen_path)))
            if honpen_count % WORLDVIEW_UPDATE_INTERVAL == 0:
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
        help="本編.txt に作成するシーン数",
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
    args = parser.parse_args(argv)
    if args.scenes < 1:
        parser.error("シーン数は 1 以上を指定してください。")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        log(f"入力ファイルが見つかりません: {input_path}")
        return 1

    source = read_text(input_path)
    if source.startswith("\ufeff"):
        source = source[1:]
    source = source.strip("\n")
    if not source.strip():
        log("入力ファイルが空です。")
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
    log(f"作業フォルダ: {work_dir}")
    log(f"モデル: {args.model}")
    log(f"作成するシーン数: {args.scenes}")

    client = OllamaClient(url=args.url, model=args.model, timeout=args.timeout)
    writer = StoryWriter(client, work_dir)
    writer.sanitize_outputs()

    try:
        worldview = writer.ensure_worldview(source, force=args.fresh)
        plot = writer.ensure_plot(source, force=args.fresh)
        writer.write_scenes(source, worldview, plot, args.scenes)
    except RuntimeError as exc:
        log(str(exc))
        return 1

    log(f"完了しました: {writer.honpen_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
