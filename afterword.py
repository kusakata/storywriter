#!/usr/bin/env python3
"""Ollama を利用して小説のあとがきを作成するスクリプト。

storywriter.py と同じ入力ファイルを指定し、
output/<入力ファイル名>/ にある世界観とプロットからあとがきを書く。
当該フォルダが無い場合は何も生成しない。

使用例:
  python afterword.py novel.txt
  python afterword.py novel.txt --url http://localhost:11434 --model llama3
  python afterword.py novel.txt -q
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from storywriter import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TIMEOUT,
    OUTPUT_ROOT,
    OllamaClient,
    PromptLogger,
    clean_llm_text,
    log,
    looks_japanese,
    read_text,
    set_quiet,
    write_text,
)

AFTERWORD_TARGET_CHARS = 1000
PLOT_CONTEXT_CHARS = 4000

SYSTEM_AFTERWORD = (
    "あなたは日本語の小説家です。単行本に載せる「あとがき」だけを、"
    "丁寧な口調で書いてください。"
    f"全体を{AFTERWORD_TARGET_CHARS}文字以下に収めてください。"
    "前置き・後書き・メタ解説は一切出力せず、あとがき本文だけを"
    "書いてください。"
    "思考過程、英語のメモ、<|channel> や <channel|> などのタグは"
    "絶対に出力しないでください。"
)


def clip_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head].rstrip() + "\n……\n" + text[-tail:].lstrip()


def prompt_afterword(
    title_hint: str,
    worldview: str,
    plot: str,
) -> str:
    return f"""次の世界観とプロットを手がかりに、この作品の著者として
「あとがき」を書いてください。事実の記録ではなく、著者の立場から
執筆の事情を想像して構いません。

# 作品の手がかり（ファイル名）
{title_hint}

# 世界観
{worldview.strip()}

# プロット
{plot.strip()}

書いてほしいこと（必須ではない）:
- どのような背景・きっかけでこの作品を書いたのか、いつ書いたのか
- 構成、人物、情景、会話など、どこに注力し、どんな工夫をしたのか
- 小説本文とは違う、丁寧な「です・ます」調、あるいはもっとフランクで砕けた書き方
- あとがきの主題は執筆の背景と工夫に置く

含めてもよい話（想像して短く書いてよい）:
- 作者がお気に入りのキャラクターと、その理由
- 作者の友達や家族の話
- 最近ハマっているゲームや音楽、映画、家電、スポーツ、旅行など
- 今後書きたいテーマ

制約:
- {AFTERWORD_TARGET_CHARS}文字以下。超えないこと
- プロットの要約やあらすじの再掲はしない
- 登場人物の口調や作中の文体をまねない
- 「以下があとがきです」などの前置きは書かない
- 見出しは「あとがき」だけでもよく、無くてもよい
- 思考過程や特殊タグは出力しない
- あとがき本文のみを日本語で出力する
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ollama を利用して小説のあとがきを作成します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="storywriter.py に渡したのと同じ入力ファイル（.txt / .md）",
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
        "-q",
        "--quiet",
        action="store_true",
        help="生成テキストと進捗メッセージの表示を抑える",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_quiet(args.quiet)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        log(f"入力ファイルが見つかりません: {input_path}", always=True)
        return 1

    work_dir = OUTPUT_ROOT / input_path.stem
    if not work_dir.is_dir():
        log(
            f"作業フォルダがありません: {work_dir}\n"
            "先に storywriter.py を実行してください。",
            always=True,
        )
        return 1

    worldview_path = work_dir / "世界観.txt"
    plot_path = work_dir / "プロット.txt"
    missing = [
        path.name
        for path in (worldview_path, plot_path)
        if not path.is_file()
    ]
    if missing:
        log(
            "必要なファイルがありません: " + "、".join(missing) + "\n"
            f"{work_dir} で storywriter.py を先に実行してください。",
            always=True,
        )
        return 1

    worldview = read_text(worldview_path).strip()
    plot = clip_text(read_text(plot_path), PLOT_CONTEXT_CHARS)
    if not worldview or not plot:
        log("世界観.txt または プロット.txt が空です。", always=True)
        return 1

    log(f"作業フォルダ: {work_dir}")
    log(f"モデル: {args.model}")
    log("あとがきを作成しています...")

    prompt_logger = PromptLogger(work_dir / "prompt", args.model)
    client = OllamaClient(
        url=args.url,
        model=args.model,
        timeout=args.timeout,
        prompt_logger=prompt_logger,
    )
    try:
        raw = client.generate(
            prompt_afterword(input_path.stem, worldview, plot),
            system=SYSTEM_AFTERWORD,
            temperature=0.8,
            num_predict=2048,
            label="あとがき",
        )
    except RuntimeError as exc:
        log(str(exc), always=True)
        return 1

    text = clean_llm_text(raw)
    if not text or not looks_japanese(text):
        log("あとがきの生成に失敗しました（空または不正な応答）。", always=True)
        return 1

    out_path = work_dir / "あとがき.txt"
    write_text(out_path, text)
    log(f"あとがき.txt を保存しました（{len(text)} 字）。")
    log(f"完了しました: {out_path}", always=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
