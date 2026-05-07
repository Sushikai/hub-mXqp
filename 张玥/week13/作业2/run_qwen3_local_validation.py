"""使用本地已下载文件运行 Qwen3 0.6B notebook 的核心流程。

这个脚本会复用 `01_Qwen3.ipynb` 里的模型定义代码单元，再加载已经下载到本地的
权重和 tokenizer，从而绕开 `hf_hub_download` 过程中可能出现的
Hugging Face SSL 下载问题。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


ROOT_DIR = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT_DIR / "01_Qwen3.ipynb"
MODEL_DIR = ROOT_DIR / "Qwen3-0.6B"
MAX_NEW_TOKENS = 80
PROMPT = "Give me a short introduction to large language models."


def load_notebook() -> dict:
    """以 JSON 形式读取 Week13 的 notebook 文件。"""
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def execute_core_cells(notebook: dict) -> dict:
    """执行构建模型和生成文本所需的 notebook 代码单元。

    这里会有意跳过原 notebook 中的下载单元，因为这次作业已经提前把公开
    模型文件下载到了 `Qwen3-0.6B` 目录中。
    """
    context: dict = {"__name__": "__main__", "Path": Path}
    code_cells = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15, 16, 17, 20, 23]

    for idx in code_cells:
        source = "".join(notebook["cells"][idx].get("source", []))
        exec(compile(source, f"{NOTEBOOK_PATH.name}#cell{idx}", "exec"), context)

    return context


def load_local_artifacts(context: dict) -> None:
    """把本地 safetensors 权重和 tokenizer 加载到 notebook 上下文中。"""
    from safetensors.torch import load_file

    weights_path = MODEL_DIR / "model.safetensors"
    tokenizer_path = MODEL_DIR / "tokenizer.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"未找到模型权重文件：{weights_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"未找到 tokenizer 文件：{tokenizer_path}")

    weights_dict = load_file(str(weights_path))
    context["load_weights_into_qwen"](context["model"], context["QWEN3_CONFIG"], weights_dict)
    context["model"].to(context["device"])
    del weights_dict

    repo_id = f"Qwen/Qwen3-{context['CHOOSE_MODEL']}"
    context["tokenizer"] = context["Qwen3Tokenizer"](
        tokenizer_file_path=str(tokenizer_path),
        repo_id=repo_id,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )


def generate_text(context: dict) -> str:
    """使用已经加载到 GPU 的模型执行一次最小文本生成。"""
    import torch

    tokenizer = context["tokenizer"]
    model = context["model"]
    device = context["device"]
    generate_text_basic_stream = context["generate_text_basic_stream"]

    input_token_ids = tokenizer.encode(PROMPT)
    input_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    pieces: list[str] = []
    for token in generate_text_basic_stream(
        model=model,
        token_ids=input_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
    ):
        token_id = token.squeeze(0).tolist()
        pieces.append(tokenizer.decode(token_id))

    return "".join(pieces)


def main() -> None:
    notebook = load_notebook()
    context = execute_core_cells(notebook)
    load_local_artifacts(context)
    output = generate_text(context)

    print("Notebook 路径:", NOTEBOOK_PATH)
    print("模型目录:", MODEL_DIR)
    print("运行设备:", context["device"])
    print("测试提示词:", PROMPT)
    print("生成结果:")
    print(output)


if __name__ == "__main__":
    main()
