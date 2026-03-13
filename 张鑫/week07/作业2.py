import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


API_KEY = "bbb"
BASE_URL = "aaa"
MODEL = "gpt-5.4"


def load_label_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def build_system_prompt(domains: List[str], intents: List[str], slots: List[str]) -> str:
    return f"""你是一个信息解析助手。你的任务是从用户输入中抽取 domain、intent 和 slots。

【标签约束】
- domain 只能从以下集合中选一个：
{", ".join(domains)}
- intent 只能从以下集合中选一个：
{", ".join(intents)}
- slots 的 key 只能从以下集合中选：
{", ".join(slots)}

【抽取规则】
1. slots 的 value 必须是用户原文中的连续片段，不得改写。
2. 只输出在原文中明确出现的槽位；不确定就不要编造。
3. 最终仅输出 JSON，不要输出解释文字，不要输出 markdown。
4. JSON 格式固定为：
{{
  "domain": "...",
  "intent": "...",
  "slots": {{
    "slot_key": "slot_value"
  }}
}}
"""


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("模型输出中未找到 JSON 对象")
    return match.group(0)


def detect_provider_issue(text: str) -> str:
    low = text.lower()
    if "not enough available apinum" in low or "please go to recharge" in low:
        return "接口账户额度不足，请先充值后再调用。"
    if "please check" in low and "key" in low:
        return "API Key 无效或不匹配当前平台，请检查 key 是否正确。"
    if "invalid api key" in low or "incorrect api key" in low:
        return "API Key 无效，请更换可用 key。"
    if "unauthorized" in low or "401" in low:
        return "鉴权失败（401），请检查 key 与 base_url。"
    return ""


def validate_output(result: Dict[str, Any], domains: List[str], intents: List[str], slots: List[str]) -> List[str]:
    errors: List[str] = []
    if result.get("domain") not in domains:
        errors.append(f"domain 不在标签集合中: {result.get('domain')}")
    if result.get("intent") not in intents:
        errors.append(f"intent 不在标签集合中: {result.get('intent')}")

    slot_map = result.get("slots", {})
    if not isinstance(slot_map, dict):
        errors.append("slots 不是对象(dict)")
    else:
        for k, v in slot_map.items():
            if k not in slots:
                errors.append(f"slot key 不在标签集合中: {k}")
            if not isinstance(v, str):
                errors.append(f"slot value 不是字符串: {k}={v}")
    return errors


def _chat_once(client: OpenAI, model: str, messages: List[Dict[str, str]], force_json: bool) -> Any:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if force_json:
        # Some OpenAI-compatible providers support this, some don't.
        kwargs["response_format"] = {"type": "json_object"}
    kwargs["timeout"] = 30
    return client.chat.completions.create(**kwargs)


def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_text: str,
    debug: bool = False,
) -> Dict[str, Any]:
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    # Try 1: JSON mode if provider supports it.
    try:
        resp = _chat_once(client, model, base_messages, force_json=True)
    except Exception:
        # Fallback for providers not supporting response_format.
        resp = _chat_once(client, model, base_messages, force_json=False)

    message = resp.choices[0].message
    content = normalize_content(getattr(message, "content", ""))

    if debug:
        print("[DEBUG] raw assistant content:")
        print(content if content else "<EMPTY>")

    issue = detect_provider_issue(content)
    if issue:
        raise RuntimeError(issue)

    # If content is empty or not JSON, do a second pass with stricter instruction.
    try:
        raw_json = extract_json_block(content)
        return json.loads(raw_json)
    except Exception:
        repair_messages = [
            {
                "role": "system",
                "content": "你必须只输出一个合法JSON对象，不要输出任何额外文本。",
            },
            {
                "role": "user",
                "content": (
                    f"请对这句话做信息抽取并只返回JSON：{user_text}\n"
                    "字段必须是 domain, intent, slots。"
                ),
            },
        ]
        resp2 = _chat_once(client, model, repair_messages, force_json=False)
        message2 = resp2.choices[0].message
        content2 = normalize_content(getattr(message2, "content", ""))
        if debug:
            print("[DEBUG] repair assistant content:")
            print(content2 if content2 else "<EMPTY>")
        issue2 = detect_provider_issue(content2)
        if issue2:
            raise RuntimeError(issue2)
        raw_json2 = extract_json_block(content2)
        return json.loads(raw_json2)


def main() -> None:
    parser = argparse.ArgumentParser(description="作业2：基于提示词的信息解析（domain/intent/slots）")
    parser.add_argument("--text", type=str, default="", help="要抽取的用户输入")
    parser.add_argument("--model", type=str, default=(MODEL or os.getenv("FE8_MODEL", "gpt-5.4")), help="模型名")
    parser.add_argument(
        "--base-url",
        type=str,
        default=(BASE_URL or os.getenv("FE8_BASE_URL", "https://api.fe8.cn/v1")),
        help="OpenAI兼容接口地址",
    )
    parser.add_argument("--api-key", type=str, default="", help="API Key（优先级高于环境变量）")
    parser.add_argument("--show-prompt", action="store_true", help="打印系统提示词")
    parser.add_argument("--debug", action="store_true", help="打印原始模型返回内容")
    args = parser.parse_args()

    api_key = (
        args.api_key
        or API_KEY
        or os.getenv("FE8_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        raise RuntimeError("请传入 --api-key，或设置 FE8_API_KEY / OPENAI_API_KEY")

    domains = load_label_list(DATA_DIR / "domains.txt")
    intents = load_label_list(DATA_DIR / "intents.txt")
    slots = load_label_list(DATA_DIR / "slots.txt")
    system_prompt = build_system_prompt(domains, intents, slots)

    if args.show_prompt:
        print(system_prompt)
        print("=" * 80)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    test_inputs = [args.text] if args.text.strip() else [
        "查询许昌到中山的高铁票",
        "帮我给张三发短信",
        "把我们一起去玩吧翻译成英文",
    ]

    for idx, text in enumerate(test_inputs, start=1):
        print(f"[{idx}] 输入: {text}")
        try:
            result = call_llm(client, args.model, system_prompt, text, debug=args.debug)
            errors = validate_output(result, domains, intents, slots)
            print("输出:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if errors:
                print("校验告警:")
                for e in errors:
                    print(f"- {e}")
        except Exception as e:
            print(f"调用失败: {e}")
            if args.debug:
                print("建议：先确认该模型是否支持 chat.completions，或换成该平台文档推荐模型名。")
        print("-" * 80)


if __name__ == "__main__":
    main()
