"""OpenAI 兼容的服务接口，带一个聊天网页：

    tinygo serve                                          # 不写存档：用 out/small 里走得最远的那个
    tinygo serve out/medium/pretrain/ckpt.pt --port 8001  # 然后浏览器打开 http://127.0.0.1:8001/

网页有两种用法：对话（SFT 之后的存档），续写（预训练、退火的存档还不会回答问题，只会接着你写的开头往下写）。
接口：POST /v1/chat/completions（messages、max_tokens、temperature、top_k、tools、stream）、POST /v1/completions（prompt，续写）、
GET /v1/models、GET /v1/info（存档、阶段、步数、参数量）。stream 为真时按 OpenAI 的格式边生成边推送（server-sent events），
读的一方断开（网页上点了停止）就不再往下生成。
模型写出工具调用时，按 OpenAI 的格式放进 message.tool_calls，finish_reason 是 tool_calls；工具的结果用 role=tool 的消息传回来。
这样小米的 mimoagent 这类智能体外壳可以直接驱动我们的模型。一次只算一个请求，够演示、评测和收集轨迹用。
"""
import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from lm import LanguageModel, ModelConfig
from training.trainer_utils import find_stage, load_checkpoint, pick_device

from .generate import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, generate, prompt_ids, stream
from .template import ChatFormat
from .tokenizer import DEFAULT_MODEL, Tokenizer

PAGE = Path(__file__).parent / "web" / "chat.html"
BASE_STAGES = ("pretrain", "anneal")  # 这些阶段的存档还没学过对话格式，网页默认用续写


def to_messages(messages: list[dict], tools: list | None) -> list[dict]:
    """OpenAI 的消息格式转成我们的：工具说明并进 system；assistant 的 tool_calls 转成 {name, arguments}。"""
    out = []
    if tools:
        specs = [t.get("function", t) for t in tools]
        out.append({"role": "system", "content": "你可以调用下面的工具。调用时在回答里写一个 JSON：{\"name\": 工具名, \"arguments\": {…}}。\n"
                                                + json.dumps(specs, ensure_ascii=False)})
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
        msg = {"role": m["role"] if m["role"] in ("system", "user", "assistant", "tool") else "user", "content": content}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = c.get("function", c)
                args = f.get("arguments") or {}
                calls.append({"name": f.get("name"), "arguments": json.loads(args) if isinstance(args, str) else args})
            msg["tool_calls"] = calls
        if out and out[0]["role"] == "system" and msg["role"] == "system" and len(out) == 1 and tools:
            out[0]["content"] = msg["content"] + "\n" + out[0]["content"]
            continue
        out.append(msg)
    return out


class Service:
    def __init__(self, model, fmt: ChatFormat, name: str, info: dict | None = None):
        self.model, self.fmt, self.name = model, fmt, name
        self.tok = fmt.tok
        self.info = {"name": name, "max_seq_len": model.cfg.max_seq_len, **(info or {})}
        self.lock = threading.Lock()  # 一次只算一个请求

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _room(self, prompt: list[int], body: dict) -> int:
        max_new = min(int(body.get("max_tokens") or 512), self.model.cfg.max_seq_len - len(prompt))
        if max_new <= 0:
            raise ValueError("输入太长，超过了模型的上限，请删掉一些再试")
        return max_new

    @staticmethod
    def _sampling(body: dict) -> dict:
        top_k = body.get("top_k", DEFAULT_TOP_K)
        return dict(temperature=float(body.get("temperature", DEFAULT_TEMPERATURE)), top_k=int(top_k) if top_k else None)

    def _whole(self, prompt: list[int], max_new: int, body: dict, stop: int) -> list[int]:
        with self.lock:
            out = generate(self.model, torch.tensor([prompt], device=self.device), max_new, eos_id=stop, **self._sampling(body))
        return out[0, len(prompt):].tolist()

    def _stream(self, prompt: list[int], max_new: int, body: dict, stop: int, visible):
        """边生成边交出 (新增的文字, 结束原因, 写了几个词)；结束原因只在最后一段有。
        visible 把已经生成的词变成要显示的文字。一个字被拆成几个字节词时先不发，等它拼完整。"""
        ids, shown, reason = [], "", "length"
        with self.lock:
            for token in stream(self.model, torch.tensor([prompt], device=self.device), max_new, eos_id=stop, **self._sampling(body)):
                ids.append(token)
                if token == stop:
                    reason = "stop"
                    break
                text = visible(ids)
                if text.endswith("�") or not text.startswith(shown) or len(text) == len(shown):
                    continue
                yield text[len(shown):], None, len(ids)
                shown = text
        text = visible(ids)
        yield (text[len(shown):] if text.startswith(shown) else ""), reason, len(ids)

    # ---- 对话

    def _chat_prompt(self, body: dict) -> tuple[list[int], int]:
        prompt = self.fmt.prompt(to_messages(body.get("messages") or [], body.get("tools")))
        return prompt, self._room(prompt, body)

    def complete(self, body: dict) -> dict:
        prompt, max_new = self._chat_prompt(body)
        generated = self._whole(prompt, max_new, body, self.fmt.end)
        finished = self.fmt.end in generated
        text, calls, _ = self.fmt.parse_turn(generated)
        message = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in calls]
        n_out = generated.index(self.fmt.end) + 1 if finished else len(generated)
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": self.name,
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else ("stop" if finished else "length")}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": n_out, "total_tokens": len(prompt) + n_out}}

    def stream_chat(self, body: dict):
        """流式对话：先检查输入（太长就在这里报错），再返回逐段的生成器。工具调用不在流里推送。"""
        prompt, max_new = self._chat_prompt(body)
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"

        def chunks():
            for delta, reason, n in self._stream(prompt, max_new, body, self.fmt.end, lambda ids: self.fmt.parse_turn(ids)[0]):
                chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": self.name,
                         "choices": [{"index": 0, "delta": {"content": delta} if delta else {}, "finish_reason": reason}]}
                if reason:
                    chunk["usage"] = {"prompt_tokens": len(prompt), "completion_tokens": n, "total_tokens": len(prompt) + n}
                yield chunk
        return chunks()

    # ---- 续写

    def _text_prompt(self, body: dict) -> tuple[list[int], int]:
        text = body.get("prompt")
        if isinstance(text, list):
            text = text[0] if text else ""
        if not isinstance(text, str):
            raise ValueError("prompt 要是一段文字")
        prompt = prompt_ids(self.tok, text)  # 和训练时一样：开头放文档结束符，表示这是新一篇的开头
        return prompt, self._room(prompt, body)

    def _visible_text(self, ids: list[int]) -> str:
        eos = self.tok.eos_id
        return self.tok.decode(ids[:ids.index(eos)] if eos in ids else ids)

    def complete_text(self, body: dict) -> dict:
        prompt, max_new = self._text_prompt(body)
        generated = self._whole(prompt, max_new, body, self.tok.eos_id)
        finished = self.tok.eos_id in generated
        n_out = generated.index(self.tok.eos_id) + 1 if finished else len(generated)
        return {"id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion", "created": int(time.time()), "model": self.name,
                "choices": [{"index": 0, "text": self._visible_text(generated), "finish_reason": "stop" if finished else "length"}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": n_out, "total_tokens": len(prompt) + n_out}}

    def stream_text(self, body: dict):
        prompt, max_new = self._text_prompt(body)
        cmpl_id = f"cmpl-{uuid.uuid4().hex}"

        def chunks():
            for delta, reason, n in self._stream(prompt, max_new, body, self.tok.eos_id, self._visible_text):
                chunk = {"id": cmpl_id, "object": "text_completion", "created": int(time.time()), "model": self.name,
                         "choices": [{"index": 0, "text": delta, "finish_reason": reason}]}
                if reason:
                    chunk["usage"] = {"prompt_tokens": len(prompt), "completion_tokens": n, "total_tokens": len(prompt) + n}
                yield chunk
        return chunks()


def make_server(service: Service, host: str = "127.0.0.1", port: int = 8001) -> ThreadingHTTPServer:
    routes = {"/v1/chat/completions": (service.complete, service.stream_chat), "/v1/completions": (service.complete_text, service.stream_text)}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload, content_type="application/json; charset=utf-8"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _events(self, chunks):
            """按 server-sent events 一段段推送；读的一方断开时停下，生成器关掉，锁随之放开。"""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for chunk in chunks:
                    self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                chunks.close()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path in ("", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": service.name, "object": "model"}]})
            elif path == "/v1/info":
                self._send(200, service.info)
            else:
                self._send(404, {"error": {"message": "没有这个接口"}})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            if path not in routes:
                return self._send(404, {"error": {"message": "没有这个接口"}})
            whole, streamed = routes[path]
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if body.get("stream"):
                    return self._events(streamed(body))
                self._send(200, whole(body))
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": {"message": str(e)}})

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def load_for_serving(ckpt: str, device: torch.device) -> tuple[LanguageModel, dict]:
    """读存档，返回模型和网页上要显示的信息：阶段（存档所在的目录名）、步数、参数量。"""
    state = load_checkpoint(Path(ckpt), torch.device("cpu"), mmap=True)
    model = LanguageModel(ModelConfig.from_json(state["model_config"]))
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    stage = state.get('artifact_metadata', {}).get('stage', Path(ckpt).parent.name)
    report = model.param_report()
    info = dict(ckpt=str(ckpt), stage=stage, step=state.get("step"), params=report["合计"], trunk=report["主干"],
                kind="base" if stage in BASE_STAGES or stage.startswith("cooldown") else "chat")
    return model, info


def main(argv=None):
    parser = argparse.ArgumentParser(description="OpenAI 兼容的服务接口和聊天网页")
    parser.add_argument("ckpt", nargs="?", default=None, help="存档；不写就用运行目录里走得最远的那个（rl、mopd、sft、退火、预训练依次找）")
    parser.add_argument("--run", default="out/small", help="运行目录")
    parser.add_argument("--host", default="127.0.0.1", help="只在本机用就别改；改成 0.0.0.0 别的机器也能连")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", default=None, help="接口里的模型名，默认是运行目录名加阶段，比如 medium/pretrain")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    ckpt = args.ckpt or str(find_stage(args.run, ("rl", "mopd", "sft", "anneal", "pretrain")))
    model, info = load_for_serving(ckpt, pick_device(args.device))
    # 发布包（下载下来的目录）用目录名；训练存档用「运行目录/阶段」，比如 medium/sft
    name = args.name or (Path(ckpt).resolve().name if Path(ckpt).is_dir() else f"{Path(ckpt).parent.parent.name}/{info['stage']}")
    from .artifact import tokenizer_path
    server = make_server(Service(model, ChatFormat(Tokenizer(tokenizer_path(ckpt, DEFAULT_MODEL))), name, info), args.host, args.port)
    print(f"存档 {ckpt}（{info['stage']}，第 {info['step']} 步）。浏览器打开 http://{args.host}:{args.port}/ 聊天；"
          f"接口在 http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
