"""Optional HTTP server:  pip install "vjev[server]"  then  python -m vjev.server --preset balanced

POST /v1/decide   body = request JSON (images as base64 strings in "assets")
GET  /health
"""
import argparse


def make_app(decider):
    from fastapi import FastAPI, HTTPException

    from .schema import RequestError

    app = FastAPI(title="vjev", version="0.1.0")

    @app.get("/health")
    def health():
        return {"ok": True, "preset": decider.preset}

    @app.post("/v1/decide")
    def decide(body: dict):
        try:
            return decider.decide(body)
        except RequestError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--preset", default="balanced", choices=["fast", "balanced", "accurate"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn

    from .backend import QwenVLBackend
    from .decider import Decider
    uvicorn.run(make_app(Decider(QwenVLBackend(args.model), preset=args.preset)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
