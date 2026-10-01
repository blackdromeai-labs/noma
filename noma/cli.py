"""Command line: `noma serve`."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(prog="noma", description="Noma decision model by Blackdrome AI Labs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve /v1/systemone and the playground")
    s.add_argument("--model", default="BlackdromeAILabs/noma",
                   help="Hugging Face repo id or local folder of an exported Noma")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    s.add_argument("--no-fast-path", action="store_true", help="disable length buckets and CUDA graphs")
    args = ap.parse_args()
    if args.cmd == "serve":
        os.environ["NOMA_MODEL"] = args.model
        if args.device:
            os.environ["NOMA_DEVICE"] = args.device
        if args.no_fast_path:
            os.environ["NOMA_BUCKET"] = "0"
        import uvicorn
        print(f"Noma: playground at http://{args.host}:{args.port}/  ·  API at /v1/systemone")
        uvicorn.run("noma.serve.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
