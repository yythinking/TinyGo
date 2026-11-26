"""Export a trusted training checkpoint to a self-contained safetensors bundle."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tokenizer', default='tokenizer/tokenizer.model')
    parser.add_argument('--stage', choices=['pretrain','anneal','sft','mopd','rl'], required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--runtime-wheel', help='Include the matching TinyGo runtime wheel in the release bundle')
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    from inference.artifact import export_checkpoint
    report = export_checkpoint(args.checkpoint, args.out, args.tokenizer, args.stage, runtime_wheel=args.runtime_wheel)
    print(json.dumps(dict(output=args.out,stage=report['stage'],step=report['step'],verification=report['verification'])))


if __name__ == '__main__':
    main()
