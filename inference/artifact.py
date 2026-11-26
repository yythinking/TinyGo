"""Portable TinyBrain weights: safetensors plus verified JSON/tokenizer files."""
import hashlib
import json
from pathlib import Path

import torch


FORMAT = 'tinybrain.safetensors.v1'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def artifact_root(path):
    path = Path(path)
    return path if path.is_dir() else path.parent


def read_artifact(path, device='cpu', verify=True):
    from safetensors.torch import load_file

    root = artifact_root(path).resolve()
    manifest = json.loads((root/'artifact.json').read_text(encoding='utf-8'))
    if manifest.get('format') != FORMAT:
        raise ValueError('Unsupported TinyBrain artifact format')
    required = {'model.safetensors', 'config.json', 'tokenizer.model', 'generation_config.json'}
    files = manifest.get('files', {})
    if not required.issubset(files):
        raise ValueError('Incomplete artifact manifest')
    for name, expected in files.items():
        file = (root/name).resolve()
        if Path(name).name != name or file.parent != root:
            raise ValueError('Artifact file must be inside the bundle')
        if file.stat().st_size != expected['bytes']:
            raise ValueError(f'Artifact size mismatch: {name}')
        if verify and sha256(file) != expected['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    return dict(model=load_file(str(root/'model.safetensors'), device=str(device)),
                model_config=json.dumps(config, ensure_ascii=False), step=manifest.get('step', 0),
                training_progress=manifest.get('training_progress', {}),
                artifact_metadata=manifest, artifact_directory=str(root))


def tokenizer_path(checkpoint, fallback):
    root = artifact_root(checkpoint)
    return root/'tokenizer.model' if (root/'artifact.json').exists() else Path(fallback)


def export_checkpoint(source, destination, tokenizer, stage, verify=True, runtime_wheel=None):
    """Preserve every tensor's dtype; publish only after exact round-trip checks."""
    import os
    import shutil
    import uuid
    from safetensors.torch import save_file
    from lm import LanguageModel, ModelConfig

    source, destination, tokenizer = Path(source), Path(destination), Path(tokenizer)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite artifact: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name+'.partial-'+uuid.uuid4().hex[:10])
    pending.mkdir()
    # Training checkpoints are trusted inputs. Released inference bundles never unpickle.
    state = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    cfg = ModelConfig.from_json(state['model_config'])
    tensors, storage = {}, set()
    for name, value in state['model'].items():
        tensor = value.detach().cpu().contiguous()
        for part in tensor.reshape(-1).split(1024 * 1024):
            if part.is_floating_point() and not bool(torch.isfinite(part).all()):
                raise ValueError(f'Non-finite model weights: {name}')
        address = tensor.untyped_storage().data_ptr()
        if address in storage:
            tensor = tensor.clone()
        storage.add(tensor.untyped_storage().data_ptr())
        tensors[name] = tensor
    save_file(tensors, str(pending/'model.safetensors'), metadata={'format': 'pt', 'tinybrain_format': FORMAT})
    (pending/'config.json').write_text(cfg.to_json()+'\n', encoding='utf-8')
    shutil.copyfile(tokenizer, pending/'tokenizer.model')
    from .tokenizer import Tokenizer
    tok = Tokenizer(pending/'tokenizer.model')
    if tok.vocab_size != cfg.vocab_size:
        raise ValueError('Export tokenizer vocabulary differs from model')
    generation = dict(temperature=.8, top_k=50, max_new_tokens=256,
                      eos_token_id=tok.control_id('<|im_end|>') if stage not in ('pretrain','anneal') else tok.eos_id,
                      document_eos_token_id=tok.eos_id, pad_token_id=3)
    (pending/'generation_config.json').write_text(json.dumps(generation, indent=2)+'\n', encoding='utf-8')
    if runtime_wheel is not None:
        runtime_wheel=Path(runtime_wheel)
        if runtime_wheel.suffix!='.whl':raise ValueError('Runtime package must be a wheel')
        shutil.copyfile(runtime_wheel,pending/runtime_wheel.name)
        install=f'python -m pip install --force-reinstall --no-deps "{runtime_wheel.name}"\npython -m pip install torch sentencepiece numpy safetensors\n'
    else:
        install='Install the runtime from the matching TinyGo source checkout.\npython -m pip install ".[artifact]"\n'
    command='chat' if stage not in ('pretrain','anneal') else 'generate'
    (pending/'README.txt').write_text(
        'TinyGo inference bundle\n\n'+install+'\nRun from this bundle directory:\n'
        +f'tinygo {command} . --device auto --temperature 0.8 --top-k 50\n\n'
        +'The weights, architecture configuration and tokenizer are included. The TinyGo runtime implements this custom architecture.\n'
        +'artifact.json records checksums and export round-trip verification; generation quality is assessed separately.\n',encoding='utf-8')
    manifest = dict(format=FORMAT, architecture='TinyBrain LanguageModel', stage=stage,
                    step=int(state.get('step', 0)), training_progress=state.get('training_progress', {}),
                    source_checkpoint=f'{source.parent.name}/{source.name}', tensor_count=len(tensors), files={})
    for file in sorted(pending.iterdir()):
        manifest['files'][file.name] = dict(bytes=file.stat().st_size, sha256=sha256(file))
    (pending/'artifact.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if verify:
        restored = read_artifact(pending)
        if set(restored['model']) != set(tensors):
            raise ValueError('Export tensor keys changed')
        for name, tensor in tensors.items():
            if tensor.dtype != restored['model'][name].dtype or not torch.equal(tensor, restored['model'][name]):
                raise ValueError(f'Export changed tensor: {name}')
        model = LanguageModel(cfg).eval()
        model.load_state_dict(tensors, strict=True)
        ids = torch.tensor([[tok.eos_id]+tok.encode('你好，世界。')[:7]])
        with torch.inference_mode():
            before = model(ids).logits
            model.load_state_dict(restored['model'], strict=True)
            after = model(ids).logits
        if not torch.equal(before, after):
            raise ValueError('Export changed inference logits')
        manifest['verification'] = dict(all_tensors_exact=True, dtype_preserved=True,
                                         cpu_logits_max_abs_difference=float((before-after).abs().max()))
        (pending/'artifact.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(pending, destination)
    return manifest
