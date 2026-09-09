"""Build P1-03 LUMIA v5 by applying one explicit CUDA-placement repair to v4."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/build_lumia_hidden_state_runtime_pilot_notebook_v4.py"
source = BASE.read_text(encoding="utf-8")

old_model = '''model = AutoModelForCausalLM.from_pretrained(\n    CONFIG.model_id,\n    revision=CONFIG.model_revision,\n    device_map="cuda",\n    torch_dtype=torch.float16,\n    low_cpu_mem_usage=True,\n).eval()\nassert next(model.parameters()).device.type == "cuda"\n'''
new_model = '''# The real v4 Kaggle runtime returned a model whose first parameter was\n# still on CPU after the Transformers device-map CUDA shorthand. Reuse the\n# explicit transfer path already proven by this repository's successful\n# StarCoder2-3B 10k cache Notebook.\nmodel = AutoModelForCausalLM.from_pretrained(\n    CONFIG.model_id,\n    revision=CONFIG.model_revision,\n    torch_dtype=torch.float16,\n    low_cpu_mem_usage=True,\n)\nmodel = model.to("cuda:0").eval()\nparameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})\nbuffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})\nassert parameter_devices == ["cuda:0"], parameter_devices\nassert not buffer_devices or buffer_devices == ["cuda:0"], buffer_devices\nassert next(model.parameters()).dtype == torch.float16\nprint(json.dumps({\n    "model_parameter_devices": parameter_devices,\n    "model_buffer_devices": buffer_devices,\n    "model_device": str(next(model.parameters()).device),\n    "gpu0": gpu_names[0],\n    "explicit_cuda_transfer": True,\n}, sort_keys=True))\n'''
if source.count(old_model) != 1:
    raise RuntimeError("v4 model-loading block identity changed")
source = source.replace(old_model, new_model)

replacements = (
    ("poisoned-chalice-lumia-hs-runtime-50-v4", "poisoned-chalice-lumia-hs-runtime-50-v5"),
    ("Poisoned Chalice Lumia Hs Runtime 50 V4", "Poisoned Chalice Lumia Hs Runtime 50 V5"),
    ("lumia-runtime-v4", "lumia-runtime-v5"),
    ("LUMIA_RUNTIME_V4", "LUMIA_RUNTIME_V5"),
    ("LUMIA_HIDDEN_STATE_RUNTIME_PILOT_V4", "LUMIA_HIDDEN_STATE_RUNTIME_PILOT_V5"),
    ("runtime/fidelity gate v4", "runtime/fidelity gate v5"),
    ('"operational_attempt": "v4_target_site_repair"', '"operational_attempt": "v5_explicit_cuda_transfer_repair"'),
    ("Operational repair of the failed v3 ensurepip/venv bootstrap. Scientific inputs, ", "Operational repair of the v4 model-placement assertion. Scientific inputs, "),
)
for old, new in replacements:
    if old not in source:
        raise RuntimeError(f"v4 identity marker missing before v5 patch: {old}")
    source = source.replace(old, new)

runtime_tail = '"target_site_packages": True,\n}'
if source.count(runtime_tail) != 2:
    raise RuntimeError("v4 runtime/manifest tail count changed")
source = source.replace(
    runtime_tail,
    '"target_site_packages": True,\n    "explicit_cuda_transfer": True,\n    "model_parameter_devices": parameter_devices,\n    "model_buffer_devices": buffer_devices,\n}',
    1,
)
source = source.replace(
    runtime_tail,
    '"target_site_packages": True,\n    "explicit_cuda_transfer": True,\n}',
    1,
)
for marker in (
    'TARGET = "renta0426/poisoned-chalice-lumia-hs-runtime-50-v5"',
    'model = model.to("cuda:0").eval()',
    'parameter_devices == ["cuda:0"]',
    '"operational_attempt": "v5_explicit_cuda_transfer_repair"',
    "LUMIA_HIDDEN_STATE_RUNTIME_PILOT_V5 COMPLETE",
    "nbformat.v4.new_notebook()",
):
    if marker not in source:
        raise RuntimeError(f"v5 patched builder marker missing: {marker}")
if 'device_map="cuda"' in source:
    raise RuntimeError("v4 device_map path survived v5 patch")
if "nbformat.v5" in source:
    raise RuntimeError("nbformat API version mutated during v5 patch")
code = compile(source, str(Path(__file__).resolve()), "exec")
exec(code, {"__file__": str(Path(__file__).resolve()), "__name__": "__main__"})
