from __future__ import annotations
 
from pathlib import Path
 
import yaml
 
 
def load_config(path: str | Path) -> dict:
    """加载 YAML 配置文件."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    return config
 
 
def save_config(config: dict, path: str | Path):
    """保存配置到 YAML 文件 (用于实验复现)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
 