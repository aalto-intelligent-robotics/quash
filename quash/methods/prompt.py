from typing import Tuple, Union, List
from pathlib import Path

def load_file_with_backup(path: str, backup: Union[str, List[str]]) -> str:
    if Path(path).exists():
        with open(path, "r") as f:
            return load_file(path)
    else:
        if isinstance(backup, str):
            backups = [backup]
        else:
            backups = backup
        for backup in backups:
            if Path(backup).exists():
                return load_file(backup)
        raise ValueError(f"Invalid prompt path (file): {path}, {backup}")

def load_file(path: str) -> str:
    with open(path, "r") as f:
        lines = f.readlines()
        output = " ".join(lines)
        return output

def load_prompts(path: str = "default") -> Tuple[str, str, str, str]:
    base_path = Path("~/<path/to/quash>/methods/config/prompts/").expanduser()
    if not Path(base_path).exists():
        raise ValueError(f"Invalid prompt path (base): {base_path}")
    synonym_prompt = load_file_with_backup(f"{base_path}/{path}/synonym_prompt.cfg", f"{base_path}/default/synonym_prompt.cfg")
    antonym_prompt = load_file_with_backup(f"{base_path}/{path}/antonym_prompt.cfg", f"{base_path}/default/antonym_prompt.cfg")
    synonym_system_prompt = load_file_with_backup(f"{base_path}/{path}/synonym_system_prompt.cfg", [f"{base_path}/{path}/system_prompt.cfg", f"{base_path}/default/system_prompt.cfg"])
    antonym_system_prompt = load_file_with_backup(f"{base_path}/{path}/antonym_system_prompt.cfg", [f"{base_path}/{path}/system_prompt.cfg", f"{base_path}/default/system_prompt.cfg"])
    return synonym_prompt, antonym_prompt, synonym_system_prompt, antonym_system_prompt
