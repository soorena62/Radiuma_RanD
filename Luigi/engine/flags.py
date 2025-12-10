import os

def flags_dir(workspace: str) -> str:
    d = os.path.join(workspace, "_flags")
    os.makedirs(d, exist_ok=True)
    return d

def flag_path(workspace: str, name: str) -> str:
    return os.path.join(flags_dir(workspace), name)

def is_set(workspace: str, name: str) -> bool:
    return os.path.exists(flag_path(workspace, name))

def set_flag(workspace: str, name: str) -> None:
    open(flag_path(workspace, name), "a").close()

def clear_flag(workspace: str, name: str) -> None:
    p = flag_path(workspace, name)
    if os.path.exists(p):
        os.remove(p)
