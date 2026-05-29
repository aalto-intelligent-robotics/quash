from typing import List, Tuple, Optional, Any
import tkinter
import tkinter.filedialog

class Dialog():
    def __init__(self):
        pass

    def __del__(self):
        pass

    def __enter__(self):
        self.root = tkinter.Tk()
        self.root.withdraw()

    def __exit__(self, exc_type, exc_value, traceback):
        self.root.destroy()

def validate_path(input: Any) -> str:
    if isinstance(input, str):
        return input
    else:
        exit()

def askdirectory(title: str, initialdir: str) -> str:
    with Dialog():
        path = tkinter.filedialog.askdirectory(title=title, initialdir=initialdir)
    return validate_path(path)

def askopenfilename(title: str, initialdir: str, filetypes: Optional[List[Tuple[str, str]]] = None) -> str:
    if filetypes is None:
        filetypes = [("All files", "*")]
    with Dialog():
        path = tkinter.filedialog.askopenfilename(title=title, initialdir=initialdir, filetypes=filetypes)
    return validate_path(path)

def asksaveasfilename(title: str, initialdir: str, filetypes: Optional[List[Tuple[str, str]]] = None) -> str:
    if filetypes is None:
        filetypes = [("All files", "*")]
    with Dialog():
        path = tkinter.filedialog.asksaveasfilename(title=title, initialdir=initialdir)
    return validate_path(path)
