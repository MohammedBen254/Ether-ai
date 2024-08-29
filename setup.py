from cx_Freeze import setup, Executable
import sys

base = None
if sys.platform == "win32":
    base = "Win32GUI"

target = Executable(
    script="main.py",
    base=base 
)

setup(
    name="Ether-ai",
    version="1.0",
    description="Ether-ai chatbot de Eco-thermique",
    executables=[target]
)