import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def uninstall(package):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])

# Desinstalar TensorFlow si está instalado
uninstall("tensorflow")

# Instalar la versión específica de TensorFlow 2.15
install("tensorflow==2.15")

# Instalar otros paquetes
# install("scikeras")
install("ipywidgets==7.5.0")
install("jupyter-ui-poll==0.1.2")