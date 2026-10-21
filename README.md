# IKT466_Project


GPU users (NVIDIA): after pip install -r requirements.txt, reinstall torch with CUDA support:
pip uninstall -y torch torchvision
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132