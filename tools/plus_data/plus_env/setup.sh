virtualenv --system-site-packages .venv2
set -e
source .venv2/bin/activate
script_dir=$(dirname "$0")
pip install -r $script_dir/requirements.txt
cp $script_dir/pip.conf .venv2/
