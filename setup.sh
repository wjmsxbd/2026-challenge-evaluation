#!/bin/bash
set -e

# Parse arguments
HELP=false
NEW_ENV=false
NEW_ENV_NAME="behavior_2026"
OMNIGIBSON=false
BDDL=false
JOYLO=false
DATASET=false
PRIMITIVES=false
EVAL=false
CHALLENGE_EVAL=false
ASSET_PIPELINE=false
DEV=false
CUDA_VERSION="12.6"
ACCEPT_CONDA_TOS=false
ACCEPT_NVIDIA_EULA=false
ACCEPT_DATASET_TOS=false
CONFIRM_NO_CONDA=false

[ "$#" -eq 0 ] && HELP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) HELP=true; shift ;;
        --new-env)
            NEW_ENV=true
            # support: --new-env NAME or just --new-env (use default)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                NEW_ENV_NAME="$2"
                shift 2
            else
                NEW_ENV_NAME="behavior_2026"
                shift 1
            fi
            ;;
        --omnigibson) OMNIGIBSON=true; shift ;;
        --bddl) BDDL=true; shift ;;
        --joylo) JOYLO=true; shift ;;
        --dataset) DATASET=true; shift ;;
        --primitives) PRIMITIVES=true; shift ;;
        --eval) EVAL=true; shift ;;
        --challenge-eval)
            CHALLENGE_EVAL=true
            OMNIGIBSON=true
            BDDL=true
            JOYLO=true
            EVAL=true
            DATASET=true
            shift
            ;;
        --asset-pipeline) ASSET_PIPELINE=true; shift ;;
        --dev) DEV=true; shift ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --accept-conda-tos) ACCEPT_CONDA_TOS=true; shift ;;
        --accept-nvidia-eula) ACCEPT_NVIDIA_EULA=true; shift ;;
        --accept-dataset-tos) ACCEPT_DATASET_TOS=true; shift ;;
        --confirm-no-conda) CONFIRM_NO_CONDA=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate CUDA_VERSION is a valid numeric version string
if ! [[ "$CUDA_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Invalid CUDA_VERSION '$CUDA_VERSION'. Must be in format X.Y (e.g., 12.4)"
    exit 1
fi

if [ "$HELP" = true ]; then
    cat << EOF
BEHAVIOR-1K Installation Script (Linux)
Usage: ./setup.sh [OPTIONS]

Options:
  -h, --help              Display this help message
  --new-env NEW_ENV_NAME  Create a new conda environment 'NEW_ENV_NAME' (default: behavior_2026)
  --omnigibson            Install OmniGibson (core physics simulator)
  --bddl                  Install BDDL (Behavior Domain Definition Language)
  --joylo                 Install JoyLo (teleoperation interface)
  --dataset               Download BEHAVIOR datasets (requires --omnigibson)
  --primitives            Install OmniGibson with primitives support
  --eval                  Install evaluation dependencies
  --challenge-eval        Install the complete 2026 challenge evaluation stack and datasets
  --asset-pipeline        Install the 3D scene and object asset pipeline
  --dev                   Install development dependencies
  --cuda-version VERSION  Specify CUDA version (default: 12.4)
  --accept-conda-tos      Automatically accept Conda Terms of Service
  --accept-nvidia-eula    Automatically accept NVIDIA Isaac Sim EULA
  --accept-dataset-tos    Automatically accept BEHAVIOR Dataset Terms
  --confirm-no-conda      Skip confirmation prompt when not in a conda environment

Example (2026 challenge evaluation): ./setup.sh --new-env --challenge-eval
Example (core components): ./setup.sh --new-env --omnigibson --bddl --dataset
Example (full customization): ./setup.sh --new-env my_env --omnigibson --bddl --dataset --joylo --eval --primitives --cuda-version 12.6
Example (non-interactive): ./setup.sh --new-env --challenge-eval --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
EOF
    exit 0
fi

# Validate dependencies
# [ "$OMNIGIBSON" = true ] && [ "$BDDL" = false ] && { echo "ERROR: --omnigibson requires --bddl"; exit 1; }
# [ "$DATASET" = true ] && [ "$OMNIGIBSON" = false ] && { echo "ERROR: --dataset requires --omnigibson"; exit 1; }
# [ "$PRIMITIVES" = true ] && [ "$OMNIGIBSON" = false ] && { echo "ERROR: --primitives requires --omnigibson"; exit 1; }
# [ "$EVAL" = true ] && [ "$OMNIGIBSON" = false ] && { echo "ERROR: --eval requires --omnigibson"; exit 1; }
# [ "$EVAL" = true ] && [ "$JOYLO" = false ] && { echo "ERROR: --eval requires --joylo"; exit 1; }
# [ "$NEW_ENV" = true ] && [ "$CONFIRM_NO_CONDA" = true ] && { echo "ERROR: --new-env and --confirm-no-conda are mutually exclusive"; exit 1; }

WORKDIR=$(pwd)
ARCH=$(uname -m)
CONDA_BIN="$(command -v conda || true)"
if [ -z "$CONDA_BIN" ] && [ -x "$HOME/miniconda3/condabin/conda" ]; then
    CONDA_BIN="$HOME/miniconda3/condabin/conda"
elif [ -z "$CONDA_BIN" ] && [ -x "$HOME/anaconda3/condabin/conda" ]; then
    CONDA_BIN="$HOME/anaconda3/condabin/conda"
fi

# The challenge preset always targets the dedicated behavior_2026 environment. If it already
# exists, reuse it; pass --new-env to create it on the first installation.
if [ "$CHALLENGE_EVAL" = true ] && [ "$NEW_ENV" = false ]; then
    [ -n "$CONDA_BIN" ] || { echo "ERROR: Conda not found"; exit 1; }
    source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"
    if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$NEW_ENV_NAME"; then
        echo "ERROR: Conda environment '$NEW_ENV_NAME' does not exist."
        echo "Create it with: ./setup.sh --new-env --challenge-eval"
        exit 1
    fi
    echo "Activating existing conda environment '$NEW_ENV_NAME'..."
    conda activate "$NEW_ENV_NAME"
    [[ "$CONDA_DEFAULT_ENV" != "$NEW_ENV_NAME" ]] && {
        echo "ERROR: Failed to activate environment '$NEW_ENV_NAME'"
        exit 1
    }
fi

# Check conda environment condition early (unless creating new environment)
if [ "$NEW_ENV" = false ]; then
    if [ -z "$CONDA_PREFIX" ]; then
        if [ "$CONFIRM_NO_CONDA" = false ]; then
            echo ""
            echo "WARNING: You are not in a conda environment."
            echo "Currently using Python from: $(which python)"
            echo ""
            echo "Continue? [y/n] (or rerun with --confirm-no-conda to skip this prompt)"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo "Installation cancelled."
                exit 1
            fi
        fi
        echo "Proceeding without conda environment..."
    fi
fi

# Function to prompt for terms acceptance
prompt_for_terms() {
    echo ""
    echo "=== TERMS OF SERVICE AND LICENSING AGREEMENTS ==="
    echo ""
    
    # Check what terms need to be accepted
    NEEDS_CONDA_TOS=false
    NEEDS_NVIDIA_EULA=false
    NEEDS_DATASET_TOS=false
    
    if [ "$NEW_ENV" = true ] && [ "$ACCEPT_CONDA_TOS" = false ]; then
        NEEDS_CONDA_TOS=true
    fi
    
    if [ "$OMNIGIBSON" = true ] && [ "$ACCEPT_NVIDIA_EULA" = false ]; then
        NEEDS_NVIDIA_EULA=true
    fi
    
    if [ "$DATASET" = true ] && [ "$ACCEPT_DATASET_TOS" = false ]; then
        NEEDS_DATASET_TOS=true
    fi
    
    # If nothing needs acceptance, return early
    if [ "$NEEDS_CONDA_TOS" = false ] && [ "$NEEDS_NVIDIA_EULA" = false ] && [ "$NEEDS_DATASET_TOS" = false ]; then
        return 0
    fi
    
    echo "This installation requires acceptance of the following terms:"
    echo ""
    
    if [ "$NEEDS_CONDA_TOS" = true ]; then
        cat << EOF
1. CONDA TERMS OF SERVICE
   - Required for creating conda environment
   - By accepting, you agree to Anaconda's Terms of Service
   - See: https://legal.anaconda.com/policies/en/

EOF
    fi
    
    if [ "$NEEDS_NVIDIA_EULA" = true ]; then
        cat << EOF
2. NVIDIA ISAAC SIM EULA
   - Required for OmniGibson installation
   - By accepting, you agree to NVIDIA Isaac Sim End User License Agreement
   - See: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement

EOF
    fi
    
    if [ "$NEEDS_DATASET_TOS" = true ]; then
        cat << EOF
3. BEHAVIOR DATA BUNDLE END USER LICENSE AGREEMENT
    Last revision: December 8, 2022
    This License Agreement is for the BEHAVIOR Data Bundle (“Data”). It works with OmniGibson (“Software”) which is a software stack licensed under the MIT License, provided in this repository: https://github.com/StanfordVL/BEHAVIOR-1K.
    The license agreements for OmniGibson and the Data are independent. This BEHAVIOR Data Bundle contains artwork and images (“Third Party Content”) from third parties with restrictions on redistribution. 
    It requires measures to protect the Third Party Content which we have taken such as encryption and the inclusion of restrictions on any reverse engineering and use. 
    Recipient is granted the right to use the Data under the following terms and conditions of this License Agreement (“Agreement”):
        1. Use of the Data is permitted after responding "Yes" to this agreement. A decryption key will be installed automatically.
        2. Data may only be used for non-commercial academic research. You may not use a Data for any other purpose.
        3. The Data has been encrypted. You are strictly prohibited from extracting any Data from OmniGibson or reverse engineering.
        4. You may only use the Data within OmniGibson.
        5. You may not redistribute the key or any other Data or elements in whole or part.
        6. THE DATA AND SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
            IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATA OR SOFTWARE OR THE USE OR OTHER DEALINGS IN THE DATA OR SOFTWARE.

EOF
    fi
    
    echo "Do you accept ALL of the above terms? (y/N)"
    read -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Terms not accepted. Installation cancelled."
        echo "You can bypass these prompts by using --accept-conda-tos, --accept-nvidia-eula, and --accept-dataset-tos flags."
        exit 1
    fi
    
    # Set acceptance flags
    [ "$NEEDS_CONDA_TOS" = true ] && ACCEPT_CONDA_TOS=true
    [ "$NEEDS_NVIDIA_EULA" = true ] && ACCEPT_NVIDIA_EULA=true
    [ "$NEEDS_DATASET_TOS" = true ] && ACCEPT_DATASET_TOS=true
    
    echo ""
    echo "✓ All terms accepted. Proceeding with installation..."
    echo ""
}

# Prompt for terms acceptance at the beginning
prompt_for_terms

# If primitives requested, ensure a matching system CUDA is available
if [ "$PRIMITIVES" = true ]; then
    NVCC_OK=false
    if command -v nvcc >/dev/null 2>&1; then
        if nvcc -V 2>&1 | grep -q "$CUDA_VERSION"; then
            NVCC_OK=true
        fi
    fi

    if [ "$NVCC_OK" = false ]; then
        echo ""
        echo "ERROR: Primitives support requires CUDA Toolkit $CUDA_VERSION and a matching 'nvcc' in PATH."
        echo "Please install the correct CUDA toolkit system-wide (for example /usr/local/cuda-$CUDA_VERSION) or ensure your PATH provides an nvcc that reports $CUDA_VERSION, then re-run this script."
        exit 1
    fi
fi

# Create conda environment
if [ "$NEW_ENV" = true ]; then
    echo "Creating conda environment '$NEW_ENV_NAME'..."
    [ -n "$CONDA_BIN" ] || { echo "ERROR: Conda not found"; exit 1; }
    
    # Set auto-accept environment variable if user agreed to TOS
    if [ "$ACCEPT_CONDA_TOS" = true ]; then
        export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
        echo "✓ Conda TOS auto-acceptance enabled"
    fi
    
    source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"

    # Check if environment already exists and exit with instructions
    if "$CONDA_BIN" env list | grep -q "^$NEW_ENV_NAME "; then
        echo ""
        echo "ERROR: Conda environment '$NEW_ENV_NAME' already exists!"
        echo ""
        echo "Please remove or rename the existing environment and re-run this script."
        echo ""
        exit 1
    fi
    
    # Create environment with Python 3.11, packaging tools, and Git LFS for LeRobot dependencies.
    "$CONDA_BIN" create -n "$NEW_ENV_NAME" python=3.11 pip "setuptools>=71,<81" wheel git-lfs -c conda-forge -y
    conda activate "$NEW_ENV_NAME"
    
    [[ "$CONDA_DEFAULT_ENV" != "$NEW_ENV_NAME" ]] && { echo "ERROR: Failed to activate environment '$NEW_ENV_NAME'"; exit 1; }

fi

# Always use the selected conda environment's interpreter, even if the caller has another
# virtual environment (for example pi05_env) active earlier in PATH.
if [ "$NEW_ENV" = true ] || [ "$CHALLENGE_EVAL" = true ]; then
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Ignoring previously active virtual environment: $VIRTUAL_ENV"
    fi
    unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
    export PYTHONNOUSERSITE=1
    export PATH="$CONDA_PREFIX/bin:$PATH"
    hash -r
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python interpreter not found or not executable: ${PYTHON_BIN:-<unset>}"
    exit 1
fi
run_python() {
    "$PYTHON_BIN" "$@"
}
echo "Using Python interpreter: $PYTHON_BIN"
echo "Python version: $(run_python --version 2>&1)"

verify_isaac_sim_installation() {
    run_python - <<'PY'
import os
from importlib.metadata import PackageNotFoundError, version

required_distributions = (
    "isaacsim",
    "isaacsim-app",
    "isaacsim-core",
    "isaacsim-kernel",
)
missing = []
for distribution in required_distributions:
    try:
        version(distribution)
    except PackageNotFoundError:
        missing.append(distribution)

if missing:
    raise SystemExit(f"Missing Isaac Sim distributions: {', '.join(missing)}")

import isaacsim  # noqa: F401, E402

isaac_path = os.environ.get("ISAAC_PATH")
if not isaac_path:
    raise SystemExit("Importing isaacsim did not set ISAAC_PATH")

version_file = os.path.join(isaac_path, "VERSION")
if not os.path.isfile(version_file):
    raise SystemExit(f"Isaac Sim VERSION file not found: {version_file}")
PY
}

# LeRobot is installed from git and may reference files managed by Git LFS.
if [ "$OMNIGIBSON" = true ]; then
    GIT_LFS_BIN="$(command -v git-lfs || true)"
    if [ -z "$GIT_LFS_BIN" ] && [ "$CHALLENGE_EVAL" = true ]; then
        echo "Installing Git LFS into '$NEW_ENV_NAME'..."
        "$CONDA_BIN" install -p "$CONDA_PREFIX" git-lfs -c conda-forge -y
        hash -r
        GIT_LFS_BIN="$CONDA_PREFIX/bin/git-lfs"
    fi
    if [ -z "$GIT_LFS_BIN" ] || [ ! -x "$GIT_LFS_BIN" ]; then
        echo "ERROR: git-lfs is required to install LeRobot. Install it in the active conda environment first."
        exit 1
    fi
    "$GIT_LFS_BIN" install --skip-repo
    echo "Git LFS: $("$GIT_LFS_BIN" version)"
fi

# Install PyTorch via pip with CUDA support
echo "Installing PyTorch with CUDA $CUDA_VERSION support..."

# Select a PyTorch release that has official wheels for the requested CUDA version.
CUDA_VER_SHORT=$(echo "$CUDA_VERSION" | sed 's/\.//g')  # e.g. convert 12.4 to 124
case "$CUDA_VERSION" in
    12.4)
        TORCH_VERSION="2.6.0"
        TORCHVISION_VERSION="0.21.0"
        TORCHAUDIO_VERSION="2.6.0"
        TORCHCODEC_VERSION="0.2"
        ;;
    12.6|12.8)
        TORCH_VERSION="2.7.0"
        TORCHVISION_VERSION="0.22.0"
        TORCHAUDIO_VERSION="2.7.0"
        TORCHCODEC_VERSION="0.5"
        ;;
    *)
        echo "ERROR: Unsupported CUDA version '$CUDA_VERSION'. Supported versions: 12.4, 12.6, 12.8"
        exit 1
        ;;
esac

run_python -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "https://download.pytorch.org/whl/cu${CUDA_VER_SHORT}"
run_python -m pip install "torchcodec==$TORCHCODEC_VERSION"

echo "✓ PyTorch installation completed"

# Install numpy <2 to avoid conflicts
echo "Installing numpy..."
run_python -m pip install "numpy<2"

# Install BDDL
if [ "$BDDL" = true ]; then
    echo "Installing BDDL..."
    [ ! -d "bddl3" ] && { echo "ERROR: bddl directory not found"; exit 1; }
    run_python -m pip install -e "$WORKDIR/bddl3"
fi

# Install OmniGibson with Isaac Sim
if [ "$OMNIGIBSON" = true ]; then
    echo "Installing OmniGibson..."
    [ ! -d "OmniGibson" ] && { echo "ERROR: OmniGibson directory not found"; exit 1; }
    
    # Check Python version
    PYTHON_VERSION=$(run_python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    [ "$PYTHON_VERSION" != "3.11" ] && { echo "ERROR: Python 3.11 required, found $PYTHON_VERSION"; exit 1; }
    
    # Check for conflicting environment variables
    if [[ -n "$EXP_PATH" || -n "$CARB_APP_PATH" || -n "$ISAAC_PATH" ]]; then
        echo "ERROR: Found existing Isaac Sim environment variables."
        echo "Please unset EXP_PATH, CARB_APP_PATH, and ISAAC_PATH and restart."
        exit 1
    fi

    # Build extras
    EXTRAS=""
    if [ "$DEV" = true ]; then
        EXTRAS="${EXTRAS}dev,"
    fi
    if [ "$PRIMITIVES" = true ]; then
        EXTRAS="${EXTRAS}primitives,"
    fi
    if [ "$EVAL" = true ]; then
        EXTRAS="${EXTRAS}eval,"
    fi
    # Remove trailing comma, if any, and add brackets only if EXTRAS is not empty
    if [ -n "$EXTRAS" ]; then
        EXTRAS="[${EXTRAS%,}]"
    fi

    run_python -m pip install -e "$WORKDIR/OmniGibson$EXTRAS" --no-build-isolation

    # Install pre-commit for dev setup
    if [ "$DEV" = true ]; then
        echo "Setting up pre-commit..."
        "$CONDA_BIN" install -c conda-forge pre-commit -y
        cd "$WORKDIR/OmniGibson"
        pre-commit install || true  # Ignore errors here in case the directory is not a git repo
        cd "$WORKDIR"
    fi
    
    # Isaac Sim installation via pip
    if [ "$ACCEPT_NVIDIA_EULA" = true ]; then
        export OMNI_KIT_ACCEPT_EULA=YES
    else
        echo "ERROR: NVIDIA EULA not accepted. Cannot install Isaac Sim."
        exit 1
    fi
    
    # A partial extscache install creates an importable isaacsim namespace package, so an
    # import alone is not enough to prove that the Isaac Sim runtime is complete.
    if verify_isaac_sim_installation >/dev/null 2>&1; then
        echo "Isaac Sim already installed, skipping..."
    else
        echo "Installing Isaac Sim via pip..."

        # For aarch, do alternative install via direct one-liner
        if [ "$ARCH" = "aarch64" ]; then
            run_python -m pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
        else
            # Helper functions
            check_glibc_old() {
                ldd --version 2>&1 | grep -qE "2\.(31|32|33|34)"
            }

            install_isaac_packages() {
                local temp_dir="${ISAACSIM_WHEEL_DIR:-/mnt/data/nas/B_zone_nas/jiaqi/b1k2026/isaacsim_wheels}"
                mkdir -p "$temp_dir"
                local packages=(
                    "omniverse_kit-107.3.1.206797"
                    "isaacsim_kernel-5.1.0.0"
                    "isaacsim_app-5.1.0.0"
                    "isaacsim_core-5.1.0.0"
                    "isaacsim_gui-5.1.0.0"
                    "isaacsim_utils-5.1.0.0"
                    "isaacsim_storage-5.1.0.0"
                    "isaacsim_asset-5.1.0.0"
                    "isaacsim_sensor-5.1.0.0"
                    "isaacsim_robot_motion-5.1.0.0"
                    "isaacsim_robot-5.1.0.0"
                    "isaacsim_benchmark-5.1.0.0"
                    "isaacsim_code_editor-5.1.0.0"
                    "isaacsim_ros1-5.1.0.0"
                    "isaacsim_cortex-5.1.0.0"
                    "isaacsim_example-5.1.0.0"
                    "isaacsim_replicator-5.1.0.0"
                    "isaacsim_rl-5.1.0.0"
                    "isaacsim_robot_setup-5.1.0.0"
                    "isaacsim_ros2-5.1.0.0"
                    "isaacsim_template-5.1.0.0"
                    "isaacsim_test-5.1.0.0"
                    "isaacsim-5.1.0.0"
                    "isaacsim_extscache_physics-5.1.0.0"
                    "isaacsim_extscache_kit-5.1.0.0"
                    "isaacsim_extscache_kit_sdk-5.1.0.0"
                )

                local wheel_files=()
                for pkg in "${packages[@]}"; do
                    local pkg_name=${pkg%-*}
                    local filename="${pkg}-cp311-none-manylinux_2_35_${ARCH}.whl"
                    local url="https://pypi.nvidia.com/${pkg_name//_/-}/$filename"
                    local filepath="$temp_dir/$filename"

                    echo "Downloading $pkg..."
                    if ! curl -4 -fL --retry 20 --retry-delay 5 --retry-all-errors --connect-timeout 30 -C - "$url" -o "$filepath"; then
                        echo "ERROR: Failed to download $pkg"
                        return 1
                    fi

                    # Rename for older GLIBC
                    if check_glibc_old; then
                        local new_filepath="${filepath/manylinux_2_35/manylinux_2_31}"
                        mv "$filepath" "$new_filepath"
                        filepath="$new_filepath"
                    fi

                    wheel_files+=("$filepath")
                done

                echo "Installing Isaac Sim packages..."
                run_python -m pip install "${wheel_files[@]}"

            }

            install_isaac_packages || { echo "ERROR: Isaac Sim installation failed"; exit 1; }
        fi

        verify_isaac_sim_installation || { echo "ERROR: Isaac Sim installation verification failed"; exit 1; }
        
        # Extract ISAAC_PATH from isaacsim module
        ISAAC_PATH=$(run_python -c "import isaacsim, os; print(os.environ.get('ISAAC_PATH', ''))" 2>/dev/null)

        # Fix websockets conflict - remove any pip_prebundle/websockets under extscache
        if [ -n "$ISAAC_PATH" ] && [ -d "$ISAAC_PATH/extscache" ]; then
            echo "Fixing websockets conflict..."
            find "$ISAAC_PATH/extscache" -type d -name "websockets" -path "*/pip_prebundle/*" -exec rm -rf {} + 2>/dev/null || true
        fi

        # Fix packaging conflict - remove conflicting version
        # There is a conflict where isaacsim enforces 23.0 but omni kit ships with 25.0
        if [ -d "$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.services.pip_archive-0.16.0+107.0.3.lx64.cp311/pip_prebundle/packaging" ]; then
            echo "Fixing packaging conflict..."
            rm -rf "$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.services.pip_archive-0.16.0+107.0.3.lx64.cp311/pip_prebundle/packaging"
        fi
    fi
    
    # Force reinstall cffi 1.17.1 to resolve compatibility issues with Isaac Sim extensions
    run_python -m pip install --force-reinstall cffi==1.17.1
    # Force reinstall websockets >= 15.0.1 because it's been overwritten by Isaac Sim with an older version
    run_python -m pip install --force-reinstall "websockets>=15.0.1"

    echo "OmniGibson installation completed successfully!"
fi

# Install JoyLo
if [ "$JOYLO" = true ]; then
    echo "Installing JoyLo..."
    [ ! -d "joylo" ] && { echo "ERROR: joylo directory not found"; exit 1; }
    run_python -m pip install -e "$WORKDIR/joylo"
fi

# Install Eval
if [ "$EVAL" = true ]; then
    # get torch version including local CUDA suffix and install corresponding torch-cluster
    INSTALLED_TORCH_VERSION=$(run_python -c "import torch; print(torch.__version__.split('+git')[0])")
    run_python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-${INSTALLED_TORCH_VERSION}.html
fi

# Install asset pipeline
if [ "$ASSET_PIPELINE" = true ]; then
    echo "Installing asset pipeline..."
    [ ! -d "asset_pipeline" ] && { echo "ERROR: asset_pipeline directory not found"; exit 1; }
    run_python -m pip install -r "$WORKDIR/asset_pipeline/requirements.txt"
fi

# Install datasets
if [ "$DATASET" = true ]; then
    run_python -c "import omnigibson" || {
        echo "ERROR: OmniGibson import failed, please make sure you have omnigibson installed before downloading datasets"
        exit 1
    }

    echo "Installing datasets..."

    # Determine if we should accept dataset license automatically
    DATASET_ACCEPT_FLAG=""
    if [ "$ACCEPT_DATASET_TOS" = true ]; then
        DATASET_ACCEPT_FLAG="True"
    else
        DATASET_ACCEPT_FLAG="False"
    fi
    
    export OMNI_KIT_ACCEPT_EULA=YES
    
    echo "Downloading OmniGibson robot assets..."
    run_python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()" || {
        echo "ERROR: OmniGibson robot assets installation failed"
        exit 1
    }

    echo "Downloading BEHAVIOR-1K assets..."
    run_python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=${DATASET_ACCEPT_FLAG})" || {
        echo "ERROR: Dataset installation failed"
        exit 1
    }

    echo "Downloading 2026 BEHAVIOR Challenge Task Instances..."
    run_python -c "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()" || {
        echo "ERROR: 2026 BEHAVIOR Challenge Task Instances installation failed"
        exit 1
    }
fi

# Verify the requested installation before reporting success.
echo "Running installation smoke checks..."
if [ "$BDDL" = true ]; then
    run_python -c "import bddl; print('✓ BDDL import check passed')"
fi
if [ "$OMNIGIBSON" = true ]; then
    verify_isaac_sim_installation
    run_python -c "import omnigibson; print('✓ Isaac Sim and OmniGibson import checks passed')"
fi
if [ "$JOYLO" = true ]; then
    run_python -c "import gello; print('✓ JoyLo import check passed')"
fi
if [ "$EVAL" = true ]; then
    OMNIGIBSON_HEADLESS=1 run_python -m omnigibson.eval.eval --help >/dev/null
    echo "✓ Evaluation CLI check passed"
fi
if [ "$DATASET" = true ]; then
    OMNIGIBSON_HEADLESS=1 run_python -c \
        "from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_INDICES; assert len(TASK_NAMES_TO_INDICES) == 100; print('✓ 2026 challenge metadata check passed')"
fi

echo ""
echo "=== Installation Complete! ==="
if [ "$NEW_ENV" = true ]; then echo "✓ Created conda environment '$NEW_ENV_NAME'"; fi
if [ "$OMNIGIBSON" = true ]; then echo "✓ Installed OmniGibson + Isaac Sim"; fi
if [ "$BDDL" = true ]; then echo "✓ Installed BDDL"; fi
if [ "$JOYLO" = true ]; then echo "✓ Installed JoyLo"; fi
if [ "$PRIMITIVES" = true ]; then echo "✓ Installed OmniGibson with primitives support"; fi
if [ "$EVAL" = true ]; then echo "✓ Installed evaluation support"; fi
if [ "$DATASET" = true ]; then echo "✓ Downloaded datasets"; fi
echo ""
if [ "$NEW_ENV" = true ]; then echo "To activate: conda activate '$NEW_ENV_NAME'"; fi
