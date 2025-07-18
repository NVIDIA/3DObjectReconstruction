#!/bin/bash

# 3D Object Reconstruction Docker Deployment Script
# Note: This script should be run from anywhere, but it will use the docker-compose.yml
# file located in the same directory as this script (deploy/compose/)

set -e

# Get the project root directory (two levels up from this script)
REPO_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")/../../" ; pwd -P )

# Get the compose directory (where this script and docker-compose.yml are located)
COMPOSE_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Docker image configuration
LOCAL_IMAGE="3d-object-reconstruction:${USER:-default}"
CONTAINER_NAME="3d-object-reconstruction-container-${USER:-default}"
NETWORK_NAME="3d-recon-network-${USER:-default}"

# Port configuration
PORT_START_INDEX=${PORT_START_INDEX:-8888}
JUPYTER_PORT=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to find an available port starting from PORT_START_INDEX
find_available_port() {
    local start_port=$1
    local port=$start_port
    
    while true; do
        # Check if port is available
        if ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
            # Double check with lsof if available
            if command -v lsof >/dev/null 2>&1; then
                if ! lsof -i :$port >/dev/null 2>&1; then
                    echo $port
                    return 0
                fi
            else
                echo $port
                return 0
            fi
        fi
        port=$((port + 1))
        
        # Prevent infinite loop - check up to 1000 ports
        if [ $((port - start_port)) -gt 1000 ]; then
            print_error "Could not find available port after checking 1000 ports from $start_port"
            exit 1
        fi
    done
}

# Function to get docker-compose command with appropriate files
get_compose_cmd() {
    # Check if docker-compose (v1) is available
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose -f docker-compose.yml"
    # Check if docker compose (v2) is available
    elif docker compose version &> /dev/null; then
        echo "docker compose -f docker-compose.yml"
    else
        echo "docker-compose -f docker-compose.yml"  # fallback
    fi
}

# Function to create required directories
create_directories() {
    print_info "Creating required directories..."
    
    directories=(
        "${REPO_DIR}/data/weights"
        "${REPO_DIR}/data/weights/roma"
        "${REPO_DIR}/data/weights/sam2"
        "${REPO_DIR}/data/weights/foundationstereo"
        "${REPO_DIR}/data/output"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_info "Directory already exists: $dir"
        fi
    done
}

# Function to check basic requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check for NVIDIA drivers
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi is not available. NVIDIA drivers are not installed."
        print_error "Please install NVIDIA drivers for GPU support."
        exit 1
    else
        print_success "NVIDIA drivers detected"
    fi

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    else
        if command -v docker-compose &> /dev/null; then
            print_success "Docker Compose v1 detected"
        elif docker compose version &> /dev/null; then
            print_success "Docker Compose v2 detected"
        fi
    fi
    
    # Check for NVIDIA Docker support
    if ! docker info | grep -q nvidia; then
        print_warning "NVIDIA Docker support not detected. GPU acceleration may not work."
        print_warning "Please install nvidia-docker2 for GPU support."
    else
        print_success "NVIDIA Docker support detected"
    fi
    
    print_success "System requirements met"
}

# Function to build Docker image locally
build_image() {
    print_info "Building Docker image locally..."
    print_info "This may take 30-60 minutes depending on your system..."
    print_warning "Ensure you have sufficient disk space (>50GB) and a stable internet connection"
    
    # Check if Dockerfile exists
    if [ ! -f "$REPO_DIR/docker/Dockerfile" ]; then
        print_error "Dockerfile not found at $REPO_DIR/docker/Dockerfile"
        exit 1
    fi
    
    # Set image name for local build
    export IMAGE_NAME="$LOCAL_IMAGE"
    
    # Build the image
    local compose_cmd=$(get_compose_cmd)
    cd "$COMPOSE_DIR"
    
    # Use cache by default, disable with NO_CACHE=1
    local cache_flag=""
    if [ "${NO_CACHE:-}" = "1" ]; then
        cache_flag="--no-cache"
        print_info "Cache disabled (NO_CACHE=1)"
    else
        print_info "Using Docker build cache (set NO_CACHE=1 to disable)"
    fi
    
    print_info "Running: $compose_cmd build $cache_flag reconstruction-app"
    if eval "$compose_cmd build $cache_flag reconstruction-app"; then
        print_success "Docker image built successfully!"
        print_info "Local image tagged as: $LOCAL_IMAGE"
    else
        print_error "Failed to build Docker image"
        print_error "Check the build logs above for details"
        exit 1
    fi
}

# Function to validate file checksums
validate_checksum() {
    local file_path="$1"
    local expected_checksum="$2"
    local file_name=$(basename "$file_path")
    
    if [ ! -f "$file_path" ]; then
        print_error "File not found for checksum validation: $file_path"
        return 1
    fi
    
    print_info "Validating checksum for $file_name..."
    local actual_checksum=$(md5sum "$file_path" | cut -d' ' -f1)
    
    if [ "$actual_checksum" = "$expected_checksum" ]; then
        print_success "Checksum validation passed for $file_name"
        return 0
    else
        print_error "Checksum validation failed for $file_name"
        print_error "Expected: $expected_checksum"
        print_error "Actual:   $actual_checksum"
        return 1
    fi
}

# Function to download weights
download_weights() {
    print_info "Downloading weights..."
    
    # Create weight directories first
    create_directories
    
    # Track download failures
    local download_failures=0
    local checksum_failures=0
    
    # Define expected checksums for all files
    declare -A expected_checksums=(
        ["roma/dinov2_vitl14_pretrain.pth"]="19a02c10947ed50096ce382b46b15662"
        ["roma/roma_outdoor.pth"]="9a451dfb65745e777bf916db6ea84933"
        ["sam2/sam2.1_hiera_large.pt"]="2b30654b6112c42a115563c638d238d9"
        ["sam2/sam2.1_hiera_l.yaml"]="e3487369c0d6a94fe67e52416f4b5a5a"
        ["foundationstereo/cfg.yaml"]="56ab75b42a5424b36c9a0d5227d0e592"
        ["foundationstereo/model_best_bp2.pth"]="fed9cbbb6f139e64520153d1f469f698"
    )
    
    # Download RoMa weights
    print_info "Downloading RoMa weights..."
    if [ ! -f "${REPO_DIR}/data/weights/roma/roma_outdoor.pth" ]; then
        print_info "Downloading roma_outdoor.pth..."
        if wget 'https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth' -O "${REPO_DIR}/data/weights/roma/roma_outdoor.pth"; then
            print_success "Downloaded roma_outdoor.pth"
        else
            print_error "Failed to download roma_outdoor.pth"
            download_failures=$((download_failures + 1))
        fi
    else
        print_info "roma_outdoor.pth already exists"
    fi
    
    if [ ! -f "${REPO_DIR}/data/weights/roma/dinov2_vitl14_pretrain.pth" ]; then
        print_info "Downloading dinov2_vitl14_pretrain.pth..."
        if wget 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth' -O "${REPO_DIR}/data/weights/roma/dinov2_vitl14_pretrain.pth"; then
            print_success "Downloaded dinov2_vitl14_pretrain.pth"
        else
            print_error "Failed to download dinov2_vitl14_pretrain.pth"
            download_failures=$((download_failures + 1))
        fi
    else
        print_info "dinov2_vitl14_pretrain.pth already exists"
    fi
    
    # Download SAM2 weights from Hugging Face
    print_info "Downloading SAM2 weights from Hugging Face..."
    
    if [ ! -f "${REPO_DIR}/data/weights/sam2/sam2.1_hiera_large.pt" ]; then
        print_info "Downloading sam2.1_hiera_large.pt from Hugging Face..."
        if wget "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt" -O "${REPO_DIR}/data/weights/sam2/sam2.1_hiera_large.pt"; then
            print_success "Downloaded sam2.1_hiera_large.pt"
        else
            print_error "Failed to download sam2.1_hiera_large.pt from Hugging Face"
            download_failures=$((download_failures + 1))
        fi
    else
        print_info "sam2.1_hiera_large.pt already exists"
    fi

    if [ ! -f "${REPO_DIR}/data/weights/sam2/sam2.1_hiera_l.yaml" ]; then
        print_info "Downloading sam2.1_hiera_l.yaml from Hugging Face..."
        if wget "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_l.yaml" -O "${REPO_DIR}/data/weights/sam2/sam2.1_hiera_l.yaml"; then
            print_success "Downloaded sam2.1_hiera_l.yaml"
        else
            print_error "Failed to download sam2.1_hiera_l.yaml from Hugging Face"
            download_failures=$((download_failures + 1))
        fi
    else
        print_info "sam2.1_hiera_l.yaml already exists"
    fi
    
    # Download FoundationStereo weights
    print_info "Downloading FoundationStereo weights..."
    print_warning "FoundationStereo weights are hosted on Google Drive and may require manual download"
    
    if [ ! -f "${REPO_DIR}/data/weights/foundationstereo/model_best_bp2.pth" ]; then
        print_info "Attempting to download model_best_bp2.pth from Google Drive..."
        if command -v curl &> /dev/null; then
            if curl -L "https://drive.usercontent.google.com/download?id=1Yh_2o9QCUrVqZrnAXZ7RUr0zTp3JrMKe&confirm=xxx" -o "${REPO_DIR}/data/weights/foundationstereo/model_best_bp2.pth"; then
                if [ -s "${REPO_DIR}/data/weights/foundationstereo/model_best_bp2.pth" ]; then
                    print_success "Downloaded model_best_bp2.pth"
                else
                    print_error "Downloaded file is empty. Google Drive may require manual download."
                    rm -f "${REPO_DIR}/data/weights/foundationstereo/model_best_bp2.pth"
                    download_failures=$((download_failures + 1))
                fi
            else
                print_error "Failed to download model_best_bp2.pth from Google Drive"
                download_failures=$((download_failures + 1))
            fi
        else
            print_error "curl not available. Cannot download model_best_bp2.pth"
            download_failures=$((download_failures + 1))
        fi
        
        if [ ! -f "${REPO_DIR}/data/weights/foundationstereo/model_best_bp2.pth" ]; then
            print_warning "Manual download required for model_best_bp2.pth:"
            print_warning "  URL: https://drive.google.com/file/d/1Yh_2o9QCUrVqZrnAXZ7RUr0zTp3JrMKe/view"
            print_warning "  Save to: ${REPO_DIR}/data/weights/foundationstereo/model_best_bp2.pth"
        fi
    else
        print_info "model_best_bp2.pth already exists"
    fi
    
    if [ ! -f "${REPO_DIR}/data/weights/foundationstereo/cfg.yaml" ]; then
        print_info "Attempting to download cfg.yaml from Google Drive..."
        if command -v curl &> /dev/null; then
            if curl -L "https://drive.usercontent.google.com/download?id=1tidGICH1_kTUUqi42aboKscuMY4IK_Xr&confirm=xxx" -o "${REPO_DIR}/data/weights/foundationstereo/cfg.yaml"; then
                if [ -s "${REPO_DIR}/data/weights/foundationstereo/cfg.yaml" ]; then
                    print_success "Downloaded cfg.yaml"
                else
                    print_error "Downloaded file is empty. Google Drive may require manual download."
                    rm -f "${REPO_DIR}/data/weights/foundationstereo/cfg.yaml"
                    download_failures=$((download_failures + 1))
                fi
            else
                print_error "Failed to download cfg.yaml from Google Drive"
                download_failures=$((download_failures + 1))
            fi
        else
            print_error "curl not available. Cannot download cfg.yaml"
            download_failures=$((download_failures + 1))
        fi
        
        if [ ! -f "${REPO_DIR}/data/weights/foundationstereo/cfg.yaml" ]; then
            print_warning "Manual download required for cfg.yaml:"
            print_warning "  URL: https://drive.google.com/file/d/1tidGICH1_kTUUqi42aboKscuMY4IK_Xr/view"
            print_warning "  Save to: ${REPO_DIR}/data/weights/foundationstereo/cfg.yaml"
        fi
    else
        print_info "cfg.yaml already exists"
    fi
    
    # Validate checksums for all downloaded files (unless skipped)
    if [ "${SKIP_CHECKSUM:-}" = "1" ]; then
        print_warning "Checksum validation skipped (SKIP_CHECKSUM=1)"
        print_warning "Downloaded files will not be verified for integrity"
    else
        print_info "Validating checksums for all weight files..."
        
        for relative_path in "${!expected_checksums[@]}"; do
            local file_path="${REPO_DIR}/data/weights/${relative_path}"
            local expected_checksum="${expected_checksums[$relative_path]}"
            
            if [ -f "$file_path" ]; then
                if ! validate_checksum "$file_path" "$expected_checksum"; then
                    checksum_failures=$((checksum_failures + 1))
                    # Remove corrupted file so it can be re-downloaded
                    print_warning "Removing corrupted file: $file_path"
                    rm -f "$file_path"
                fi
            else
                print_warning "File not found for checksum validation: $relative_path"
            fi
        done
    fi
    
    # Report download and validation results
    if [ $download_failures -eq 0 ] && [ $checksum_failures -eq 0 ]; then
        if [ "${SKIP_CHECKSUM:-}" = "1" ]; then
            print_success "All weight downloads completed successfully (checksum validation skipped)!"
        else
            print_success "All weight downloads completed successfully and checksums validated!"
        fi
    else
        local total_failures=$((download_failures + checksum_failures))
        print_error "$total_failures issue(s) found:"
        [ $download_failures -gt 0 ] && print_error "  - $download_failures download failure(s)"
        [ $checksum_failures -gt 0 ] && print_error "  - $checksum_failures checksum validation failure(s)"
        print_warning "Some weights may need to be downloaded manually or re-downloaded"
        print_info "Check the error messages above for specific failures"
        
        # List missing files
        print_info "Checking for missing weight files..."
        local missing_files=()
        
        [ ! -f "${REPO_DIR}/data/weights/roma/roma_outdoor.pth" ] && missing_files+=("roma/roma_outdoor.pth")
        [ ! -f "${REPO_DIR}/data/weights/roma/dinov2_vitl14_pretrain.pth" ] && missing_files+=("roma/dinov2_vitl14_pretrain.pth")
        [ ! -f "${REPO_DIR}/data/weights/sam2/sam2.1_hiera_large.pt" ] && missing_files+=("sam2/sam2.1_hiera_large.pt")
        [ ! -f "${REPO_DIR}/data/weights/sam2/sam2.1_hiera_l.yaml" ] && missing_files+=("sam2/sam2.1_hiera_l.yaml")
        [ ! -f "${REPO_DIR}/data/weights/foundationstereo/model_best_bp2.pth" ] && missing_files+=("foundationstereo/model_best_bp2.pth")
        [ ! -f "${REPO_DIR}/data/weights/foundationstereo/cfg.yaml" ] && missing_files+=("foundationstereo/cfg.yaml")
        
        if [ ${#missing_files[@]} -gt 0 ]; then
            print_error "Missing weight files:"
            for file in "${missing_files[@]}"; do
                print_error "  - data/weights/$file"
            done
            print_warning "The application may not work correctly without all required weights"
            return 1
        fi
        
        # If files exist but had checksum failures, still return error
        if [ $checksum_failures -gt 0 ]; then
            print_error "Checksum validation failed for some files. Please re-run the download process."
            return 1
        fi
    fi
    
    print_info "Weight download and validation process completed successfully"
    if [ "${SKIP_CHECKSUM:-}" = "1" ]; then
        print_info "All weights are downloaded (checksum validation was skipped)"
    else
        print_info "All weights are downloaded and verified with correct checksums"
    fi
}

# Function to display usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup         Complete setup: create directories, download weights, and build Docker image"
    echo "  start         Start the container with Jupyter notebook"
    echo "  stop          Stop the running container"
    echo "  clean         Stop container, remove volumes, and clean output directory"
    echo "  shell         Open shell in running container"
    echo ""
    echo "Environment Variables:"
    echo "  NO_CACHE          Set to '1' to disable Docker build cache during setup (default: use cache)"
    echo "  SKIP_CHECKSUM     Set to '1' to skip checksum validation for downloaded weights (default: validate)"
    echo "  USER              Current user (default: 'default') - used for container naming"
    echo "  PORT_START_INDEX  Starting port for dynamic allocation (default: 8888)"
    echo ""
    echo "Examples:"
    echo "  $0 setup                        # Complete setup with cache"
    echo "  NO_CACHE=1 $0 setup             # Complete setup without cache"
    echo "  SKIP_CHECKSUM=1 $0 setup        # Complete setup without checksum validation"
    echo "  $0 start                        # Start container"
    echo "  $0 shell                        # Access container shell"
    echo "  $0 clean                        # Clean everything"
    echo "  PORT_START_INDEX=9000 $0 start  # Start with custom port range"
    echo ""
    echo "Docker Configuration:"
    echo "  Local Image: $LOCAL_IMAGE"
    echo "  Container: $CONTAINER_NAME"
    echo "  Network: $NETWORK_NAME"
    echo "  Port Range: ${PORT_START_INDEX}+"
    echo ""
    echo "Directories:"
    echo "  Project root: $REPO_DIR"
    echo "  Compose directory: $COMPOSE_DIR"
}

# Function to start container
start_container() {
    print_info "Starting 3D reconstruction container for user ${USER:-default}..."
    create_directories
    
    # Find available port for Jupyter
    JUPYTER_PORT=$(find_available_port $PORT_START_INDEX)
    export JUPYTER_HOST_PORT="$JUPYTER_PORT"
    
    export IMAGE_NAME="$LOCAL_IMAGE"
    print_info "Using locally built image: $LOCAL_IMAGE"
    print_info "Container name: $CONTAINER_NAME"
    print_info "Network name: $NETWORK_NAME"
    print_info "Allocated Jupyter port: $JUPYTER_PORT"
    
    # Use docker-compose configuration
    local compose_cmd=$(get_compose_cmd)
    
    # Container runs Jupyter by default as per Dockerfile CMD
    cd "$COMPOSE_DIR"
    eval "$compose_cmd up -d reconstruction-app"
    
    print_success "Container started successfully!"
    print_info "Jupyter notebook is available at: http://localhost:$JUPYTER_PORT"
    print_info "No authentication required (token disabled)"
    print_info ""
    print_info "To access shell: $0 shell"
}

# Function to stop containers
stop_containers() {
    print_info "Stopping container..."
    local compose_cmd=$(get_compose_cmd)
    cd "$COMPOSE_DIR" && eval "$compose_cmd down"
    print_success "Container stopped"
}

# Function to open shell
open_shell() {
    print_info "Opening shell in container..."
    if docker ps | grep -q "$CONTAINER_NAME"; then
        docker exec -it "$CONTAINER_NAME" bash
    else
        print_error "No running 3D reconstruction container found for user ${USER:-default}. Start a container first with: $0 start"
        exit 1
    fi
}

# Function to clean up everything
clean_up() {
    print_warning "This will:"
    print_warning "  - Stop and remove the container"
    print_warning "  - Remove volumes and network"
    print_warning "  - Clean the output directory"
    print_warning "  - Preserve the Docker image"
    echo ""
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning up..."
        
        # Stop and remove containers and volumes
        cd "$COMPOSE_DIR"
        docker-compose -f docker-compose.yml down -v --remove-orphans 2>/dev/null || true
        
        # Clean up user-specific network if it exists
        if docker network ls | grep -q "$NETWORK_NAME"; then
            print_info "Removing user-specific network: $NETWORK_NAME"
            docker network rm "$NETWORK_NAME" 2>/dev/null || true
        fi
        
        # Clean output directory
        local output_dir="${REPO_DIR}/data/output"
        if [ -d "$output_dir" ]; then
            print_info "Cleaning output directory: $output_dir"
            rm -rf "${output_dir:?}"/*
            print_success "Output directory cleaned"
        fi
        
        # Clean up unused Docker resources
        docker system prune -f
        
        print_success "Cleanup completed"
        print_info "Docker image preserved: $LOCAL_IMAGE"
        print_info "To remove the image: docker rmi $LOCAL_IMAGE"
    else
        print_info "Cleanup cancelled"
    fi
}

# Function to perform complete setup
setup() {
    print_info "Starting complete setup..."
    print_info "This will create directories, download weights, and build the Docker image"
    print_warning "This process may take 30-60 minutes and requires >20GB disk space"
    echo ""
    
    check_requirements
    create_directories
    download_weights
    build_image
    
    print_success "Setup completed successfully!"
    print_info "Next steps:"
    print_info "  1. Run '$0 start' to start the container"
    print_info "  2. Run '$0 shell' to access the container"
}

# Main script logic
case "${1:-}" in
    "setup")
        setup
        ;;
    "start")
        start_container
        ;;
    "stop")
        stop_containers
        ;;
    "clean")
        clean_up
        ;;
    "shell")
        open_shell
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    "")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        print_info "Available commands: setup, start, stop, clean, shell"
        show_usage
        exit 1
        ;;
esac 