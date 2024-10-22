# Face Attendance System Setup and Run Script

# Function to check if a command exists
function Test-Command($command) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try { if (Get-Command $command) { return $true } }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

# Check if Python is installed
if (-not (Test-Command python)) {
    Write-Host "Python is not installed. Please install Python 3.7 or later and try again."
    exit 1
}

# Check if pip is installed
if (-not (Test-Command pip)) {
    Write-Host "pip is not installed. Please install pip and try again."
    exit 1
}

# Create a virtual environment
Write-Host "Creating a virtual environment..."
python -m venv venv

# Activate the virtual environment
Write-Host "Activating the virtual environment..."
.\venv\Scripts\Activate

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt

# Download LFW dataset
$lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
$lfw_file = "lfw.tgz"
$lfw_dir = "lfw_dataset"

Write-Host "Downloading LFW dataset..."
Invoke-WebRequest -Uri $lfw_url -OutFile $lfw_file

# Check if 7-Zip is installed for extraction
if (Test-Command 7z) {
    Write-Host "Extracting LFW dataset..."
    7z x $lfw_file -o"$lfw_dir" -y
} else {
    Write-Host "7-Zip is not installed. Please extract the $lfw_file manually to $lfw_dir directory."
}

# Update the data_dir path in the Python script
Write-Host "Updating the data_dir path in the Python script..."
(Get-Content face_attendance.py) -replace "data_dir = 'path/to/lfw/dataset'", "data_dir = '$lfw_dir'" | Set-Content face_attendance.py

# Run the Python script
Write-Host "Running the face attendance script..."
python face_attendance.py

# Deactivate the virtual environment
deactivate

Write-Host "Script execution completed. Check the 'preprocessing_comparison.png' for results."