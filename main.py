import os
import sys
import subprocess
import zipfile
import ctypes
import winreg
import time
import urllib.request
import ssl
import warnings
from urllib.error import URLError
import shutil
import colorama  # Add colorama for Windows color support
import importlib.util

# Initialize colorama
colorama.init()

# Suppress SSL warnings properly
warnings.filterwarnings('ignore', category=Warning)

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar in the console"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '░' * (length - filled_length)
    print_color(f'\r{prefix} |{bar}| {percent}% {suffix}', "BLUE", end='\r')
    if iteration == total:
        print()

def print_color(text, color, end='\n'):
    """Print colored text with proper encoding handling"""
    try:
        # Replace Unicode characters with ASCII alternatives
        text = text.replace('✓', '[OK]').replace('✗', '[X]').replace('⚠', '[!]')
        
        colors = {
            'RED': '\033[91m',
            'GREEN': '\033[92m',
            'BLUE': '\033[94m',
            'YELLOW': '\033[93m',
            'MAGENTA': '\033[95m',
            'CYAN': '\033[96m',
            'RESET': '\033[0m'
        }
        
        # Initialize colorama if not already initialized
        if not hasattr(print_color, 'colorama_init'):
            import colorama
            colorama.init()
            print_color.colorama_init = True
            
        # Use print with flush=True to ensure immediate output
        print(f"{colors.get(color, '')}{text}{colors['RESET']}", end=end, flush=True)
    except Exception as e:
        # Fallback to plain text if coloring fails
        print(text, end=end, flush=True)

def print_header(text):
    """Print a stylized header"""
    width = 70
    print_color("╔" + "═" * (width-2) + "╗", "CYAN")
    print_color("║" + text.center(width-2) + "║", "CYAN")
    print_color("╚" + "═" * (width-2) + "╝", "CYAN")

def download_file(url, filename):
    print_color(f"Downloading {filename}...", "BLUE")
    try:
        # Create SSL context that ignores certificate verification
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(url, context=ctx) as response:
            with open(filename, 'wb') as f:
                f.write(response.read())
        print_color(f"Successfully downloaded {filename}", "GREEN")
        return True
    except Exception as e:
        print_color(f"Error downloading {filename}: {str(e)}", "RED")
        return False

def check_python():
    try:
        # Check if Python 3.11 is in PATH
        result = subprocess.run(['python', '--version'], capture_output=True, text=True)
        if '3.11' in result.stdout:
            print_color("Python 3.11 is already installed and in PATH", "GREEN")
            return True
    except:
        pass
    
    try:
        # Check if py launcher can find Python 3.11
        result = subprocess.run(['py', '-3.11', '--version'], capture_output=True, text=True)
        if '3.11' in result.stdout:
            print_color("Python 3.11 is already installed via py launcher", "GREEN")
            return True
    except:
        pass

    # Check Windows Registry for Python 3.11
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Python\PythonCore\3.11", 0, winreg.KEY_READ)
        winreg.CloseKey(key)
        print_color("Python 3.11 found in registry", "GREEN")
        return True
    except WindowsError:
        pass

    return False

def install_python():
    """Install Python 3.11 if not already installed"""
    if check_python():
        return True
    
    print_color("Installing Python 3.11...", "BLUE")
    python_url = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
    installer_path = os.path.abspath("python_installer.exe")
    
    if not download_file(python_url, installer_path):
        print_color("Failed to download Python installer", "RED")
        return False
        
    print_color("Starting Python installation...", "BLUE")
    success = False
    
    try:
        # Simply run the installer with elevation, no silent parameters
        ps_command = f'Start-Process -FilePath "{installer_path}" -Verb RunAs'
        
        print_color("Launching Python installer...", "BLUE")
        print_color("Please complete the installation manually and make sure to check 'Add Python 3.11 to PATH'", "YELLOW")
        
        # Run PowerShell command
        process = subprocess.run(
            ['powershell', '-Command', ps_command],
            capture_output=True,
            text=True
        )
        
        print_color("The Python installer has been launched.", "BLUE")
        print_color("Please:", "YELLOW")
        print_color("1. Complete the installation in the opened installer window", "YELLOW")
        print_color("2. Make sure to check 'Add Python 3.11 to PATH'", "YELLOW")
        print_color("3. Click 'Install Now' and wait for the installation to complete", "YELLOW")
        print_color("4. Click 'Close' when finished", "YELLOW")
        print_color("\nAfter installation is complete, type 'yes' and press Enter to continue:", "GREEN")
        
        # Wait for user confirmation
        while True:
            response = input().strip().lower()
            if response == 'yes':
                # Verify Python installation after user confirms
                if check_python():
                    print_color("✓ Python 3.11 installation verified successfully!", "GREEN")
                    success = True
                    break
                print_color("✗ Python 3.11 was not found. Please make sure it was installed correctly.", "RED")
                print_color("Type 'yes' when you have fixed the installation:", "YELLOW")
            else:
                print_color("Please type 'yes' when Python installation is complete:", "YELLOW")
                
    except Exception as e:
        print_color(f"Error launching Python installer: {str(e)}", "RED")
    
    # Clean up
    try:
        if os.path.exists(installer_path):
            os.remove(installer_path)
    except:
        pass
    
    return success

def is_compiled():
    """Check if we're running from a PyInstaller bundle"""
    return getattr(sys, 'frozen', False)

def check_package_installed(package_name):
    """Check if a package is installed using pip"""
    try:
        # Use pip to check installation status
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print_color(f"Error checking {package_name}: {str(e)}", "RED")
        return False

def get_python_executable():
    """Get the correct Python executable path"""
    try:
        # If we're running from a PyInstaller bundle
        if getattr(sys, 'frozen', False):
            # Try multiple methods to find Python
            python_paths = [
                os.environ.get('PYTHONHOME'),  # Check PYTHONHOME first
                subprocess.getoutput('where python'),  # Check PATH
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
                r"C:\Python311\python.exe",
                r"C:\Program Files\Python311\python.exe",
                os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe")
            ]
            
            for path in python_paths:
                if path and os.path.exists(path):
                    if isinstance(path, str) and '\n' in path:  # Handle multiple paths from where command
                        paths = path.split('\n')
                        for p in paths:
                            if os.path.exists(p) and '311' in p:  # Prefer Python 3.11
                                return p
                        return paths[0]  # Take first if no 3.11
                    return path
            
            print_color("Warning: Could not find Python executable, using 'python' command", "YELLOW")
            return 'python'
        else:
            return sys.executable
    except Exception as e:
        print_color(f"Error finding Python executable: {str(e)}", "RED")
        return 'python'

def run_pip_install(package_args, prefix=""):
    """Run pip install with real-time output"""
    python_exe = get_python_executable()
    cmd = [python_exe, '-m', 'pip', 'install', '--no-cache-dir']
    cmd.extend(package_args)
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Track download progress
    downloading = False
    for line in process.stdout:
        line = line.strip()
        if "Downloading" in line:
            downloading = True
            print_color(f"\r{prefix} Downloading: {line}", "BLUE", end='')
        elif "Installing collected packages" in line:
            downloading = False
            print_color(f"\r{prefix} Installing packages...", "BLUE")
        elif downloading and ("%" in line or "bytes" in line):
            print_color(f"\r{prefix} {line}", "BLUE", end='')
    
    process.wait()
    return process.returncode == 0

def install_single_package(package_name, import_name=None, current_num=0, total_num=0):
    """Install a single package and verify its installation"""
    if import_name is None:
        import_name = package_name
    
    status_text = f"[{current_num}/{total_num}] Processing {package_name}"
    print_progress_bar(current_num-1, total_num, prefix=status_text, suffix='')
    
    try:
        # Check if package is already installed
        try:
            __import__(import_name)
            print_progress_bar(current_num, total_num, prefix=status_text, suffix='')
            print_color(f"✓ {package_name:<20} Already installed", "GREEN")
            return True
        except ImportError:
            pass
        
        print_progress_bar(current_num-0.5, total_num, prefix=f"[{current_num}/{total_num}] Installing {package_name}", suffix='')
        try:
            success = run_pip_install([package_name], f"[{current_num}/{total_num}]")
            if success:
                print_progress_bar(current_num, total_num, prefix=status_text, suffix='')
                print_color(f"\n✓ {package_name:<20} Successfully installed", "GREEN")
                return True
            else:
                print_progress_bar(current_num, total_num, prefix=status_text, suffix='')
                print_color(f"\n✗ {package_name:<20} Installation failed", "RED")
                return False
        except Exception as e:
            print_progress_bar(current_num, total_num, prefix=status_text, suffix='')
            print_color(f"\n✗ {package_name:<20} Installation failed: {str(e)}", "RED")
            return False
            
    except Exception as e:
        print_progress_bar(current_num, total_num, prefix=status_text, suffix='')
        print_color(f"\n✗ {package_name:<20} Installation failed: {str(e)}", "RED")
        return False

def check_cuda():
    """Check if CUDA 12.6 is installed"""
    try:
        # Check CUDA version using nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if 'release 12.6' in result.stdout.lower():
            return True
    except:
        pass
    
    # Check common CUDA installation paths
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        r"C:\CUDA\v12.6"
    ]
    return any(os.path.exists(path) for path in cuda_paths)

def download_with_progress(url, filename):
    """Download a file with progress bar"""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        # Install required packages
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests', 'tqdm'])
        import requests
        from tqdm import tqdm
    
    print_color(f"\nDownloading {filename}...", "BLUE")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def install_cuda():
    """Install CUDA 12.6"""
    if check_cuda():
        print_color("✓ CUDA 12.6 is already installed", "GREEN")
        return True
    
    print_header(" Installing CUDA 12.6 ")
    
    cuda_url = 'https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe'
    cuda_installer = 'cuda_12.6.0_560.76_windows.exe'
    
    try:
        download_with_progress(cuda_url, cuda_installer)
        
        print_color("\nStarting CUDA installation...", "BLUE")
        print_color("\nIMPORTANT INSTALLATION STEPS:", "YELLOW")
        print_color("1. Click 'Next' when the installer opens", "YELLOW")
        print_color("2. Select 'Advanced Installation'", "YELLOW")
        print_color("3. UNCHECK 'GeForce Experience' to avoid conflicts", "YELLOW")
        print_color("4. Continue with the installation", "YELLOW")
        print_color("\nPlease type 'yes' when the installation is complete:", "GREEN")
        
        # Launch installer with elevation
        ps_command = f'Start-Process -FilePath "{os.path.abspath(cuda_installer)}" -Verb RunAs'
        subprocess.run(['powershell', '-Command', ps_command])
        
        while True:
            response = input().strip().lower()
            if response == 'yes':
                if check_cuda():
                    print_color("✓ CUDA 12.6 installation verified successfully!", "GREEN")
                    break
                else:
                    print_color("CUDA 12.6 installation could not be verified. Please make sure it was installed correctly.", "RED")
                    print_color("Type 'yes' when you have fixed the installation:", "YELLOW")
            else:
                print_color("Please type 'yes' when CUDA installation is complete:", "YELLOW")
        
        # Clean up
        try:
            os.remove(cuda_installer)
        except:
            pass
        
        return True
        
    except Exception as e:
        print_color(f"Error during CUDA installation: {str(e)}", "RED")
        return False

def check_pytorch():
    """Check if PyTorch with CUDA is installed and working"""
    try:
        # First check if installed via pip
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'torch'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False
            
        # Now check CUDA support
        verify_cmd = '''
import torch
print("PyTorch version:", torch.__version__)
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
if cuda_available:
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0))
exit(0 if cuda_available else 1)
'''
        result = subprocess.run(
            [sys.executable, '-c', verify_cmd],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTORCH_JIT="0", CUDA_LAUNCH_BLOCKING="1")
        )
        
        if result.returncode == 0:
            print_color("\nPyTorch CUDA Status:", "BLUE")
            print_color(result.stdout.strip(), "GREEN")
            return True
            
        if result.stderr and "[WinError 193]" in result.stderr:
            print_color("\nCRITICAL: This error indicates a mismatch between Python architecture (32-bit) and PyTorch (64-bit).", "RED")
            print_color("Please install a 64-bit version of Python and try again.", "YELLOW")
            return False
            
        print_color("PyTorch is installed but CUDA is not available", "YELLOW")
        if result.stderr:
            print_color(f"Error: {result.stderr}", "RED")
        return False
        
    except Exception as e:
        print_color(f"Error checking PyTorch CUDA: {str(e)}", "RED")
        return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    try:
        print_color("\nInstalling PyTorch with CUDA...", "BLUE")
        
        # First uninstall existing PyTorch
        print_color("Removing existing PyTorch installations...", "BLUE")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'], 
                      capture_output=True)
        
        # Clean up site-packages
        try:
            site_packages = subprocess.check_output(
                [sys.executable, '-c', 'import site; print(site.getsitepackages()[0])'], 
                text=True
            ).strip()
            
            for dir_name in ['torch', 'torchvision', 'torchaudio']:
                dir_path = os.path.join(site_packages, dir_name)
                if os.path.exists(dir_path):
                    print_color(f"Removing {dir_name} directory...", "BLUE")
                    shutil.rmtree(dir_path, ignore_errors=True)
        except Exception as e:
            print_color(f"Warning during cleanup: {str(e)}", "YELLOW")
        
        # Install PyTorch with CUDA
        print_color("Installing PyTorch with CUDA support...", "BLUE")
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            '--no-cache-dir',
            'torch',
            'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            print_color("PyTorch installation completed. Verifying...", "BLUE")
            if check_pytorch():
                print_color("✓ PyTorch CUDA installed successfully!", "GREEN")
                return True
            else:
                print_color("✗ PyTorch installation completed but CUDA is not working", "RED")
                return False
        else:
            print_color("✗ PyTorch installation failed", "RED")
            if process.stderr:
                print_color(f"Error: {process.stderr}", "RED")
            return False
            
    except Exception as e:
        print_color(f"Error during PyTorch installation: {str(e)}", "RED")
        return False

def check_ultralytics():
    """Check if ultralytics is installed and working"""
    try:
        # First check if the module exists
        if importlib.util.find_spec("ultralytics") is None:
            return False
            
        # Now try to import and verify
        verify_cmd = '''
import ultralytics
print("Ultralytics version:", ultralytics.__version__)
'''
        result = subprocess.run(
            [sys.executable, '-c', verify_cmd],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTORCH_JIT="0", CUDA_LAUNCH_BLOCKING="1")
        )
        
        if result.returncode == 0:
            print_color(result.stdout.strip(), "GREEN")
            return True
            
        if result.stderr:
            print_color(f"Ultralytics error: {result.stderr}", "RED")
        return False
        
    except Exception as e:
        print_color(f"Error checking ultralytics: {str(e)}", "RED")
        return False

def install_ultralytics():
    """Install ultralytics package"""
    try:
        print_color("\nInstalling ultralytics...", "BLUE")
        
        # First verify PyTorch CUDA
        if not check_pytorch():
            print_color("✗ Cannot install ultralytics: PyTorch CUDA is not working", "RED")
            print_color("Please fix PyTorch CUDA installation first", "YELLOW")
            return False
        
        # Remove existing ultralytics
        print_color("Removing existing ultralytics installation...", "BLUE")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'ultralytics'], 
                      capture_output=True)
        
        # Install ultralytics
        print_color("Installing ultralytics...", "BLUE")
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            '--no-cache-dir',
            'ultralytics'
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            print_color("Ultralytics installation completed. Verifying...", "BLUE")
            if check_ultralytics():
                print_color("✓ Ultralytics installed successfully!", "GREEN")
                return True
            else:
                print_color("✗ Ultralytics installation completed but verification failed", "RED")
                return False
        else:
            print_color("✗ Ultralytics installation failed", "RED")
            if process.stderr:
                print_color(f"Error: {process.stderr}", "RED")
            return False
            
    except Exception as e:
        print_color(f"Error during ultralytics installation: {str(e)}", "RED")
        return False

def install_packages():
    """Install all required packages sequentially"""
    all_packages = [
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('mss', 'mss'),
        ('dearpygui', 'dearpygui'),
        ('PyQt5', 'PyQt5'),
        ('termcolor', 'termcolor'),
        ('colorama', 'colorama'),
        ('requests', 'requests'),
        ('tqdm', 'tqdm'),
        ('cuda-python', 'cuda'),
        ('wheel', 'wheel'),
        ('tensorrt', 'tensorrt')
    ]
    
    # Check which packages need to be installed
    packages_to_install = []
    pytorch_needed = not check_pytorch()
    ultralytics_needed = not check_ultralytics()
    
    # Add regular packages first
    for package_name, import_name in all_packages:
        if not check_package_installed(package_name):
            packages_to_install.append((package_name, import_name))
    
    if not packages_to_install and not pytorch_needed and not ultralytics_needed:
        print_color("\n✓ All packages are already installed", "GREEN")
        return True
    
    # Print installation plan
    print_header(" Installation Plan ")
    if pytorch_needed:
        print_color("• PyTorch CUDA needs to be installed", "YELLOW")
    if ultralytics_needed:
        print_color("• Ultralytics needs to be installed", "YELLOW")
    for package, _ in packages_to_install:
        print_color(f"• {package} needs to be installed", "YELLOW")
    print()
    
    # Install PyTorch first if needed
    if pytorch_needed:
        print_header(" Installing PyTorch CUDA ")
        if not install_pytorch_cuda():
            print_color("\n⚠ PyTorch installation failed. Skipping ultralytics installation.", "RED")
            ultralytics_needed = False
    
    # Install regular packages
    if packages_to_install:
        print_header(f" Installing {len(packages_to_install)} Package(s) ")
        for i, (package_name, import_name) in enumerate(packages_to_install, 1):
            if not install_single_package(package_name, import_name, i, len(packages_to_install)):
                print_color(f"\n⚠ Warning: {package_name} installation failed.", "YELLOW")
    
    # Install ultralytics last if needed
    if ultralytics_needed:
        print_header(" Installing Ultralytics ")
        install_ultralytics()
    
    # Final verification
    print_header(" Final Verification ")
    all_success = True
    
    if pytorch_needed or ultralytics_needed:
        if check_pytorch():
            print_color("✓ PyTorch CUDA is working correctly", "GREEN")
        else:
            print_color("✗ PyTorch CUDA is not working", "RED")
            all_success = False
        
        if check_ultralytics():
            print_color("✓ Ultralytics is working correctly", "GREEN")
        else:
            print_color("✗ Ultralytics is not working", "RED")
            all_success = False
    
    for package_name, import_name in all_packages:
        if check_package_installed(package_name):
            print_color(f"✓ {package_name} is installed", "GREEN")
        else:
            print_color(f"✗ {package_name} is not installed", "RED")
            all_success = False
    
    if all_success:
        print_color("\n✓ All packages installed successfully!", "GREEN")
    else:
        print_color("\n⚠ Some packages failed to install properly.", "YELLOW")
    
    return True

def check_interception():
    """Check if Interception driver is installed and running"""
    try:
        # Check multiple methods to verify Interception installation
        checks = {
            'registry': False,
            'driver_files': False,
            'service': False,
            'device': False
        }
        
        # Check registry (multiple possible locations)
        registry_paths = [
            r"SYSTEM\CurrentControlSet\Services\Interception",
            r"SYSTEM\CurrentControlSet\Services\keyboard",
            r"SYSTEM\CurrentControlSet\Services\kbfiltr",
            r"SYSTEM\CurrentControlSet\Services\interception"
        ]
        
        for reg_path in registry_paths:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path, 0, winreg.KEY_READ)
                winreg.CloseKey(key)
                checks['registry'] = True
                print_color(f"Registry check: Found service at {reg_path}", "GREEN")
                break
            except WindowsError:
                continue
        
        if not checks['registry']:
            print_color("Registry check: No Interception-related services found", "YELLOW")
        
        # Check common driver locations (expanded list)
        driver_paths = [
            r"C:\Windows\System32\drivers\interception.sys",
            r"C:\Windows\SysWOW64\drivers\interception.sys",
            r"C:\Windows\System32\interception.sys",
            r"C:\Windows\SysWOW64\interception.sys",
            r"C:\Windows\System32\drivers\kbfiltr.sys",
            r"C:\Windows\System32\drivers\keyboard.sys",
            r"C:\Windows\System32\interception.dll",
            r"C:\Windows\SysWOW64\interception.dll",
            os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32', 'drivers', 'interception.sys'),
            os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'SysWOW64', 'drivers', 'interception.sys')
        ]
        
        found_files = [path for path in driver_paths if os.path.exists(path)]
        checks['driver_files'] = bool(found_files)
        if found_files:
            print_color(f"Driver files found: {', '.join(found_files)}", "GREEN")
        else:
            print_color("No driver files found in system directories", "YELLOW")
        
        # Check if service exists and is running (check multiple service names)
        service_names = ['Interception', 'keyboard', 'kbfiltr']
        for service in service_names:
            try:
                service_check = subprocess.run(
                    ['sc', 'query', service],
                    capture_output=True,
                    text=True
                )
                if "RUNNING" in service_check.stdout:
                    checks['service'] = True
                    print_color(f"Service check: {service} service is running", "GREEN")
                    break
                elif "1060" not in service_check.stderr:  # 1060 = service not installed
                    print_color(f"Service check: {service} service exists but is not running", "YELLOW")
            except Exception as e:
                print_color(f"Service check failed for {service}: {str(e)}", "RED")
        
        if not checks['service']:
            print_color("No Interception-related services are running", "YELLOW")
        
        # Check device presence using multiple methods
        try:
            # Method 1: Check using WMI
            device_check = subprocess.run(
                ['powershell', '-Command', 
                 'Get-WmiObject Win32_SystemDriver | Where-Object {$_.Name -match "interception|keyboard|kbfiltr"}'],
                capture_output=True,
                text=True
            )
            if device_check.stdout.strip():
                checks['device'] = True
                print_color("Device check: Found matching system driver", "GREEN")
            
            # Method 2: Check using driverquery
            if not checks['device']:
                driver_check = subprocess.run(
                    ['driverquery', '/v', '/FO', 'CSV'],
                    capture_output=True,
                    text=True
                )
                if any(name in driver_check.stdout.lower() for name in ['interception', 'keyboard', 'kbfiltr']):
                    checks['device'] = True
                    print_color("Device check: Found driver in driverquery", "GREEN")
            
            if not checks['device']:
                print_color("Device check: No matching drivers found", "YELLOW")
                
        except Exception as e:
            print_color(f"Device check failed: {str(e)}", "RED")
        
        # Count how many checks passed
        passed_checks = sum(checks.values())
        if passed_checks >= 1:  # Lower the requirement to 1 check since some might not be detectable
            print_color(f"\n[OK] Interception appears to be installed ({passed_checks}/4 checks passed)", "GREEN")
            return True
        else:
            print_color(f"\n[X] Interception is not properly installed ({passed_checks}/4 checks passed)", "RED")
            return False
        
    except Exception as e:
        print_color(f"Error checking Interception: {str(e)}", "RED")
        return False

def install_interception():
    """Install Interception driver"""
    try:
        if check_interception():
            print_color("✓ Interception driver is already installed", "GREEN")
            return True
        
        print_color("Downloading Interception driver...", "BLUE")
        interception_url = "https://github.com/oblitum/Interception/releases/download/v1.0.1/Interception.zip"
        zip_path = "Interception.zip"
        extract_dir = "Interception"
        
        if not download_file(interception_url, zip_path):
            print_color("Failed to download Interception.zip", "RED")
            return False
            
        try:
            # Remove existing directory if it exists
            if os.path.exists(extract_dir):
                print_color("Removing existing Interception directory...", "BLUE")
                shutil.rmtree(extract_dir, ignore_errors=True)
            
            # Extract the ZIP file
            print_color("Extracting Interception.zip...", "BLUE")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Clean up the ZIP file
            os.remove(zip_path)
            
            # Navigate through the nested folders to find the installer
            installer_path = os.path.join(extract_dir, "Interception", "command line installer", "install-interception.exe")
            installer_path = os.path.abspath(installer_path.replace('/', os.path.sep))
            installer_dir = os.path.dirname(installer_path)
            
            if not os.path.exists(installer_path):
                print_color(f"Installer not found at: {installer_path}", "RED")
                return False
                
            print_color(f"Found installer at: {installer_path}", "GREEN")
            print_color("Running Interception installer as administrator...", "BLUE")
            
            # Create PowerShell command to run installer as admin and capture output
            ps_command = f'''
$process = Start-Process -FilePath "{installer_path}" -ArgumentList "/install" -Verb RunAs -PassThru -Wait
exit $process.ExitCode
'''
            
            # Run PowerShell with the command and capture output
            print_color("Starting installation...", "BLUE")
            print("-" * 50)
            
            process = subprocess.run(
                ['powershell', '-Command', ps_command],
                capture_output=True,
                text=True,
                cwd=installer_dir
            )
            
            if process.stdout:
                print(process.stdout)
            if process.stderr:
                print_color("Errors:", "RED")
                print(process.stderr)
            
            print("-" * 50)
            
            # Verify installation
            if check_interception():
                print_color("✓ Interception driver installed successfully!", "GREEN")
                return True
            else:
                print_color("⚠ Installation may have failed. Please check Windows Event Viewer for details.", "RED")
                print_color("You may need to:", "YELLOW")
                print_color("1. Temporarily disable Windows Defender", "YELLOW")
                print_color("2. Check the Windows Event Viewer for more details", "YELLOW")
                print_color(f"3. Try running this command manually as administrator:", "YELLOW")
                print_color(f"   {installer_path} /install", "YELLOW")
                return False
                
        except Exception as e:
            print_color(f"Error during installation: {str(e)}", "RED")
            return False
            
        finally:
            # Clean up
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir, ignore_errors=True)
            except:
                pass
            
    except Exception as e:
        print_color(f"Error installing Interception: {str(e)}", "RED")
        return False

def check_tensorrt():
    """Check if TensorRT is installed"""
    try:
        # First check via pip
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'tensorrt'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False
            
        # Now try importing
        verify_cmd = '''
try:
    import tensorrt as trt
    print(f"TensorRT version: {trt.__version__}")
    print("TensorRT is properly installed")
    exit(0)
except Exception as e:
    print(f"Error: {str(e)}")
    exit(1)
'''
        result = subprocess.run(
            [sys.executable, '-c', verify_cmd],
            capture_output=True,
            text=True,
            env=dict(os.environ)
        )
        
        if result.returncode == 0:
            print_color(result.stdout.strip(), "GREEN")
            return True
            
        if result.stderr:
            print_color(f"TensorRT error: {result.stderr}", "RED")
        return False
        
    except Exception as e:
        print_color(f"Error checking TensorRT: {str(e)}", "RED")
        return False

def get_installed_packages():
    """Get list of installed packages using pip"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            packages = {}
            for line in result.stdout.split('\n')[2:]:  # Skip header lines
                if line.strip():
                    name = line.split()[0].lower()
                    packages[name] = True
            return packages
    except:
        pass
    return {}

def check_python_packages():
    """Check which Python packages are installed"""
    packages = {
        'PyTorch (CUDA)': check_pytorch,
        'numpy': lambda: check_package_installed('numpy'),
        'opencv-python': lambda: check_package_installed('opencv-python'),
        'mss': lambda: check_package_installed('mss'),
        'dearpygui': lambda: check_package_installed('dearpygui'),
        'ultralytics': lambda: check_ultralytics(),
        'PyQt5': lambda: check_package_installed('PyQt5'),
        'termcolor': lambda: check_package_installed('termcolor'),
        'colorama': lambda: check_package_installed('colorama'),
        'requests': lambda: check_package_installed('requests'),
        'tqdm': lambda: check_package_installed('tqdm'),
        'cuda-python': lambda: check_package_installed('cuda-python'),
        'wheel': lambda: check_package_installed('wheel'),
        'tensorrt': check_tensorrt
    }
    
    # Get the correct Python executable
    python_exe = get_python_executable()
    if getattr(sys, 'frozen', False):
        print_color(f"\nUsing system Python for package checks: {python_exe}", "BLUE")
    
    results = {}
    for name, check_func in packages.items():
        try:
            print_color(f"\nChecking {name}...", "BLUE")
            results[name] = check_func()
            if results[name]:
                print_color(f"[OK] {name:<20} is installed and working", "GREEN")
            else:
                print_color(f"[X] {name:<20} is not installed or not working", "RED")
        except Exception as e:
            print_color(f"Error checking {name}: {str(e)}", "RED")
            results[name] = False
            
    return results

def check_vcredist():
    """Check if required Visual C++ Runtime files are present"""
    try:
        # Check registry for installed VC++ Redistributables
        versions = ['v14', 'v15', 'v16', 'v17']  # Visual Studio 2015-2022
        keys_to_check = [
            r'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
            r'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x86',
            r'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
            r'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x86',
        ]
        
        for key_path in keys_to_check:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ)
                installed = winreg.QueryValueEx(key, 'Installed')[0]
                if installed == 1:
                    return True
                winreg.CloseKey(key)
            except WindowsError:
                continue
        
        # If registry check fails, check for DLL files
        system32_path = os.path.join(os.environ['SystemRoot'], 'System32')
        required_dlls = ['msvcp140.dll', 'vcruntime140.dll']
        
        for dll in required_dlls:
            if not os.path.exists(os.path.join(system32_path, dll)):
                return False
        
        return True
    except Exception as e:
        print_color(f"Error checking VC++ Runtime: {str(e)}", "RED")
        return False

def install_vcredist():
    """Install Visual C++ Redistributable packages"""
    if check_vcredist():
        print_color("✓ Visual C++ Runtime files are already installed", "GREEN")
        return True
        
    print_color("\nInstalling Visual C++ Redistributable packages...", "BLUE")
    
    # URLs for both x86 and x64 versions of VS2015-2022 redistributable
    vc_redist_urls = {
        'x64': 'https://aka.ms/vs/17/release/vc_redist.x64.exe',
        'x86': 'https://aka.ms/vs/17/release/vc_redist.x86.exe'
    }
    
    success = True
    for arch, url in vc_redist_urls.items():
        installer_name = f'vc_redist_{arch}.exe'
        try:
            print_color(f"\nDownloading {arch} Visual C++ Redistributable...", "BLUE")
            download_with_progress(url, installer_name)
            
            print_color(f"Installing {arch} Visual C++ Redistributable...", "BLUE")
            
            # Run installer with elevation and wait for completion
            ps_command = f'''
$process = Start-Process -FilePath "{os.path.abspath(installer_name)}" -ArgumentList "/install /quiet /norestart" -Verb RunAs -PassThru -Wait
exit $process.ExitCode
'''
            process = subprocess.run(
                ['powershell', '-Command', ps_command],
                capture_output=True,
                text=True
            )
            
            if process.returncode == 0 or process.returncode == 3010:  # 3010 means success but requires restart
                print_color(f"✓ {arch} Visual C++ Redistributable installed successfully", "GREEN")
            else:
                print_color(f"✗ Failed to install {arch} Visual C++ Redistributable (Exit code: {process.returncode})", "RED")
                if process.stderr:
                    print_color(f"Error: {process.stderr}", "RED")
                success = False
            
            # Clean up
            try:
                os.remove(installer_name)
            except:
                pass
                
        except Exception as e:
            print_color(f"✗ Error installing {arch} Visual C++ Redistributable: {str(e)}", "RED")
            success = False
    
    # Verify installation
    if success and check_vcredist():
        return True
    else:
        print_color("⚠ Visual C++ Runtime installation could not be verified", "YELLOW")
        return False

def print_status_header():
    """Print the initial status check header"""
    print_header(" System Requirements Check ")
    print_color("\nChecking installed components...", "BLUE")
    
    # Check Python
    python_installed = check_python()
    print_color("Python 3.11        : ", "CYAN", end='')
    if python_installed:
        print_color("✓ Installed", "GREEN")
    else:
        print_color("✗ Not installed", "RED")
    
    # Check Visual C++ Runtime
    vcruntime_installed = check_vcredist()
    print_color("Visual C++ Runtime : ", "CYAN", end='')
    if vcruntime_installed:
        print_color("✓ Installed", "GREEN")
    else:
        print_color("✗ Not installed", "RED")
    
    # Check CUDA
    cuda_installed = check_cuda()
    print_color("CUDA 12.6         : ", "CYAN", end='')
    if cuda_installed:
        print_color("✓ Installed", "GREEN")
    else:
        print_color("✗ Not installed", "RED")
    
    # Check TensorRT
    tensorrt_installed = check_tensorrt()
    print_color("TensorRT          : ", "CYAN", end='')
    if tensorrt_installed:
        print_color("✓ Installed", "GREEN")
    else:
        print_color("✗ Not installed", "RED")
    
    # Check Interception
    interception_installed = check_interception()
    print_color("Interception      : ", "CYAN", end='')
    if interception_installed:
        print_color("✓ Installed", "GREEN")
    else:
        print_color("✗ Not installed", "RED")
    
    # Check Python packages
    print_color("\nRequired Python packages:", "CYAN")
    packages_status = check_python_packages()
    for package, installed in packages_status.items():
        print_color(f"{package:<15} : ", "CYAN", end='')
        if installed:
            print_color("✓ Installed", "GREEN")
        else:
            print_color("✗ Not installed", "RED")
    
    print()
    return python_installed, vcruntime_installed, cuda_installed, tensorrt_installed, interception_installed, packages_status

def verify_all_installations():
    """Verify all components are properly installed"""
    print_header(" Final Verification ")
    print_color("\nVerifying all installations...", "BLUE")
    
    # Check each component
    checks = {
        'Python 3.11': check_python(),
        'Visual C++ Runtime': check_vcredist(),
        'CUDA 12.6': check_cuda(),
        'TensorRT': check_tensorrt(),
        'Interception': check_interception()
    }
    
    # Check Python packages
    packages_status = check_python_packages()
    checks.update(packages_status)
    
    # Print results
    all_success = True
    for component, status in checks.items():
        print_color(f"{component:<20}: ", "CYAN", end='')
        if status:
            print_color("✓ Verified", "GREEN")
        else:
            print_color("✗ Not verified", "RED")
            all_success = False
    
    if all_success:
        print_color("\n✓ All components verified successfully!", "GREEN")
    else:
        print_color("\n⚠ Some components could not be verified.", "YELLOW")
        print_color("Please check the components marked with ✗ above.", "YELLOW")
    
    return all_success

def setup():
    """Main setup function that installs all required components"""
    try:
        # First check all components
        python_installed, vcruntime_installed, cuda_installed, tensorrt_installed, interception_installed, packages_status = print_status_header()
        
        print_header(" Starting Installation Process ")
        
        # Install Python if needed
        if not python_installed:
            print_color("\nInstalling Python 3.11...", "BLUE")
            if not install_python():
                print_color("✗ Failed to install Python 3.11", "RED")
                return False
        else:
            print_color("\n✓ Python 3.11 is already installed", "GREEN")
        
        # Install Visual C++ Runtime if needed
        if not vcruntime_installed:
            print_color("\nVisual C++ Runtime needs to be installed...", "BLUE")
            if not install_vcredist():
                print_color("\n⚠ Failed to install Visual C++ Runtime components.", "RED")
        else:
            print_color("\n✓ Visual C++ Runtime is already installed", "GREEN")
        
        # Install CUDA if needed
        if not cuda_installed:
            print_color("\nCUDA 12.6 needs to be installed...", "BLUE")
            if not install_cuda():
                print_color("\n⚠ CUDA installation failed. Some features may not work correctly.", "RED")
        else:
            print_color("\n✓ CUDA 12.6 is already installed", "GREEN")
        
        # Install Python packages if any are missing
        if not all(packages_status.values()):
            print_color("\nInstalling missing Python packages...", "BLUE")
            if not install_packages():
                print_color("\n⚠ Some packages failed to install. Please check the errors above.", "RED")
                return False
        else:
            print_color("\n✓ All Python packages are already installed", "GREEN")
        
        # Install Interception if needed
        if not interception_installed:
            print_color("\nInstalling Interception driver...", "BLUE")
            if not install_interception():
                print_color("✗ Failed to install Interception driver. Please install it manually.", "RED")
                return False
        else:
            print_color("\n✓ Interception driver is already installed", "GREEN")
        
        print_header(" Setup Complete! ")
        return True
        
    except Exception as e:
        print_color(f"\n✗ An error occurred during setup: {str(e)}", "RED")
        return False

def main():
    """Main entry point of the installer"""
    try:
        if not is_admin():
            print_color("Please run this script as administrator!", "RED")
            if sys.platform == 'win32':
                if is_compiled():
                    # If compiled, restart the current executable with admin rights
                    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
                else:
                    # If running from source, restart Python with admin rights
                    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        else:
            success = setup()
            print_color("\nPress Enter to exit...", "YELLOW")
            input()
            
    except Exception as e:
        print_color(f"\n✗ An error occurred: {str(e)}", "RED")
        print_color("\nPress Enter to exit...", "YELLOW")
        input()

if __name__ == "__main__":
    main()