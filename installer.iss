[Setup]
AppName=Reactive CCTV System
AppVersion=1.0
DefaultDirName=C:\ReactiveCCTV
DefaultGroupName=Reactive CCTV System
OutputBaseFilename=ReactiveCCTV_Installer
Compression=lzma
SolidCompression=yes
OutputDir=installer_output
SetupIconFile=experimental-logo.ico

[Files]
; Launcher exe
Source: "dist\ReactiveCCTV.exe"; DestDir: "{app}"

; Icon
Source: "experimental logo.ico"; DestDir: "{app}"

; Main scripts
Source: "scripts\*"; DestDir: "{app}\scripts"; Flags: recursesubdirs

; Core files
Source: "requirements.txt"; DestDir: "{app}"
Source: "main.py"; DestDir: "{app}"
Source: "authorization_config_main.py"; DestDir: "{app}"
Source: "authorization_map.json"; DestDir: "{app}"
Source: "yolov8n.pt"; DestDir: "{app}"

; Folders
Source: "models\*"; DestDir: "{app}\models"; Flags: recursesubdirs
Source: "config\*"; DestDir: "{app}\config"; Flags: recursesubdirs
Source: "utils\*"; DestDir: "{app}\utils"; Flags: recursesubdirs
Source: "face_recognition\*"; DestDir: "{app}\face_recognition"; Flags: recursesubdirs
Source: "datasets\*"; DestDir: "{app}\datasets"; Flags: recursesubdirs

; Batch files
Source: "install.bat"; DestDir: "{app}"

[Icons]
Name: "{group}\Reactive CCTV System"; Filename: "{app}\ReactiveCCTV.exe"; IconFilename: "{app}\experimental logo.ico"
Name: "{group}\Install Dependencies"; Filename: "{app}\install.bat"
Name: "{commondesktop}\Reactive CCTV System"; Filename: "{app}\ReactiveCCTV.exe"; IconFilename: "{app}\experimental logo.ico"

[Run]
Filename: "{app}\install.bat"; Description: "Install Python dependencies"; Flags: postinstall waituntilterminated