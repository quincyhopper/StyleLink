[Setup]
AppName=StyleLink
AppVersion=0.3.0
DefaultDirName={autopf}\StyleLink
DefaultGroupName=StyleLink
PrivilegesRequired=lowest
OutputDir=..\..\Output
OutputBaseFilename=StyleLink-Windows-Setup
Compression=lzma2
SolidCompression=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &Desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\..\dist\StyleLink\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\StyleLink"; Filename: "{app}\StyleLink.exe"
Name: "{autodesktop}\StyleLink"; Filename: "{app}\StyleLink.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\StyleLink.exe"; Description: "Launch StyleLink"; Flags: nowait postinstall skipifsilent
