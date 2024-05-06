echo ON

%PYTHON% setup.py install --prefix=%PREFIX%
if errorlevel 1 exit 
    
echo OFF