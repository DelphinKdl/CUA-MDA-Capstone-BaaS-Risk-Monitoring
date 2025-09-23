This guide will walk you through setting up the virtual environment, installing dependencies, and running initial scripts for data preparation and development.


# (1) Clone repo and cd into it
git clone https://github.com/DelphinKdl/CUA-MDA-Capstone-BaaS-Risk-Monitoring.git
cd CUA-MDA-Capstone-BaaS-Risk-Monitoring

# (2) Create virtual environment
python -m venv .venv  # Enter
.\.venv\Scripts\activate on Windows # source venv/bin/activate  on Mac


## if you get the error message run
Set-EXecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# (3) Install dependencies
pip install -r requirements.txt -c constraints-aws.txt








