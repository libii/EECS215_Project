# EECS215 Fall 2024 Project

## About
A project for an algorithms class using AR/VR data

## Installation
### Install Python
[A python guide](https://realpython.com/installing-python/).
You can look for others if you like.

### Python Dependencies ###
###### Optional: Create a Virtual Environment
```
cd [repository name]
python -m venv venv
source venv/bin/activate
```
Why use a virtual environment? A virtual environment allows you to use different version of python and dependancy with conflict with other projects. 

#### Install Python Dependencies
Two options either manually or through the requirements.txt
##### Option 1: Through Requirements.txt
Recommended if the group is good at updating the requirements.txt when they add libraries to the code.

```
pip install -r requirements.txt
pip install -e .
```
Requirements.txt contains the dependancies and version for this project.

##### Option 2: Manually #####
install python libraries:
```
pip install matplotlib
pip install networkx
pip install numpy
pip install -U scikit-learn
```

## Data List
- completion_time_and_accuracy.csv
- converstation_graphs.json
- proximity_graphs.json
- shared_attention_graphs.json

---


# End of Project Checklist
- [ ] Does it work?
- [ ] Write report
- [ ] Update readme.md Dependencies section
    - [ ] requirement.txt
    - [ ] manually install code block
