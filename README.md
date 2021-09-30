# Data analysis
- Document here the project: mnist_neural_network
- Description: Project Description
- Data Source:
- Type of analysis:


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Functional test with a script:

```bash
cd
mkdir tmp
cd tmp
mnist_neural_network-run
```

# Install

Go to `https://github.com/{group}/mnist_neural_network` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/mnist_neural_network.git
cd mnist_neural_network
pip install -r requirements.txt
make clean install test                # install and test
```
