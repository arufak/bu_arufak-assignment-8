# Define your virtual environment and flask app
VENV = venv
FLASK_APP = app.py

# Create a virtual environment and install dependencies
install:
	python3 -m venv $(VENV)
	venv\Scripts\activate && pip install -r requirements.txt

# Run the Flask application on localhost:3000
run:
	venv\Scripts\activate && set FLASK_APP=$(FLASK_APP) && flask run --host=0.0.0.0 --port=3000

# Clean up virtual environment
clean:
	if exist $(VENV) rmdir /S /Q $(VENV)

# Reinstall all dependencies
reinstall: clean install
