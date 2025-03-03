# Make sure these targets do not reference any file named "run", "api", or "web"
.PHONY: run api web packages clean

packages:
	pip install -r requirements.txt

# Target to run only the FastAPI server
api:
	uvicorn app.main_api:app --reload

# Target to run only the Streamlit app
web:
	streamlit run app/main.py

# Combined target: run the FastAPI server (background) and the Streamlit app
run:
	@if ! command -v tmux &> /dev/null; then \
		echo "tmux not found. Installing..."; \
		sudo apt-get update && sudo apt-get install -y tmux; \
	fi
	tmux new-session -d -s my_app "uvicorn app.main_api:app --reload"
	tmux split-window -t my_app:0 -v "streamlit run app/main.py"



clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints