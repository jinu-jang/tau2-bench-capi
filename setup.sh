# OS Dependencies

apt install portaudio19-dev

# Setup virtual env
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[all]"

# Run

# airline
tau2 run --domain airline --agent-llm azure/gpt-5.2 --user-llm azure/gpt-5.3-codex --num-trials 3 --max-concurrency 15