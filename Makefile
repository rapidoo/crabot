.PHONY: install test test-unit test-integration test-neo4j lint benchmark setup-neo4j clean daemon service-install service-start service-stop service-restart service-logs service-status

PLIST_NAME = com.nano-agent.daemon
PLIST_SRC  = $(CURDIR)/$(PLIST_NAME).plist
PLIST_DEST = $(HOME)/Library/LaunchAgents/$(PLIST_NAME).plist

# Install
install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,memory,telegram]"

# Tests
test:
	.venv/bin/python -m pytest tests/ -m "not neo4j" -v

test-unit:
	.venv/bin/python -m pytest tests/ -m "not integration and not neo4j" -v

test-integration:
	.venv/bin/python -m pytest tests/ -m "integration" -v

test-neo4j:
	.venv/bin/python -m pytest tests/ -m "neo4j" -v

test-all:
	.venv/bin/python -m pytest tests/ -v

# Lint
lint:
	.venv/bin/python -m mypy agent/ --ignore-missing-imports

# Benchmark
benchmark:
	.venv/bin/python benchmarks/run_benchmark.py

# Neo4j
setup-neo4j:
	docker run -d --name nano-neo4j \
		-p 7687:7687 -p 7474:7474 \
		-e NEO4J_AUTH=neo4j/password \
		neo4j:5
	@echo "Neo4j started. Browser: http://localhost:7474"

stop-neo4j:
	docker stop nano-neo4j && docker rm nano-neo4j

# Run agent
chat:
	.venv/bin/python -m agent

run:
	.venv/bin/python -m agent $(PROMPT)

# Daemon (foreground — Ctrl+C to stop)
daemon:
	.venv/bin/python -m agent --daemon

# macOS service (launchd)
service-install:
	@mkdir -p logs
	cp $(PLIST_SRC) $(PLIST_DEST)
	@echo "Service installed: $(PLIST_DEST)"
	@echo "Run 'make service-start' to start it."

service-start:
	launchctl load $(PLIST_DEST)
	@echo "Nano Agent daemon started. Check: make service-status"

service-stop:
	launchctl unload $(PLIST_DEST)
	@echo "Nano Agent daemon stopped."

service-restart: service-stop service-start

service-status:
	@launchctl list | grep nano-agent || echo "Service not running."
	@echo "---"
	@tail -5 logs/daemon-stderr.log 2>/dev/null || echo "No logs yet."

service-logs:
	@tail -f logs/daemon-stderr.log

service-uninstall: service-stop
	rm -f $(PLIST_DEST)
	@echo "Service uninstalled."

# Clean
clean:
	rm -rf .venv .pytest_cache .mypy_cache __pycache__ logs/ state/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
