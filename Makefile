.PHONY: all partA partB partC clean

PYTHON = .venv/bin/python

all: partA partB partC

# Part A — execution simulator and slippage waterfall
partA:
	$(PYTHON) simulator/engine.py

# Part B — replication problem: follower analysis + optimisations
partB:
	$(PYTHON) replication/followers.py
	$(PYTHON) replication/distribution.py
	$(PYTHON) optimisations/vol_norm_latency_sizing.py
	$(PYTHON) optimisations/local_stop_execution.py
	$(PYTHON) replication/latency_sensitivity_check.py
	$(PYTHON) replication/anchoring_check.py

# Part C — pre-trade gate experiment
partC:
	$(PYTHON) gate/evaluate.py
	$(PYTHON) gate/c_postprocess.py

clean:
	rm -f results/*.csv results/*.png
