# Numra figures — build all SVGs into the right Astro public dirs.
#
# Usage:
#   make all          # everything
#   make smoke        # render the smoke-test figure (no real data needed)
#   make home         # home-page evidence (correctness heatmap + hero bench)
#   make og           # marketing-site OG default image
#   make perf         # performance chapter charts (Phase 2)
#   make comparisons  # SciPy comparison charts (Phase 2)
#   make examples     # examples-gallery thumbnails (Phase 2)
#   make clean        # delete generated SVGs (sources untouched)
#
# Each rule runs `uv run python <script>` and emits SVG into the appropriate
# Astro public dir (so Astro serves the file as a static asset).

UV ?= uv
SITE_OUT  := ../site/public/figures
BOOK_OUT  := ../book/src/assets/figures
EX_OUT    := ../examples/public/thumbs
OG_OUT    := ../site/public/og

.PHONY: all smoke home og perf comparisons examples clean ensure-dirs

all: home og perf comparisons examples

ensure-dirs:
	@mkdir -p $(SITE_OUT) $(BOOK_OUT) $(EX_OUT) $(OG_OUT)

smoke: ensure-dirs
	$(UV) run python smoke_test.py

home: ensure-dirs
	@for f in home/*.py; do \
		[ -f "$$f" ] || continue; \
		echo "→ $$f"; \
		$(UV) run python "$$f"; \
	done

og: ensure-dirs
	@for f in og/*.py; do \
		[ -f "$$f" ] || continue; \
		echo "→ $$f"; \
		$(UV) run python "$$f"; \
	done

perf: ensure-dirs
	@for f in perf/*.py; do \
		[ -f "$$f" ] || continue; \
		echo "→ $$f"; \
		$(UV) run python "$$f"; \
	done

comparisons: ensure-dirs
	@for f in comparisons/*.py; do \
		[ -f "$$f" ] || continue; \
		echo "→ $$f"; \
		$(UV) run python "$$f"; \
	done

examples: ensure-dirs
	@for f in examples/*.py; do \
		[ -f "$$f" ] || continue; \
		echo "→ $$f"; \
		$(UV) run python "$$f"; \
	done

clean:
	rm -f $(SITE_OUT)/*.svg $(SITE_OUT)/*.provenance.txt
	rm -f $(BOOK_OUT)/*.svg $(BOOK_OUT)/*.provenance.txt
	rm -f $(EX_OUT)/*.svg   $(EX_OUT)/*.provenance.txt
	rm -f $(OG_OUT)/*.png   $(OG_OUT)/*.provenance.txt
