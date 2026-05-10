/*
 * Gallery filter island for examples.numra-rs.org.
 *
 * Lives under /public so it ships as an external script the CSP can
 * cover with `script-src 'self'`. (Astro inlines small <script> blocks
 * by default, which the strict CSP would block.) Each filter group is
 * OR within the group, AND across groups — standard faceted search.
 */
(function () {
  var list = document.getElementById('card-list');
  var empty = document.getElementById('empty-state');
  var visibleCount = document.getElementById('visible-count');
  var clearBtn = document.getElementById('clear-filters');
  if (!list || !empty || !visibleCount || !clearBtn) {
    // Page without the gallery markup; nothing to wire up.
    return;
  }

  var inputs = Array.prototype.slice.call(
    document.querySelectorAll('input[data-filter]'),
  );

  function activeValues(filter) {
    var out = new Set();
    for (var i = 0; i < inputs.length; i++) {
      var input = inputs[i];
      if (input.dataset.filter === filter && input.checked) {
        out.add(input.value);
      }
    }
    return out;
  }

  function applyFilters() {
    var eqClasses = activeValues('equation_class');
    var stiff = activeValues('stiff');
    var events = activeValues('has_events');
    var dense = activeValues('dense_output');
    var complexity = activeValues('complexity');

    var visible = 0;
    var cards = list.querySelectorAll('li.card');
    for (var i = 0; i < cards.length; i++) {
      var card = cards[i];
      var matchesClass =
        eqClasses.size === 0 || eqClasses.has(card.dataset.equationClass || '');
      var matchesStiff =
        stiff.size === 0 || stiff.has(card.dataset.stiff || 'false');
      var matchesEvents =
        events.size === 0 || events.has(card.dataset.events || 'false');
      var matchesDense =
        dense.size === 0 || dense.has(card.dataset.dense || 'false');
      var matchesComplexity =
        complexity.size === 0 || complexity.has(card.dataset.complexity || '');
      var show =
        matchesClass &&
        matchesStiff &&
        matchesEvents &&
        matchesDense &&
        matchesComplexity;
      card.hidden = !show;
      if (show) visible++;
    }
    visibleCount.textContent = String(visible);
    empty.hidden = visible !== 0;
  }

  inputs.forEach(function (input) {
    input.addEventListener('change', applyFilters);
  });
  clearBtn.addEventListener('click', function () {
    inputs.forEach(function (input) {
      input.checked = false;
    });
    applyFilters();
  });
})();
