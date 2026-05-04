// Theme persistence — read the user's stored choice and apply it to
// <html data-theme="..."> before the body paints, to avoid a flash of
// incorrect theme. Loaded as an external file (rather than inline) so
// `script-src 'self'` covers it without an `'unsafe-inline'` exemption.
(function () {
  try {
    var stored = localStorage.getItem('numra-theme');
    if (stored === 'light' || stored === 'dark') {
      document.documentElement.setAttribute('data-theme', stored);
    }
  } catch (_) {
    /* localStorage may be blocked; fall back to system theme */
  }
})();
