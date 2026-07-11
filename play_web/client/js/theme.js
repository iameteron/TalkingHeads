(function () {
  const THEME_STORAGE_KEY = "playWebTheme";

  function resolveTheme(pref) {
    if (pref === "system" || !pref) {
      return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }
    return pref === "light" ? "light" : "dark";
  }

  function applyTheme(pref) {
    const stored = pref || getStoredThemePref();
    const resolved = resolveTheme(stored);
    document.documentElement.setAttribute("data-theme", resolved);
    document.documentElement.setAttribute("data-theme-pref", stored);
    return resolved;
  }

  function getStoredThemePref() {
    try {
      return localStorage.getItem(THEME_STORAGE_KEY) || "system";
    } catch (e) {
      return "system";
    }
  }

  function setStoredThemePref(pref) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, pref);
    } catch (e) {}
  }

  function readCssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function chartThemeColors() {
    return {
      legend: readCssVar("--chart-legend"),
      tick: readCssVar("--chart-tick"),
      grid: readCssVar("--chart-grid"),
      gridAlt: readCssVar("--chart-grid-alt"),
      control: readCssVar("--chart-control-text"),
    };
  }

  applyTheme(getStoredThemePref());

  window.PlayWebTheme = {
    THEME_STORAGE_KEY,
    resolveTheme,
    applyTheme,
    getStoredThemePref,
    setStoredThemePref,
    chartThemeColors,
  };
})();
