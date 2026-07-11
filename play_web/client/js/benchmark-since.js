(function initBenchmarkSince(global) {
  const STORAGE_KEY = "playWebBenchmarkSince";
  const WORLD_MODE_STORAGE_KEY = "playWebBenchmarkWorldMode";
  const PANEL_COLLAPSED_STORAGE_KEY = "playWebBenchmarkPanelCollapsed";
  const VALID_WORLD_MODES = new Set(["craftax", "exo-planet"]);
  const DEFAULT_BENCHMARK_WORLD_MODE = "exo-planet";

  function loadBenchmarkSince() {
    try {
      const raw = global.localStorage?.getItem(STORAGE_KEY);
      return raw && /^\d{4}-\d{2}-\d{2}$/.test(raw) ? raw : "";
    } catch (_err) {
      return "";
    }
  }

  function saveBenchmarkSince(value) {
    const next = String(value || "").trim();
    try {
      if (next) global.localStorage?.setItem(STORAGE_KEY, next);
      else global.localStorage?.removeItem(STORAGE_KEY);
    } catch (_err) {
      /* ignore quota / private mode */
    }
  }

  function buildCampaignBenchmarkUrl(apiBase, since) {
    const base = String(apiBase || "").replace(/\/$/, "");
    const raw = String(since || "").trim();
    if (!raw) return `${base}/campaign_benchmark`;
    return `${base}/campaign_benchmark?since=${encodeURIComponent(raw)}`;
  }

  function loadBenchmarkWorldMode(fallback = DEFAULT_BENCHMARK_WORLD_MODE) {
    try {
      const raw = global.localStorage?.getItem(WORLD_MODE_STORAGE_KEY);
      if (raw && VALID_WORLD_MODES.has(raw)) return raw;
    } catch (_err) {
      /* ignore */
    }
    const next = String(fallback || DEFAULT_BENCHMARK_WORLD_MODE).trim();
    return VALID_WORLD_MODES.has(next) ? next : DEFAULT_BENCHMARK_WORLD_MODE;
  }

  function saveBenchmarkWorldMode(value) {
    const next = String(value || "").trim();
    try {
      if (VALID_WORLD_MODES.has(next)) global.localStorage?.setItem(WORLD_MODE_STORAGE_KEY, next);
      else global.localStorage?.removeItem(WORLD_MODE_STORAGE_KEY);
    } catch (_err) {
      /* ignore quota / private mode */
    }
  }

  function loadBenchmarkPanelCollapsed() {
    try {
      return global.localStorage?.getItem(PANEL_COLLAPSED_STORAGE_KEY) === "1";
    } catch (_err) {
      return false;
    }
  }

  function saveBenchmarkPanelCollapsed(collapsed) {
    try {
      if (collapsed) global.localStorage?.setItem(PANEL_COLLAPSED_STORAGE_KEY, "1");
      else global.localStorage?.removeItem(PANEL_COLLAPSED_STORAGE_KEY);
    } catch (_err) {
      /* ignore quota / private mode */
    }
  }

  function normalizeBenchmarkExtendedRow(row) {
    if (!row || typeof row !== "object") return row;
    return {
      ...row,
      exploration_rt_max_label: row.exploration_rt_max_label ?? row.kara_rt_max_label ?? "—",
      exploration_rt_max_icon: row.exploration_rt_max_icon ?? row.kara_rt_max_icon ?? "",
      exploration_q_max: row.exploration_q_max ?? row.kara_q_max ?? "—",
      exploration_runs: row.exploration_runs ?? row.kara_runs ?? 0,
      deployment: row.deployment ?? row.dusa ?? {},
    };
  }

  function normalizeBenchmarkWorldMode(block) {
    if (!block || typeof block !== "object") return block;
    const deploymentTasks = block.deployment_tasks ?? block.dusa_tasks ?? [];
    const extended = Array.isArray(block.extended)
      ? block.extended.map(normalizeBenchmarkExtendedRow)
      : [];
    const compact = Array.isArray(block.compact) ? block.compact : [];
    return {
      ...block,
      exploration_runs: Number(block.exploration_runs ?? block.kara_runs ?? 0),
      deployment_tasks: Array.isArray(deploymentTasks) ? deploymentTasks : [],
      deployment_task_keys: block.deployment_task_keys ?? block.dusa_task_keys ?? [],
      deployment_tests: block.deployment_tests ?? block.dusa_tests ?? 0,
      extended,
      compact,
    };
  }

  function normalizeCampaignBenchmarkPayload(data) {
    if (!data || typeof data !== "object") return data;
    const worldModes = {};
    for (const [mode, block] of Object.entries(data.world_modes || {})) {
      worldModes[mode] = normalizeBenchmarkWorldMode(block);
    }
    return { ...data, world_modes: worldModes };
  }

  global.PlayWebBenchmark = {
    STORAGE_KEY,
    WORLD_MODE_STORAGE_KEY,
    PANEL_COLLAPSED_STORAGE_KEY,
    DEFAULT_BENCHMARK_WORLD_MODE,
    loadBenchmarkSince,
    saveBenchmarkSince,
    loadBenchmarkWorldMode,
    saveBenchmarkWorldMode,
    loadBenchmarkPanelCollapsed,
    saveBenchmarkPanelCollapsed,
    buildCampaignBenchmarkUrl,
    normalizeBenchmarkExtendedRow,
    normalizeBenchmarkWorldMode,
    normalizeCampaignBenchmarkPayload,
  };
})(window);
