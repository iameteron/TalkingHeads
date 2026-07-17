(function () {
  const NAV = [
    {
      id: "env-agent",
      label: "Env Agent",
      href: "env-agent.html",
      children: [
        { id: "exo-planet", label: "exo-planet prompt", href: "exo-planet.html" },
        { id: "crafter", label: "crafter prompt", href: "crafter.html" },
        { id: "arc-ls20", label: "arc-agi3 ls20", href: "arc-ls20.html" },
        { id: "arc-ar25", label: "arc-agi3 ar25", href: "arc-ar25.html" },
      ],
    },
    {
      id: "operator-agent",
      label: "Operator Agent",
      href: "operator-agent.html",
      children: [
        { id: "goal-expert", label: "Goal expert", href: "goal-expert.html" },
        { id: "state-expert", label: "State expert", href: "state-expert.html" },
        { id: "mechanics-expert", label: "Mechanics expert", href: "mechanics-expert.html" },
        { id: "action-expert", label: "Action expert", href: "action-expert.html" },
      ],
    },
  ];

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderNav(activeId) {
    return NAV.map((group) => {
      const parentCurrent = activeId === group.id ? ' aria-current="page"' : "";
      const kids = group.children
        .map((child) => {
          const current = activeId === child.id ? ' aria-current="page"' : "";
          return `<li><a href="${child.href}"${current}>${escapeHtml(child.label)}</a></li>`;
        })
        .join("");
      return `
        <div class="nav-group">
          <a class="nav-parent" href="${group.href}"${parentCurrent}>${escapeHtml(group.label)}</a>
          <ul class="nav-children">${kids}</ul>
        </div>`;
    }).join("");
  }

  function mountChrome(activeId) {
    const root = document.getElementById("prompt-root");
    if (!root) return;

    const mainHtml = root.innerHTML;
    root.innerHTML = `
      <a class="skip-link" href="#main">Skip to content</a>
      <header class="topbar">
        <a class="brand" href="../index.html">
          <span class="brand-mark" aria-hidden="true"></span>
          TalkingHeads
        </a>
        <nav class="topbar-links" aria-label="Site">
          <a href="../index.html">Home</a>
          <a href="env-agent.html" aria-current="page">Prompts</a>
          <a href="https://github.com/iameteron/TalkingHeads">Code</a>
        </nav>
        <button class="menu-toggle" type="button" aria-expanded="false" aria-controls="prompt-sidebar" id="menuToggle">
          Menu
        </button>
      </header>
      <div class="sidebar-backdrop" id="sidebarBackdrop" hidden></div>
      <div class="shell">
        <aside class="sidebar" id="prompt-sidebar" aria-label="Prompt navigation">
          <p class="sidebar-label">Prompt library</p>
          ${renderNav(activeId)}
        </aside>
        <main class="content" id="main">
          <div class="content-inner">
            ${mainHtml}
          </div>
        </main>
      </div>
    `;

    const toggle = document.getElementById("menuToggle");
    const backdrop = document.getElementById("sidebarBackdrop");
    function closeNav() {
      document.body.classList.remove("nav-open");
      if (toggle) toggle.setAttribute("aria-expanded", "false");
      if (backdrop) backdrop.hidden = true;
    }
    function openNav() {
      document.body.classList.add("nav-open");
      if (toggle) toggle.setAttribute("aria-expanded", "true");
      if (backdrop) backdrop.hidden = false;
    }
    if (toggle) {
      toggle.addEventListener("click", () => {
        if (document.body.classList.contains("nav-open")) closeNav();
        else openNav();
      });
    }
    if (backdrop) backdrop.addEventListener("click", closeNav);
    document.querySelectorAll(".sidebar a").forEach((link) => {
      link.addEventListener("click", closeNav);
    });
  }

  function getMarkedParser() {
    const marked = window.marked;
    if (!marked) return null;
    if (typeof marked.parse === "function") return marked.parse.bind(marked);
    if (marked.marked && typeof marked.marked.parse === "function") {
      return marked.marked.parse.bind(marked.marked);
    }
    return null;
  }

  function renderMarkdown(markdown) {
    const text = String(markdown || "");
    const parse = getMarkedParser();
    const purify = window.DOMPurify;
    if (!parse || !purify) {
      return `<pre class="prompt-md-fallback">${escapeHtml(text)}</pre>`;
    }
    const html = parse(text, { gfm: true, breaks: false });
    return purify.sanitize(html);
  }

  async function loadPromptMarkdown(src, targetId) {
    const target = document.getElementById(targetId || "promptMarkdown");
    if (!target) return;
    target.innerHTML = `<div class="status">Loading prompt…</div>`;
    try {
      const res = await fetch(src, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      target.innerHTML = `<div class="prompt-md">${renderMarkdown(text)}</div>`;
      target.dataset.raw = text;
    } catch (err) {
      target.innerHTML = `<div class="status error">Could not load prompt: ${escapeHtml(
        err && err.message ? err.message : String(err)
      )}</div>`;
    }
  }

  function bindCopyButton() {
    const btn = document.getElementById("copyPromptBtn");
    const target = document.getElementById("promptMarkdown");
    if (!btn || !target) return;
    btn.addEventListener("click", async () => {
      const raw = target.dataset.raw || target.innerText || "";
      try {
        await navigator.clipboard.writeText(raw);
        const prev = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => {
          btn.textContent = prev;
        }, 1200);
      } catch (_) {
        btn.textContent = "Copy failed";
      }
    });
  }

  window.TalkingHeadsPrompts = {
    mountChrome,
    loadPromptMarkdown,
    bindCopyButton,
  };
})();
