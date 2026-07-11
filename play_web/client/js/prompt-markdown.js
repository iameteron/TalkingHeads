(function (global) {
  function stripInlineMarkdown(text) {
    return String(text || "")
      .replace(/\*\*(.+?)\*\*/g, "$1")
      .replace(/\*(.+?)\*/g, "$1")
      .replace(/`(.+?)`/g, "$1")
      .replace(/\[(.+?)\]\(.+?\)/g, "$1")
      .trim();
  }

  function slugify(text) {
    const base = stripInlineMarkdown(text)
      .toLowerCase()
      .replace(/[^\w\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-")
      .replace(/^-|-$/g, "");
    return base || "section";
  }

  function extractSections(markdown) {
    const sections = [];
    const used = new Set();
    for (const line of String(markdown || "").split("\n")) {
      const match = /^(#{1,6})\s+(.+)$/.exec(line);
      if (!match) continue;
      const level = match[1].length;
      const title = stripInlineMarkdown(match[2]);
      let id = slugify(title);
      let suffix = 2;
      while (used.has(id)) {
        id = `${slugify(title)}-${suffix}`;
        suffix += 1;
      }
      used.add(id);
      sections.push({ id, title, level });
    }
    return sections;
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function getMarkedParser() {
    const marked = global.marked;
    if (!marked) return null;
    if (typeof marked.parse === "function") return marked.parse.bind(marked);
    if (marked.marked && typeof marked.marked.parse === "function") {
      return marked.marked.parse.bind(marked.marked);
    }
    return null;
  }

  function renderMarkdown(markdown) {
    const text = String(markdown || "");
    if (!text.trim()) return "";

    const parse = getMarkedParser();
    const purify = global.DOMPurify;
    if (!parse || !purify) {
      return `<pre class="prompt-md-fallback">${escapeHtml(text)}</pre>`;
    }

    const sections = extractSections(text);
    let headingIndex = 0;
    const renderer = new marked.Renderer();
    renderer.heading = function heading(text, level) {
      const section = sections[headingIndex];
      headingIndex += 1;
      const id = section ? section.id : slugify(stripInlineMarkdown(text));
      return `<h${level} id="${id}" class="prompt-md-heading">${text}</h${level}>`;
    };

    const html = parse(text, {
      renderer,
      gfm: true,
      breaks: true,
    });

    return purify.sanitize(html, {
      ADD_ATTR: ["id", "target", "rel"],
      ALLOWED_TAGS: [
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "br", "strong", "em", "del",
        "code", "pre",
        "ul", "ol", "li",
        "blockquote", "hr",
        "table", "thead", "tbody", "tr", "th", "td",
        "a",
      ],
    });
  }

  global.PlayWebPromptMarkdown = {
    extractSections,
    renderMarkdown,
    slugify,
  };
})(window);
