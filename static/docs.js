// "?" popup shared by the chat and sim pages: is this model supported, check another model before
// downloading it, and the supported-models guide (docs/supported-models.md).
//
//   VCDocs.open("model" | "check" | "guide")
//   VCDocs.showReport(report)   open on a specific report (e.g. a model the server refused)

(() => {
  const css = `
  .vcd-scrim { position: fixed; inset: 0; background: rgba(0,0,0,.55); z-index: 50; display: flex; align-items: center; justify-content: center; padding: 16px; }
  .vcd { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; width: min(860px, 100%); max-height: 88vh;
         display: flex; flex-direction: column; box-shadow: 0 20px 60px rgba(0,0,0,.5); color: var(--text); font: 14px/1.55 var(--sans); }
  .vcd-head { display: flex; align-items: center; gap: 6px; padding: 10px 12px; border-bottom: 1px solid var(--border); }
  .vcd-head .tab { font: 12px var(--mono); padding: 5px 10px; border-radius: 6px; border: 1px solid transparent; background: none; color: var(--muted); cursor: pointer; }
  .vcd-head .tab.on { color: var(--text); border-color: var(--border); background: var(--panel-2); }
  .vcd-head .x { margin-left: auto; font: 16px var(--mono); padding: 2px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--panel-2); color: var(--text); cursor: pointer; }
  .vcd-body { padding: 18px 22px 24px; overflow-y: auto; }
  .vcd-verdict { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
  .vcd-verdict b { font: 600 16px var(--mono); }
  .vcd-verdict .yes { color: #4ade80; } .vcd-verdict .no { color: #f87171; }
  .vcd-verdict span { font: 12px var(--mono); color: var(--muted); }
  .vcd-check { display: grid; grid-template-columns: 20px 150px 1fr; gap: 8px; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
  .vcd-check .i { font-family: var(--mono); } .vcd-check .i.ok { color: #4ade80; } .vcd-check .i.no { color: #f87171; }
  .vcd-check .l { font: 12px var(--mono); color: var(--muted); padding-top: 1px; }
  .vcd-note { font: 12px var(--mono); color: var(--muted); margin-top: 14px; }
  .vcd-row { display: flex; gap: 8px; margin-bottom: 16px; }
  .vcd-row input { flex: 1; background: var(--panel-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px; font: 13px var(--mono); }
  .vcd-row button { font: 13px var(--sans); border-radius: 6px; border: 1px solid var(--accent); background: var(--accent); color: #042f2e; font-weight: 600; padding: 7px 14px; cursor: pointer; }
  .vcd-md h1 { font-size: 20px; margin: 0 0 10px; } .vcd-md h2 { font-size: 15px; margin: 24px 0 8px; color: var(--accent); }
  .vcd-md p, .vcd-md li { color: var(--text); } .vcd-md p { margin: 8px 0; } .vcd-md ul { padding-left: 20px; margin: 8px 0; }
  .vcd-md code { font: 12px var(--mono); background: var(--panel-2); padding: 1px 5px; border-radius: 3px; }
  .vcd-md pre { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; overflow-x: auto; font: 12px/1.5 var(--mono); }
  .vcd-md pre code { background: none; padding: 0; }
  .vcd-md table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0; display: block; overflow-x: auto; }
  .vcd-md th, .vcd-md td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  .vcd-md th { font: 600 11px var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .vcd-md a { color: var(--accent); }`;

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // Just enough Markdown for docs/supported-models.md: headings, paragraphs, lists, tables, code.
  function markdown(src) {
    const inline = (s) => esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    const lines = src.split("\n"), out = [];
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      if (l.startsWith("```")) {
        const code = [];
        while (++i < lines.length && !lines[i].startsWith("```")) code.push(lines[i]);
        out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
      } else if (/^#{1,3} /.test(l)) {
        const n = l.match(/^#+/)[0].length;
        out.push(`<h${n}>${inline(l.slice(n + 1))}</h${n}>`);
      } else if (l.startsWith("|")) {
        const rows = [];
        for (; i < lines.length && lines[i].startsWith("|"); i++) rows.push(lines[i]);
        i--;
        const cells = (r) => r.slice(1, -1).split("|").map((c) => c.trim());
        const [head, , ...body] = rows;
        out.push(`<table><tr>${cells(head).map((c) => `<th>${inline(c)}</th>`).join("")}</tr>` +
          body.map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("") + "</table>");
      } else if (/^- /.test(l)) {
        const items = [];
        for (; i < lines.length && /^- /.test(lines[i]); i++) items.push(`<li>${inline(lines[i].slice(2))}</li>`);
        i--;
        out.push(`<ul>${items.join("")}</ul>`);
      } else if (l.trim()) {
        const para = [l];
        while (i + 1 < lines.length && lines[i + 1].trim() && !/^(#|\||- |```)/.test(lines[i + 1])) para.push(lines[++i]);
        out.push(`<p>${inline(para.join(" "))}</p>`);
      }
    }
    return out.join("");
  }

  function reportHtml(r) {
    const verdict = r.supported
      ? `<b class="yes">✓ Supported</b>` : `<b class="no">✗ Not supported</b>`;
    return `<div class="vcd-verdict">${verdict}<span>${esc(r.model)}</span></div>` +
      r.checks.map((c) => `<div class="vcd-check"><span class="i ${c.ok ? "ok" : "no"}">${c.ok ? "✓" : "✗"}</span>
        <span class="l">${esc(c.label)}</span><span>${markdown(c.detail).replace(/^<p>|<\/p>$/g, "")}</span></div>`).join("") +
      (r.tools != null ? `<div class="vcd-note">Tool calling: ${r.tools ? "available" : "not available (chat template has no tool support)"}.</div>` : "");
  }

  let root = null, body = null, current = null;

  function build() {
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);
    root = document.createElement("div");
    root.className = "vcd-scrim";
    root.style.display = "none";
    root.innerHTML = `<div class="vcd" role="dialog" aria-label="Supported models">
      <div class="vcd-head">
        <button class="tab" data-tab="model">This model</button>
        <button class="tab" data-tab="check">Check a model</button>
        <button class="tab" data-tab="guide">Supported models guide</button>
        <button class="x" title="Close (Esc)">×</button>
      </div><div class="vcd-body"></div></div>`;
    document.body.appendChild(root);
    body = root.querySelector(".vcd-body");
    root.querySelectorAll(".tab").forEach((t) => (t.onclick = () => show(t.dataset.tab)));
    root.querySelector(".x").onclick = close;
    root.onclick = (e) => { if (e.target === root) close(); };
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && root.style.display !== "none") close(); });
  }

  function tabs(name) {
    current = name;
    root.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.dataset.tab === name));
  }

  async function show(name) {
    tabs(name);
    if (name === "model") {
      body.innerHTML = "Loading…";
      body.innerHTML = reportHtml(await (await fetch("/api/compat")).json());
    } else if (name === "check") {
      body.innerHTML = `<div class="vcd-row"><input id="vcdRepo" placeholder="Hugging Face repo id, e.g. Qwen/Qwen3-1.7B"><button id="vcdGo">Check</button></div>
        <div id="vcdOut" class="vcd-note">Reads only the model's config.json and size from the Hub. Nothing is downloaded.</div>`;
      const go = async () => {
        const id = body.querySelector("#vcdRepo").value.trim();
        if (!id) return;
        body.querySelector("#vcdOut").innerHTML = "Checking…";
        const r = await (await fetch(`/api/models/check?model=${encodeURIComponent(id)}`)).json();
        body.querySelector("#vcdOut").className = "";
        body.querySelector("#vcdOut").innerHTML = reportHtml(r);
      };
      body.querySelector("#vcdGo").onclick = go;
      body.querySelector("#vcdRepo").onkeydown = (e) => { if (e.key === "Enter") go(); };
      body.querySelector("#vcdRepo").focus();
    } else {
      body.innerHTML = "Loading…";
      body.innerHTML = `<div class="vcd-md">${markdown(await (await fetch("/docs/supported-models.md")).text())}</div>`;
    }
  }

  function open(name = "model") {
    if (!root) build();
    root.style.display = "flex";
    show(name);
  }

  function showReport(report) {
    if (!root) build();
    root.style.display = "flex";
    tabs("check");
    body.innerHTML = reportHtml(report) +
      `<div class="vcd-note">See the <a href="#" id="vcdGuide">supported models guide</a> for the full list.</div>`;
    body.querySelector("#vcdGuide").onclick = (e) => { e.preventDefault(); show("guide"); };
  }

  function close() { root.style.display = "none"; }

  window.VCDocs = { open, showReport, close };
})();
