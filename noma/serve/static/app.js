/* Noma playground. Talks to /v1/systemone like any client; nothing here is special server-side. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const PRESETS = {
    "Ticket triage": {
      state: "From: dana.k@corvid-supply.com\nSubject: charged twice + locked out\n\nHi, my card was charged twice for order #4471 (EUR 189.00 each, 28 Sep). I also can't log in since yesterday, the reset link says it expired. We need the duplicate refunded before our month-end close on Friday. Thanks, Dana",
      questions: {
        team: { type: "choice", instructions: "Which team should handle this ticket first?",
          criteria: { billing: "Billing and refunds", identity: "Login and account access", shipping: "Shipping and delivery", sales: "Sales and upgrades" } },
        refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
        urgency: { type: "score", instructions: "How urgent is this ticket?",
          criteria: ["Low", "Normal", "High", "Critical"] } } },
    "Agent step check": {
      state: "Task: rotate the API key for service billing-worker without downtime.\n\nStep 4 (agent): created new key bk_live_7Q... in the secrets manager.\nStep 5 (agent): updated deployment billing-worker env BILLING_KEY to the new key; rollout started.\nStep 6 tool output: rollout status: 2/3 pods ready, 1 pod CrashLoopBackOff (auth error 401 from payments API).\nStep 7 (agent, proposed): revoke the old key now.",
      questions: {
        step_ok: { type: "noul", instructions: "Did the last completed step (the rollout) succeed?" },
        safe_next: { type: "noul", instructions: "Is the proposed next action safe to run now?" },
        done: { type: "noul", instructions: "Is the overall task complete?" } } },
    "Moderation": {
      state: "Comment on a product review: \"Honestly this blender is garbage and whoever designed the lid should be fired. Mine cracked in a week. Returning it.\"",
      questions: {
        action: { type: "choice", instructions: "What should the moderation system do with this comment?",
          criteria: { allow: "Allow: harsh but acceptable criticism", review: "Send to human review", remove: "Remove: harassment or abuse" } },
        toxicity: { type: "score", instructions: "How toxic is the comment?", criteria: ["None", "Mild", "Strong", "Severe"] } } },
    "Model routing": {
      state: "User request: \"Refactor this 400-line Python module to use async IO, keep the public API unchanged, and add tests for the three network functions.\"",
      questions: {
        route: { type: "choice", instructions: "Which model tier should serve this request?",
          criteria: { small_fast: "Small fast model (simple lookups and short answers)", mid: "Mid-size model (standard writing and simple code)", frontier: "Frontier model (long, multi-step coding and reasoning)" } },
        needs_tools: { type: "noul", instructions: "Does the request need code execution or tools to verify the result?" } } },
    "Contract clause": {
      state: "12.3 Either party may terminate this Agreement for convenience on ninety (90) days' written notice to the other party. 12.4 Upon termination the Customer shall pay all fees accrued up to the effective date of termination.",
      questions: {
        termination: { type: "noul", instructions: "Does the excerpt allow termination for convenience?" },
        notice: { type: "choice", instructions: "What notice period applies to termination for convenience?",
          criteria: { d30: "30 days", d60: "60 days", d90: "90 days", none: "No notice period stated" } } } },
  };

  let questions = [];      // [{key, type, instructions, options:[[k, desc]]}]
  let lastBody = null;
  let tab = "curl";

  // ---------------------------------------------------------------- title reveal
  const title = $("#title");
  const words = title.textContent.split(" ");
  title.innerHTML = words.map((w, i) =>
    `<span class="w" style="animation-delay:${reduced ? 0 : 120 + i * 90}ms">${i === 1 ? `<em>${w}</em>` : w}</span>`
  ).join(" ");

  // ---------------------------------------------------------------- theme
  const setTheme = (t) => { document.documentElement.dataset.theme = t; try { localStorage.setItem("noma-theme", t); } catch {} };
  try { const t = localStorage.getItem("noma-theme"); if (t) setTheme(t); } catch {}
  $("#themeBtn").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

  // ---------------------------------------------------------------- status
  async function health() {
    const dot = $("#statusDot"), txt = $("#statusText");
    try {
      const r = await fetch("/v1/info");
      const i = await r.json();
      dot.className = "dot " + (i.ready ? "ok" : "");
      txt.textContent = i.ready ? `${i.model} · ${i.device}${i.fast_path ? " · fast path" : ""}` : "Warming up…";
      if (!i.ready) setTimeout(health, 1500);
    } catch { dot.className = "dot bad"; txt.textContent = "Server unreachable"; setTimeout(health, 3000); }
  }
  health();

  // ---------------------------------------------------------------- presets
  const presetBar = $("#presets");
  Object.keys(PRESETS).forEach((name, i) => {
    const b = document.createElement("button");
    b.className = "chip"; b.textContent = name; b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
    b.onclick = () => { presetBar.querySelectorAll(".chip").forEach(c => c.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true"); load(PRESETS[name]); };
    presetBar.appendChild(b);
  });

  function load(p) {
    $("#state").value = p.state;
    questions = Object.entries(p.questions).map(([key, q]) => ({
      key, type: q.type, instructions: q.instructions,
      options: q.type === "choice" ? Object.entries(q.criteria)
        : q.type === "score" ? q.criteria.map((d, i) => [String(i), d]) : [["true", "Yes"], ["false", "No"]] }));
    renderQuestions(); stateMeta(); renderCode();
    $("#answers").innerHTML = '<p class="empty">Press <kbd>Decide</kbd> or <kbd>Ctrl</kbd> + <kbd>Enter</kbd>.</p>';
    $("#latency").textContent = "";
  }

  // ---------------------------------------------------------------- questions editor
  function renderQuestions() {
    const box = $("#questions"); box.innerHTML = "";
    questions.forEach((q, qi) => {
      const el = $("#qTpl").content.firstElementChild.cloneNode(true);
      const key = $(".q-key", el); key.value = q.key; key.oninput = () => { q.key = key.value.trim(); renderCode(); };
      const ins = $(".q-ins", el); ins.value = q.instructions; ins.oninput = () => { q.instructions = ins.value; renderCode(); };
      el.querySelectorAll(".seg button").forEach(b => {
        b.setAttribute("aria-checked", b.dataset.type === q.type ? "true" : "false");
        b.onclick = () => {
          if (q.type === b.dataset.type) return;
          q.type = b.dataset.type;
          q.options = q.type === "noul" ? [["true", "Yes"], ["false", "No"]]
            : q.type === "score" ? [["0", "Low"], ["1", "Medium"], ["2", "High"]]
            : [["option_a", "First option"], ["option_b", "Second option"]];
          renderQuestions(); renderCode();
        };
      });
      $(".rm", el).onclick = () => { questions.splice(qi, 1); renderQuestions(); renderCode(); };
      const opts = $(".opts", el);
      q.options.forEach((o, oi) => {
        const row = document.createElement("div"); row.className = "opt";
        const k = document.createElement("input"); k.className = "k"; k.value = o[0]; k.setAttribute("aria-label", "Option key");
        const d = document.createElement("input"); d.value = o[1]; d.setAttribute("aria-label", "Option description");
        const x = document.createElement("button"); x.textContent = "×"; x.setAttribute("aria-label", "Remove option");
        k.disabled = q.type !== "choice"; x.hidden = q.type === "noul";
        if (q.type === "score") k.value = String(oi);
        k.oninput = () => { o[0] = k.value.trim(); renderCode(); };
        d.oninput = () => { o[1] = d.value; renderCode(); };
        x.onclick = () => { q.options.splice(oi, 1); renderQuestions(); renderCode(); };
        row.append(k, d, x); opts.appendChild(row);
      });
      const add = $(".add-opt", el); add.hidden = q.type === "noul";
      add.onclick = () => { q.options.push([q.type === "score" ? String(q.options.length) : `option_${q.options.length + 1}`, ""]); renderQuestions(); renderCode(); };
      box.appendChild(el);
    });
  }
  $("#addQ").onclick = () => {
    questions.push({ key: `q${questions.length + 1}`, type: "noul", instructions: "", options: [["true", "Yes"], ["false", "No"]] });
    renderQuestions(); renderCode();
  };

  // ---------------------------------------------------------------- request body
  function body() {
    let state = $("#state").value;
    try { const j = JSON.parse(state); if (j && typeof j === "object") state = j; } catch {}
    const qs = {};
    questions.forEach(q => {
      if (!q.key) return;
      const o = { type: q.type, instructions: q.instructions };
      if (q.type === "choice") o.criteria = Object.fromEntries(q.options.filter(x => x[0]));
      else if (q.type === "score") o.criteria = q.options.map(x => x[1]);
      else o.criteria = { true: q.options[0][1], false: q.options[1][1] };
      qs[q.key] = o;
    });
    const b = { model: "noma", state, questions: qs };
    const ref = $("#refTime").value.trim(); if (ref) b.reference_time = ref;
    return b;
  }

  function stateMeta() {
    const n = $("#state").value.trim().split(/\s+/).filter(Boolean).length;
    $("#stateMeta").textContent = `${n} word${n === 1 ? "" : "s"}`;
  }
  $("#state").addEventListener("input", () => { stateMeta(); renderCode(); });
  $("#refTime").addEventListener("input", renderCode);

  // ---------------------------------------------------------------- decide
  async function decide() {
    const btn = $("#decideBtn");
    const b = body();
    if (!Object.keys(b.questions).length) { $("#answers").innerHTML = '<p class="err">Add at least one question.</p>'; return; }
    lastBody = b;
    btn.disabled = true; btn.classList.remove("pinging"); void btn.offsetWidth; btn.classList.add("pinging");
    const t0 = performance.now();
    let res, data;
    try {
      res = await fetch("/v1/systemone", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });
      data = await res.json();
    } catch (e) { data = { detail: "Server unreachable" }; res = { ok: false }; }
    const rtt = performance.now() - t0;
    btn.disabled = false;
    if (!res.ok) { $("#answers").innerHTML = `<p class="err">${esc(data.detail || "Request failed")}</p>`; return; }
    render(data, rtt);
  }
  $("#decideBtn").onclick = decide;
  document.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); decide(); } });

  function render(data, rtt) {
    const box = $("#answers"); box.innerHTML = "";
    const model = data.noma && data.noma.model_ms;
    countUp($("#latency"), model != null ? model : rtt, v => `${v.toFixed(1)} ms model · ${rtt.toFixed(0)} ms total`);
    Object.entries(data.answers).forEach(([key, a], i) => {
      const ranked = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]);
      // Score levels stay in their own order; other types rank by probability.
      const probs = a.type === "score" ? Object.entries(a.probabilities).sort((x, y) => x[0] - y[0]) : ranked;
      const top = ranked[0][0];
      const q = questions.find(x => x.key === key);
      const label = k => { const o = q && q.options.find(x => x[0] === k); return o && o[1] ? o[1] : k; };
      const verdict = a.type === "noul" ? (a.noul >= 0.5 ? "Yes" : "No")
        : a.type === "score" ? label(String(top)) : label(a.choice);
      const ab = a.noma ? a.noma.abstain : 0, un = a.noma ? a.noma.uncertainty : 0;
      const card = document.createElement("article"); card.className = "card";
      card.style.animationDelay = `${reduced ? 0 : i * 90}ms`;
      card.innerHTML = `
        <div class="card-top"><span class="card-key">${esc(key)}</span>
          <span class="verdict">${esc(verdict)}</span>
          <span class="conf">${(a.confidence * 100).toFixed(1)}%</span></div>
        <div class="bars">${probs.map(([k, p], j) => `
          <div class="bar ${k === top ? "win" : ""}"><span class="lbl" title="${esc(label(k))}">${esc(a.type === "noul" ? (k === "true" ? "yes" : "no") : a.type === "score" ? `${k} · ${label(k)}` : k)}</span>
            <span class="track"><span class="fill" data-p="${p}"></span></span><span class="pct">${(p * 100).toFixed(1)}%</span></div>`).join("")}</div>
        <div class="signals"><span class="${ab >= 0.5 ? "warn" : ""}">abstain <b>${(ab * 100).toFixed(1)}%</b></span>
          <span class="${un >= 0.1 ? "warn" : ""}">uncertainty <b>${un.toFixed(3)}</b></span>
          ${a.type === "score" ? `<span>expected level <b>${a.score.toFixed(2)}</b></span>` : ""}</div>`;
      box.appendChild(card);
      requestAnimationFrame(() => setTimeout(() => card.querySelectorAll(".fill").forEach((f, j) => {
        f.style.transitionDelay = `${reduced ? 0 : j * 60}ms`; f.style.width = `${Math.max(1.5, f.dataset.p * 100)}%`;
      }), reduced ? 0 : 60 + i * 90));
    });
    renderCode();
  }

  function countUp(el, target, fmt) {
    if (reduced) { el.innerHTML = `<span class="lat">${fmt(target)}</span>`; return; }
    const t0 = performance.now(), dur = 520;
    const step = t => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.innerHTML = `<span class="lat">${fmt(target * e)}</span>`; if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }

  // ---------------------------------------------------------------- copy as code
  function renderCode() {
    const b = lastBody || body();
    const json = JSON.stringify(b, null, 2);
    const origin = location.origin;
    const snippets = {
      curl: `curl -s ${origin}/v1/systemone \\\n  -H "Content-Type: application/json" \\\n  -d '${json.replace(/'/g, "'\\''")}'`,
      python: `from noma import Noma\n\nmodel = Noma.from_pretrained("BlackdromeAILabs/noma")\nanswers, _ = model.decide(\n    state=${pyRepr(b.state)},\n    questions=${pyRepr(b.questions, 4)},\n)\nfor key, (probs, abstain, uncertainty) in answers.items():\n    print(key, max(probs, key=probs.get), probs)`,
      jev: `# Any Jev (TypeSafe /v1/systemone) client works unchanged: point it at Noma.\nimport requests\n\nr = requests.post("${origin}/v1/systemone", json=${pyRepr(b, 0)})\nprint(r.json()["answers"])`,
    };
    $("#codeBox").textContent = snippets[tab];
  }
  function pyRepr(v, indent = 0) {
    return JSON.stringify(v, null, 2).replace(/(?<!")\btrue\b(?!")/g, "True").replace(/(?<!")\bfalse\b(?!")/g, "False")
      .replace(/(?<!")\bnull\b(?!")/g, "None").split("\n").map((l, i) => i ? " ".repeat(indent) + l : l).join("\n");
  }
  document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
    tab = b.dataset.tab; document.querySelectorAll(".tabs button").forEach(x => x.setAttribute("aria-selected", x === b ? "true" : "false")); renderCode();
  });
  $("#copyBtn").onclick = async () => {
    try { await navigator.clipboard.writeText($("#codeBox").textContent); $("#copyBtn").textContent = "Copied"; }
    catch { $("#copyBtn").textContent = "Select and copy"; }
    setTimeout(() => $("#copyBtn").textContent = "Copy", 1400);
  };

  function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  load(PRESETS["Ticket triage"]);
})();
