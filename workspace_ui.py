"""HTML for the persistent reference workspace."""

from __future__ import annotations

import json
from typing import Any, Sequence


def _json_for_html(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def workspace_html(
    bookmarks: Sequence[dict[str, Any]],
    collections: Sequence[dict[str, Any]],
    history: Sequence[dict[str, Any]],
) -> str:
    """Render a self-contained research workspace using safe JSON data."""

    data = _json_for_html(
        {
            "bookmarks": list(bookmarks),
            "collections": list(collections),
            "history": list(history),
        }
    )
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research workspace</title>
  <style>
    :root { color-scheme:dark; --bg:#0f1117; --panel:#171a22; --soft:#20242e; --line:#343a48; --ink:#f3f5f8; --muted:#aab2c0; --accent:#8eb7ff; --green:#8de4b7; --danger:#ff9d9d; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
    button,input,textarea,select,a { font:inherit; }
    button,input,textarea,select { border:1px solid var(--line); border-radius:9px; background:#252a35; color:var(--ink); }
    button { padding:9px 12px; cursor:pointer; }
    button:hover:not(:disabled), .button:hover { background:#333a48; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    input,select { min-height:40px; padding:8px 10px; }
    textarea { width:100%; min-height:86px; padding:10px; resize:vertical; line-height:1.45; }
    a { color:var(--accent); }
    .topbar { position:sticky; top:0; z-index:20; display:flex; justify-content:space-between; align-items:center; gap:16px; padding:14px 24px; background:rgba(15,17,23,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }
    .topbar strong { font-size:1.05rem; }
    .nav { display:flex; gap:8px; }
    .button { display:inline-flex; align-items:center; min-height:38px; padding:8px 12px; border:1px solid var(--line); border-radius:9px; background:#252a35; color:var(--ink); text-decoration:none; }
    .button.primary { background:#315c9e; border-color:#477bc6; }
    .layout { display:grid; grid-template-columns:minmax(250px,310px) minmax(480px,1fr); min-height:calc(100vh - 68px); }
    .sidebar { padding:22px; border-right:1px solid var(--line); background:#14171e; }
    .sidebar section + section { margin-top:30px; }
    h1 { margin:0 0 6px; font-size:1.55rem; }
    h2 { margin:0 0 11px; font-size:.82rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
    .subtitle { margin:0 0 22px; color:var(--muted); line-height:1.45; }
    .collection-form { display:grid; gap:8px; }
    .collection-list,.history-list { display:grid; gap:7px; margin-top:12px; }
    .collection-item { display:flex; justify-content:space-between; gap:10px; color:#dce1e9; }
    .count { color:var(--muted); }
    .history-list { max-height:48vh; overflow:auto; padding-right:4px; }
    .history-item { display:block; padding:10px; border:1px solid #2e3440; border-radius:9px; background:#1c2029; text-decoration:none; }
    .history-item:hover { background:#252b37; }
    .history-query { color:var(--ink); line-height:1.35; overflow-wrap:anywhere; }
    .history-meta { margin-top:5px; color:var(--muted); font-size:.72rem; }
    .main { min-width:0; padding:24px; }
    .controls { display:grid; grid-template-columns:minmax(220px,1fr) 210px auto; gap:10px; align-items:end; margin:18px 0; }
    .field { display:grid; gap:5px; }
    .field label { color:var(--muted); font-size:.75rem; font-weight:700; text-transform:uppercase; }
    .selection-bar { position:sticky; top:68px; z-index:10; display:flex; flex-wrap:wrap; align-items:center; gap:9px; margin:0 0 16px; padding:10px; border:1px solid var(--line); border-radius:11px; background:rgba(30,34,43,.96); }
    .selection-count { margin-right:auto; color:var(--muted); }
    .cards { display:grid; gap:14px; }
    .card { display:grid; grid-template-columns:auto minmax(0,1fr); gap:13px; padding:17px; border:1px solid var(--line); border-radius:13px; background:var(--panel); }
    .card.hidden { display:none; }
    .select-passage { width:19px; height:19px; margin-top:4px; accent-color:#6ea8ff; }
    .card-head { display:flex; justify-content:space-between; align-items:flex-start; gap:14px; }
    .card h3 { margin:0; font-size:1.02rem; line-height:1.35; }
    .meta { margin:5px 0 0; color:var(--muted); font-size:.8rem; line-height:1.45; }
    .pill { flex:none; padding:4px 8px; border-radius:999px; background:#253c34; color:var(--green); font-size:.72rem; }
    .excerpt { margin:14px 0; padding:13px 14px; border-left:3px solid #d1a900; background:#20232a; color:#e8e9ec; white-space:pre-wrap; line-height:1.55; max-height:260px; overflow:auto; }
    .card-tools { display:grid; grid-template-columns:minmax(150px,220px) 1fr; gap:10px; margin-bottom:10px; }
    .card-actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    .status { min-height:1.2em; color:var(--green); font-size:.78rem; }
    .danger { color:var(--danger); }
    .empty { padding:42px 20px; border:1px dashed var(--line); border-radius:13px; color:var(--muted); text-align:center; }
    dialog { width:min(1180px,calc(100vw - 30px)); max-height:88vh; padding:0; border:1px solid var(--line); border-radius:15px; background:#14171e; color:var(--ink); box-shadow:0 30px 100px rgba(0,0,0,.7); }
    dialog::backdrop { background:rgba(5,7,11,.76); }
    .dialog-head { position:sticky; top:0; display:flex; justify-content:space-between; align-items:center; padding:15px 18px; background:#171b23; border-bottom:1px solid var(--line); }
    .dialog-head h2 { margin:0; color:var(--ink); font-size:1rem; text-transform:none; letter-spacing:0; }
    .compare-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; padding:18px; }
    .compare-card { min-width:0; padding:15px; border:1px solid var(--line); border-radius:11px; background:#20242d; }
    .compare-card h3 { margin:0 0 7px; font-size:.98rem; }
    .compare-card .excerpt { max-height:52vh; }
    @media (max-width:820px) { .layout{grid-template-columns:1fr}.sidebar{border-right:0;border-bottom:1px solid var(--line)}.controls{grid-template-columns:1fr}.card-tools{grid-template-columns:1fr}.selection-bar{top:66px}.history-list{max-height:260px} }
  </style>
</head>
<body>
  <header class="topbar">
    <strong>Research workspace</strong>
    <nav class="nav"><a class="button" href="/documents">Manage documents</a><a class="button" href="/quality">Reference quality</a><a class="button primary" href="/">Search documents</a></nav>
  </header>
  <div class="layout">
    <aside class="sidebar">
      <h1>Your reference desk</h1>
      <p class="subtitle">Organize evidence, annotate citations, and return to earlier searches.</p>
      <section>
        <h2>Collections</h2>
        <form id="collectionForm" class="collection-form">
          <input id="collectionName" maxlength="120" placeholder="New collection name" required>
          <button type="submit">Create collection</button>
          <span id="collectionStatus" class="status" aria-live="polite"></span>
        </form>
        <div id="collectionList" class="collection-list"></div>
      </section>
      <section>
        <h2>Search history</h2>
        <div id="historyList" class="history-list"></div>
      </section>
    </aside>
    <main class="main">
      <h1>Saved passages</h1>
      <p class="subtitle">Select passages to compare or export them with their citations.</p>
      <div class="controls">
        <div class="field"><label for="workspaceSearch">Search within saved passages</label><input id="workspaceSearch" type="search" placeholder="Words in excerpt, note, title, or section"></div>
        <div class="field"><label for="collectionFilter">Collection</label><select id="collectionFilter"></select></div>
        <button id="clearFilters" type="button">Clear</button>
      </div>
      <div class="selection-bar">
        <span id="selectionCount" class="selection-count">No passages selected</span>
        <button id="compareButton" type="button" disabled>Compare side by side</button>
        <a id="markdownExport" class="button" href="#" aria-disabled="true">Export Markdown</a>
        <a id="wordExport" class="button" href="#" aria-disabled="true">Export Word</a>
      </div>
      <div id="cards" class="cards"></div>
    </main>
  </div>
  <dialog id="compareDialog">
    <div class="dialog-head"><h2>Passage comparison</h2><button id="closeCompare" type="button">Close</button></div>
    <div id="compareGrid" class="compare-grid"></div>
  </dialog>
  <script id="workspaceData" type="application/json">@@DATA@@</script>
  <script>
    const state = JSON.parse(document.getElementById('workspaceData').textContent);
    const byId = id => document.getElementById(id);
    const cards = byId('cards');
    const collectionFilter = byId('collectionFilter');
    const workspaceSearch = byId('workspaceSearch');
    const selectedIds = () => [...document.querySelectorAll('.select-passage:checked')].map(input => Number(input.value));
    const escapeQuery = item => {
      const params = new URLSearchParams({q:item.query || ''});
      if (item.source_filter) params.set('source',item.source_filter);
      if (item.section_filter) params.set('section',item.section_filter);
      if (item.content_filter) params.set('type',item.content_filter);
      if (item.date_filter) params.set('date',item.date_filter);
      return `/?${params.toString()}`;
    };
    const pageLabel = item => item.page_start
      ? (item.page_end && item.page_end !== item.page_start ? `pages ${item.page_start}-${item.page_end}` : `page ${item.page_start}`)
      : '';
    const setText = (element, value) => { element.textContent = value || ''; return element; };

    function collectionOptions(selected = null) {
      const fragment = document.createDocumentFragment();
      const none = document.createElement('option'); none.value = ''; none.textContent = 'No collection'; fragment.appendChild(none);
      state.collections.forEach(item => {
        const option = document.createElement('option'); option.value = item.id; option.textContent = item.name; option.selected = Number(selected) === Number(item.id); fragment.appendChild(option);
      });
      return fragment;
    }

    function renderCollectionControls() {
      collectionFilter.replaceChildren();
      const all = document.createElement('option'); all.value = ''; all.textContent = 'All collections'; collectionFilter.appendChild(all);
      state.collections.forEach(item => {
        const option = document.createElement('option'); option.value = item.id; option.textContent = `${item.name} (${item.bookmark_count || 0})`; collectionFilter.appendChild(option);
      });
      const list = byId('collectionList'); list.replaceChildren();
      if (!state.collections.length) setText(list,'No collections yet.');
      state.collections.forEach(item => {
        const row = document.createElement('div'); row.className = 'collection-item';
        setText(row.appendChild(document.createElement('span')),item.name);
        const count = row.appendChild(document.createElement('span')); count.className = 'count'; count.textContent = item.bookmark_count || 0;
        list.appendChild(row);
      });
    }

    function renderHistory() {
      const list = byId('historyList'); list.replaceChildren();
      if (!state.history.length) { setText(list,'Your searches will appear here.'); return; }
      state.history.forEach(item => {
        const link = document.createElement('a'); link.className = 'history-item'; link.href = escapeQuery(item);
        const query = link.appendChild(document.createElement('div')); query.className = 'history-query'; query.textContent = item.query;
        const meta = link.appendChild(document.createElement('div')); meta.className = 'history-meta';
        const filters = [item.source_filter,item.section_filter,item.content_filter,item.date_filter].filter(Boolean);
        meta.textContent = `${item.result_count} result${item.result_count === 1 ? '' : 's'}${filters.length ? ' · filtered' : ''} · ${new Date(item.created_at).toLocaleString()}`;
        list.appendChild(link);
      });
    }

    function buildCard(item) {
      const card = document.createElement('article'); card.className = 'card'; card.dataset.id = item.id;
      card.dataset.search = [item.document_title,item.source_id,item.section,item.excerpt,item.note,item.collection_name,item.content_type,item.document_date].join(' ').toLocaleLowerCase();
      card.dataset.collection = item.collection_id || '';
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.className = 'select-passage'; checkbox.value = item.id; checkbox.setAttribute('aria-label','Select passage'); checkbox.addEventListener('change',updateSelection); card.appendChild(checkbox);
      const content = card.appendChild(document.createElement('div'));
      const head = content.appendChild(document.createElement('div')); head.className = 'card-head';
      const heading = head.appendChild(document.createElement('div'));
      const title = heading.appendChild(document.createElement('h3')); title.textContent = item.document_title || item.source_id;
      const meta = heading.appendChild(document.createElement('p')); meta.className = 'meta'; meta.textContent = [item.source_id,pageLabel(item),item.section,item.content_type,item.document_date].filter(Boolean).join(' · ');
      const pill = head.appendChild(document.createElement('span')); pill.className = 'pill'; pill.textContent = item.collection_name || ''; pill.hidden = !item.collection_name;
      const excerpt = content.appendChild(document.createElement('div')); excerpt.className = 'excerpt'; excerpt.textContent = item.excerpt;
      const tools = content.appendChild(document.createElement('div')); tools.className = 'card-tools';
      const select = tools.appendChild(document.createElement('select')); select.setAttribute('aria-label','Collection'); select.appendChild(collectionOptions(item.collection_id));
      const note = tools.appendChild(document.createElement('textarea')); note.placeholder = 'Add a note to this citation…'; note.value = item.note || ''; note.setAttribute('aria-label','Citation note');
      const actions = content.appendChild(document.createElement('div')); actions.className = 'card-actions';
      const save = actions.appendChild(document.createElement('button')); save.type = 'button'; save.textContent = 'Save note and collection';
      const open = actions.appendChild(document.createElement('a')); open.className = 'button'; open.href = item.citation_url || '#'; open.textContent = 'Open source';
      const remove = actions.appendChild(document.createElement('button')); remove.type = 'button'; remove.className = 'danger'; remove.textContent = 'Remove bookmark';
      const status = actions.appendChild(document.createElement('span')); status.className = 'status'; status.setAttribute('aria-live','polite');
      save.addEventListener('click',async () => {
        save.disabled = true; status.textContent = 'Saving…';
        try {
          const response = await fetch(`/workspace/api/bookmarks/${item.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({note:note.value,collection_id:select.value || null})});
          if (!response.ok) throw new Error('Save failed');
          const updated = await response.json(); Object.assign(item,updated);
          card.dataset.collection = item.collection_id || '';
          card.dataset.search = [item.document_title,item.source_id,item.section,item.excerpt,item.note,item.collection_name,item.content_type,item.document_date].join(' ').toLocaleLowerCase();
          pill.textContent = item.collection_name || ''; pill.hidden = !item.collection_name;
          status.textContent = 'Saved'; setTimeout(() => { status.textContent = ''; },1600);
        } catch (_) { status.textContent = 'Could not save'; status.classList.add('danger'); }
        finally { save.disabled = false; }
      });
      remove.addEventListener('click',async () => {
        if (!confirm('Remove this saved passage?')) return;
        const response = await fetch(`/workspace/api/bookmarks/${item.id}`,{method:'DELETE'});
        if (response.ok) { card.remove(); updateSelection(); if (!cards.children.length) renderEmpty(); }
      });
      return card;
    }

    function renderEmpty() {
      if (cards.children.length) return;
      const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = 'No saved passages yet. Open a citation and choose “Save passage”.'; cards.appendChild(empty);
    }

    function renderCards() {
      cards.replaceChildren(); state.bookmarks.forEach(item => cards.appendChild(buildCard(item))); renderEmpty();
    }

    function filterCards() {
      const query = workspaceSearch.value.trim().toLocaleLowerCase();
      const collection = collectionFilter.value;
      document.querySelectorAll('.card').forEach(card => card.classList.toggle('hidden',Boolean((query && !card.dataset.search.includes(query)) || (collection && card.dataset.collection !== collection))));
    }

    function updateSelection() {
      const ids = selectedIds(); const count = ids.length;
      byId('selectionCount').textContent = count ? `${count} passage${count === 1 ? '' : 's'} selected` : 'No passages selected';
      byId('compareButton').disabled = count < 2;
      [['markdownExport','markdown'],['wordExport','word']].forEach(([id,format]) => {
        const link = byId(id); link.href = count ? `/workspace/export/${format}?ids=${ids.join(',')}` : '#'; link.setAttribute('aria-disabled',String(!count));
      });
    }

    byId('collectionForm').addEventListener('submit',async event => {
      event.preventDefault(); const name = byId('collectionName').value.trim(); if (!name) return;
      const status = byId('collectionStatus'); status.textContent = 'Creating…';
      const response = await fetch('/workspace/api/collections',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
      const payload = await response.json();
      if (!response.ok) { status.textContent = payload.detail || 'Could not create collection'; return; }
      state.collections.push({...payload,bookmark_count:0}); state.collections.sort((a,b) => a.name.localeCompare(b.name));
      byId('collectionName').value = ''; status.textContent = 'Created'; renderCollectionControls(); renderCards(); filterCards(); setTimeout(() => { status.textContent = ''; },1600);
    });
    workspaceSearch.addEventListener('input',filterCards); collectionFilter.addEventListener('change',filterCards);
    byId('clearFilters').addEventListener('click',() => { workspaceSearch.value = ''; collectionFilter.value = ''; filterCards(); });
    byId('markdownExport').addEventListener('click',event => { if (!selectedIds().length) event.preventDefault(); });
    byId('wordExport').addEventListener('click',event => { if (!selectedIds().length) event.preventDefault(); });
    byId('compareButton').addEventListener('click',() => {
      const ids = new Set(selectedIds()); const grid = byId('compareGrid'); grid.replaceChildren();
      state.bookmarks.filter(item => ids.has(Number(item.id))).forEach(item => {
        const card = document.createElement('article'); card.className = 'compare-card';
        setText(card.appendChild(document.createElement('h3')),item.document_title || item.source_id);
        const meta = card.appendChild(document.createElement('p')); meta.className = 'meta'; meta.textContent = [pageLabel(item),item.section].filter(Boolean).join(' · ');
        const excerpt = card.appendChild(document.createElement('div')); excerpt.className = 'excerpt'; excerpt.textContent = item.excerpt;
        if (item.note) { const note = card.appendChild(document.createElement('p')); note.textContent = `Note: ${item.note}`; }
        grid.appendChild(card);
      });
      byId('compareDialog').showModal();
    });
    byId('closeCompare').addEventListener('click',() => byId('compareDialog').close());
    byId('compareDialog').addEventListener('click',event => { if (event.target === byId('compareDialog')) byId('compareDialog').close(); });

    renderCollectionControls(); renderHistory(); renderCards(); updateSelection();
  </script>
</body>
</html>""".replace("@@DATA@@", data)
