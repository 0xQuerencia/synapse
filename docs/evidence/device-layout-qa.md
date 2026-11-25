# Device workspace visual and interaction checks

The actual authenticated GUI server is exercised separately by
`scripts/gui_smoke.py`; its live two-process/API/WebSocket results are recorded
in `gui-http-smoke.md`. The browser rendered the real login page and then the
unchanged GUI assets through the isolated `gui_layout_preview.py` fixture.
The fixture is visibly labelled, has no credentials or live nodes, and rejects
every session mutation. It is not presented as an authenticated inference run.

Observed checks in the cloud Chromium browser:

- Desktop: complete two-column workspace, readable route/resource summaries,
  and no external font/script/image dependency.
- Mobile wrapper 360px: effective inner viewport 345px (scrollbar included),
  document scroll width 345px, so no horizontal overflow.
- Tablet wrapper 768px: document width and scroll width both 768px.
- Arrow Right selects the worker tab and focuses it; Home selects the host tab.
  Associated role-form labels and visible fields update with the selection.
- Space toggles the raw-prompt checkbox. The actual model text input accepts
  the original fixture identifier.
- Submitting the layout fixture's host form shows the explicit server rejection
  in the status/error region, preserves the form and returns controls to usable
  state. It never fabricates a connected session.
- Idle generation is disabled. Input labels, skip link, named tab group,
  role panels, output region and status updates appear in the accessibility tree.
- CSS disables transitions/animation under reduced-motion preference; the
  current UI uses no animation in its default state.

These are browser layout/keyboard checks, not an independent human usability
study or native screen-reader certification. Both remain acceptance gates.
