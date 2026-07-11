/**
 * Full-window Craftax world map: base tiles + OBS overlay, fog, camera pan/zoom.
 */
(function () {
  const FOG_BASE_ALPHA = 0.2;
  const FOG_VIGNETTE_ALPHA = 0.5;
  const FOG_FEATHER_TILES = 3.1;
  const FOG_INNER_EXPAND_TILES = 0.38;
  const MIN_ZOOM = 0.35;
  const MAX_ZOOM = 4;
  const ZOOM_STEP = 1.12;
  const HOVER_DWELL_MS = 2000;
  const HOVER_BLINK_PERIOD_MS = 850;
  const HOVER_RIPPLE_PERIOD_MS = 2600;
  const HOVER_RIPPLE_COUNT = 3;
  const CLICK_MOVE_THRESHOLD_PX = 5;

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  function createWorldMapRenderer(canvasEl, options) {
    const canvas = canvasEl;
    const opts = options || {};
    const onTileSelect = typeof opts.onTileSelect === "function" ? opts.onTileSelect : null;
    const onHoverChange = typeof opts.onHoverChange === "function" ? opts.onHoverChange : null;
    const onFollowChange = typeof opts.onFollowChange === "function" ? opts.onFollowChange : null;
    const onFieldClick = typeof opts.onFieldClick === "function" ? opts.onFieldClick : null;
    const onMapEpochMismatch = typeof opts.onMapEpochMismatch === "function" ? opts.onMapEpochMismatch : null;
    const ctx = canvas.getContext("2d");
    const baseCanvas = document.createElement("canvas");
    const baseCtx = baseCanvas.getContext("2d");
    const fogCanvas = document.createElement("canvas");
    const fogCtx = fogCanvas.getContext("2d");

    let worldMeta = null;
    let baseMapReady = false;
    let baseMapPending = null;
    let baseMapEpoch = null;
    let obsOverlayImg = null;
    let obsOverlayOrigin = null;
    let obsOverlayPending = null;
    let rafId = 0;
    let blockGrid = null;
    let lastCamera = { offsetX: 0, offsetY: 0 };

    let zoom = 1;
    let panX = 0;
    let panY = 0;
    let followAgent = true;
    let dragging = false;
    let didDrag = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let dragPanStartX = 0;
    let dragPanStartY = 0;

    function setFollowAgent(value) {
      if (followAgent === value) return;
      followAgent = value;
      if (onFollowChange) onFollowChange(followAgent);
    }

    let hoverTile = null;
    let hoverActive = false;
    let hoverDwellTimer = 0;
    let hoverBlinkStart = 0;

    function worldPixelSize() {
      if (!worldMeta) return { w: 0, h: 0 };
      const bp = worldMeta.block_px;
      return {
        w: worldMeta.map_w * bp,
        h: worldMeta.map_h * bp,
      };
    }

    function obsRectPx() {
      if (!worldMeta) return null;
      const bp = worldMeta.block_px;
      const [ox, oy] = worldMeta.obs_origin;
      const [oh, ow] = worldMeta.obs_dim;
      return {
        x: oy * bp,
        y: ox * bp,
        w: ow * bp,
        h: oh * bp,
      };
    }

    function agentCenterPx() {
      if (!worldMeta) return { x: 0, y: 0 };
      const bp = worldMeta.block_px;
      const [px, py] = worldMeta.player_pos;
      return {
        x: (py + 0.5) * bp,
        y: (px + 0.5) * bp,
      };
    }

    function drawSoftEllipseClear(ctx, cx, cy, rx, ry, feather) {
      ctx.save();
      ctx.translate(cx, cy);
      const scaleY = rx > 0 ? ry / rx : 1;
      ctx.scale(1, scaleY);
      const outerR = rx + feather;
      const innerR = Math.max(1, rx * 0.42);
      const grad = ctx.createRadialGradient(0, 0, innerR, 0, 0, outerR);
      grad.addColorStop(0, "rgba(255,255,255,1)");
      grad.addColorStop(0.2, "rgba(255,255,255,1)");
      grad.addColorStop(0.38, "rgba(255,255,255,0.9)");
      grad.addColorStop(0.54, "rgba(255,255,255,0.58)");
      grad.addColorStop(0.68, "rgba(255,255,255,0.24)");
      grad.addColorStop(0.82, "rgba(255,255,255,0.07)");
      grad.addColorStop(0.94, "rgba(255,255,255,0.01)");
      grad.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(0, 0, outerR, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    function rebuildFogMask() {
      if (!worldMeta) return;
      const { w, h } = worldPixelSize();
      if (w <= 0 || h <= 0) return;
      fogCanvas.width = w;
      fogCanvas.height = h;
      fogCtx.clearRect(0, 0, w, h);

      const obs = obsRectPx();
      if (!obs) return;

      const bp = worldMeta.block_px;
      const cx = obs.x + obs.w / 2;
      const cy = obs.y + obs.h / 2;
      const rx = obs.w / 2 + bp * FOG_INNER_EXPAND_TILES;
      const ry = obs.h / 2 + bp * FOG_INNER_EXPAND_TILES;
      const feather = Math.max(16, Math.round(bp * FOG_FEATHER_TILES));

      fogCtx.fillStyle = `rgba(6, 10, 16, ${FOG_BASE_ALPHA})`;
      fogCtx.fillRect(0, 0, w, h);
      fogCtx.fillStyle = `rgba(0, 0, 0, ${FOG_VIGNETTE_ALPHA})`;
      fogCtx.fillRect(0, 0, w, h);

      fogCtx.save();
      fogCtx.globalCompositeOperation = "destination-out";
      drawSoftEllipseClear(fogCtx, cx, cy, rx, ry, feather);
      fogCtx.restore();
    }

    function mapViewportInsets() {
      const root = getComputedStyle(document.documentElement);
      const stats = parseFloat(root.getPropertyValue("--stats-panel-width")) || 0;
      const operator = parseFloat(root.getPropertyValue("--operator-panel-width")) || 0;
      return { left: 0, right: stats + operator };
    }

    function visibleViewport(viewportW, viewportH) {
      const { left, right } = mapViewportInsets();
      return {
        left,
        right,
        w: Math.max(1, viewportW - left - right),
        h: viewportH,
        centerX: left + Math.max(1, viewportW - left - right) / 2,
      };
    }

    function fitToView() {
      if (!worldMeta || !canvas) return;
      const viewportW = canvas.clientWidth || canvas.width;
      const viewportH = canvas.clientHeight || canvas.height;
      const visible = visibleViewport(viewportW, viewportH);
      const { w, h } = worldPixelSize();
      if (visible.w <= 0 || viewportH <= 0 || w <= 0 || h <= 0) return;
      const obs = obsRectPx();
      const focusW = obs?.w || w;
      const focusH = obs?.h || h;
      zoom = Math.min(
        MAX_ZOOM,
        Math.max(MIN_ZOOM, Math.min(visible.w / (focusW * 1.05), viewportH / (focusH * 1.2))),
      );
      setFollowAgent(true);
      panX = 0;
      panY = 0;
    }

    function applyMapFull(b64, epoch) {
      if (!b64) return Promise.resolve();
      // Last-sent map wins: without this guard a slow PNG decode of an older
      // full map (e.g. pre-reset world during onboarding) can land after a
      // newer one and leave a stale base under fresh obs overlays and diffs.
      const job = loadImage(`data:image/png;base64,${b64}`).then((img) => {
        if (baseMapPending !== job) return;
        baseMapPending = null;
        baseCanvas.width = img.width;
        baseCanvas.height = img.height;
        baseCtx.drawImage(img, 0, 0);
        baseMapReady = true;
        baseMapEpoch = epoch != null ? epoch : null;
        rebuildFogMask();
        fitToView();
        scheduleDraw();
      });
      baseMapPending = job;
      return job;
    }

    function applyMapDiff(patches, epoch) {
      if (!patches || patches.length === 0) return Promise.resolve();
      // Wait for any in-flight full map so patches never draw onto a canvas
      // that is about to be replaced (or belongs to the previous world).
      const barrier = baseMapPending || Promise.resolve();
      return barrier.then(() => {
        if (!baseMapReady) return undefined;
        // Never patch a base that belongs to a different world snapshot.
        if (epoch != null && baseMapEpoch != null && epoch !== baseMapEpoch) return undefined;
        const bp = worldMeta?.block_px || 64;
        const jobs = patches.map((patch) =>
          loadImage(`data:image/png;base64,${patch.png_b64}`).then((img) => {
            baseCtx.drawImage(img, patch.y * bp, patch.x * bp);
            if (blockGrid && blockGrid[patch.x] && patch.block_id != null) {
              blockGrid[patch.x][patch.y] = patch.block_id;
            }
          }),
        );
        return Promise.all(jobs).then(() => {
          rebuildFogMask();
          scheduleDraw();
        });
      });
    }

    function applyMapBlocks(blocks) {
      if (Array.isArray(blocks) && blocks.length) {
        blockGrid = blocks;
      }
    }

    function blockIdAt(row, col) {
      if (!blockGrid || !blockGrid[row]) return null;
      const value = blockGrid[row][col];
      return value == null ? null : value;
    }

    function screenToTile(clientX, clientY) {
      if (!worldMeta) return null;
      const rect = canvas.getBoundingClientRect();
      const bp = worldMeta.block_px;
      const localX = clientX - rect.left;
      const localY = clientY - rect.top;
      const worldX = (localX - lastCamera.offsetX) / zoom;
      const worldY = (localY - lastCamera.offsetY) / zoom;
      const col = Math.floor(worldX / bp);
      const row = Math.floor(worldY / bp);
      if (row < 0 || col < 0 || row >= worldMeta.map_h || col >= worldMeta.map_w) {
        return null;
      }
      return { x: row, y: col };
    }

    function clearHover() {
      if (hoverDwellTimer) {
        clearTimeout(hoverDwellTimer);
        hoverDwellTimer = 0;
      }
      const wasActive = hoverActive;
      const hadTile = hoverTile;
      hoverTile = null;
      hoverActive = false;
      canvas.style.cursor = "";
      if (wasActive) scheduleDraw();
      if (hadTile && onHoverChange) onHoverChange(null);
    }

    function setHoverTile(tile) {
      if (!tile) {
        clearHover();
        return;
      }
      if (hoverTile && hoverTile.x === tile.x && hoverTile.y === tile.y) {
        return;
      }
      if (hoverDwellTimer) clearTimeout(hoverDwellTimer);
      if (onHoverChange) onHoverChange({ x: tile.x, y: tile.y });
      const hadActive = hoverActive;
      hoverTile = tile;
      hoverActive = false;
      if (hadActive) scheduleDraw();
      hoverDwellTimer = window.setTimeout(() => {
        hoverDwellTimer = 0;
        hoverActive = true;
        hoverBlinkStart = performance.now();
        canvas.style.cursor = "pointer";
        scheduleDraw();
      }, HOVER_DWELL_MS);
    }

    function obsOriginsMatch(a, b) {
      return Boolean(a && b && a[0] === b[0] && a[1] === b[1]);
    }

    function applyObsOverlay(b64, obsOrigin) {
      if (!b64) {
        obsOverlayImg = null;
        obsOverlayOrigin = null;
        scheduleDraw();
        return Promise.resolve();
      }
      const frameOrigin = Array.isArray(obsOrigin) ? [obsOrigin[0], obsOrigin[1]] : null;
      // Drop the previous overlay as soon as the agent window moves; until the
      // matching PNG arrives, draw only the base map so hit-testing stays aligned.
      if (frameOrigin && !obsOriginsMatch(obsOverlayOrigin, frameOrigin)) {
        obsOverlayImg = null;
        obsOverlayOrigin = null;
        scheduleDraw();
      }
      const pending = loadImage(`data:image/png;base64,${b64}`).then((img) => {
        if (obsOverlayPending !== pending) return;
        obsOverlayImg = img;
        obsOverlayOrigin = frameOrigin;
        scheduleDraw();
      });
      obsOverlayPending = pending;
      return pending;
    }

    function updateMeta(world) {
      if (!world) return;
      const prev = worldMeta;
      worldMeta = {
        map_h: world.map_h,
        map_w: world.map_w,
        block_px: world.block_px,
        obs_dim: world.obs_dim,
        obs_origin: world.obs_origin,
        player_pos: world.player_pos,
        stats: world.stats || {},
        inventory_items: world.inventory_items || [],
      };
      if (
        !prev
        || prev.map_h !== worldMeta.map_h
        || prev.map_w !== worldMeta.map_w
        || prev.block_px !== worldMeta.block_px
        || prev.obs_origin?.[0] !== worldMeta.obs_origin[0]
        || prev.obs_origin?.[1] !== worldMeta.obs_origin[1]
      ) {
        rebuildFogMask();
      }
    }

    function cameraOffset(viewportW, viewportH) {
      const { w, h } = worldPixelSize();
      const visible = visibleViewport(viewportW, viewportH);
      const center = agentCenterPx();
      let targetX = w / 2;
      let targetY = h / 2;
      if (followAgent && worldMeta) {
        targetX = center.x;
        targetY = center.y;
      }
      const offsetX = visible.centerX - targetX * zoom + panX;
      const offsetY = viewportH / 2 - targetY * zoom + panY;
      return { offsetX, offsetY, worldW: w, worldH: h };
    }

    function draw() {
      rafId = 0;
      if (!worldMeta || !canvas) return;
      const viewportW = canvas.clientWidth || canvas.width;
      const viewportH = canvas.clientHeight || canvas.height;
      if (viewportW <= 0 || viewportH <= 0) return;

      const dpr = window.devicePixelRatio || 1;
      const pxW = Math.max(1, Math.floor(viewportW * dpr));
      const pxH = Math.max(1, Math.floor(viewportH * dpr));
      if (canvas.width !== pxW || canvas.height !== pxH) {
        canvas.width = pxW;
        canvas.height = pxH;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, viewportW, viewportH);
      ctx.fillStyle = "#0f1116";
      ctx.fillRect(0, 0, viewportW, viewportH);

      const { offsetX, offsetY, worldW, worldH } = cameraOffset(viewportW, viewportH);
      if (worldW <= 0 || worldH <= 0) return;
      lastCamera = { offsetX, offsetY };

      ctx.save();
      ctx.translate(offsetX, offsetY);
      ctx.scale(zoom, zoom);

      if (baseCanvas.width > 0) {
        ctx.drawImage(baseCanvas, 0, 0);
      }

      if (obsOverlayImg && obsOverlayOrigin && worldMeta) {
        const bp = worldMeta.block_px;
        const [ox, oy] = obsOverlayOrigin;
        ctx.drawImage(obsOverlayImg, oy * bp, ox * bp);
      }

      if (fogCanvas.width > 0) {
        ctx.drawImage(fogCanvas, 0, 0);
      }

      if (hoverActive && hoverTile && worldMeta) {
        drawHoverHighlight();
      }

      ctx.restore();

      if (hoverActive && hoverTile) {
        scheduleDraw();
      }
    }

    function drawHoverHighlight() {
      const bp = worldMeta.block_px;
      const x = hoverTile.y * bp;
      const y = hoverTile.x * bp;
      const cx = x + bp / 2;
      const cy = y + bp / 2;
      const elapsed = performance.now() - hoverBlinkStart;
      const maxRadius = bp * 0.55;

      ctx.save();

      // Soft breathing glow on the tile so it stays clearly selected.
      const pulse = (Math.sin((elapsed / HOVER_BLINK_PERIOD_MS) * Math.PI * 2) + 1) / 2;
      ctx.fillStyle = `rgba(120, 220, 255, ${0.10 + pulse * 0.12})`;
      ctx.fillRect(x, y, bp, bp);

      // Concentric rings expanding from the center, like ripples on water.
      ctx.lineWidth = Math.max(1.5, bp * 0.045);
      for (let i = 0; i < HOVER_RIPPLE_COUNT; i++) {
        const t = ((elapsed / HOVER_RIPPLE_PERIOD_MS) + i / HOVER_RIPPLE_COUNT) % 1;
        // Ease-out radius so rings glide outward and gently slow down.
        const radius = (1 - (1 - t) * (1 - t)) * maxRadius;
        if (radius <= 0.5) continue;
        // Fade in quickly, then fade out as the ring expands.
        const alpha = Math.sin(t * Math.PI) * 0.7;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(170, 235, 255, ${alpha})`;
        ctx.stroke();
      }

      ctx.restore();
    }

    function scheduleDraw() {
      if (rafId) return;
      rafId = requestAnimationFrame(draw);
    }

    function resize() {
      scheduleDraw();
    }

    async function applyFrame(frame) {
      const world = frame?.world;
      if (!world) return worldMeta;
      updateMeta(world);
      const epoch = world.map_epoch != null ? world.map_epoch : null;
      const tasks = [];
      if (world.map_full_b64) {
        applyMapBlocks(world.map_blocks);
        tasks.push(applyMapFull(world.map_full_b64, epoch));
      } else if (
        epoch != null
        && !baseMapPending
        && (!baseMapReady || (baseMapEpoch != null && epoch !== baseMapEpoch))
      ) {
        // Our base is missing or belongs to an older world (e.g. the world was
        // regenerated by an HTTP config save this client never rendered).
        if (onMapEpochMismatch) onMapEpochMismatch(epoch);
      }
      if (world.map_diff?.length && (baseMapReady || baseMapPending)) {
        tasks.push(applyMapDiff(world.map_diff, epoch));
      }
      tasks.push(applyObsOverlay(world.obs_overlay_b64, world.obs_origin));
      await Promise.all(tasks);
      if (followAgent) scheduleDraw();
      else scheduleDraw();
      return worldMeta;
    }

    function recenter() {
      setFollowAgent(true);
      panX = 0;
      panY = 0;
      fitToView();
      scheduleDraw();
    }

    function zoomBy(factor) {
      setFollowAgent(false);
      zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom * factor));
      scheduleDraw();
    }

    function onWheel(event) {
      event.preventDefault();
      const factor = event.deltaY > 0 ? 1 / ZOOM_STEP : ZOOM_STEP;
      zoomBy(factor);
    }

    function onPointerDown(event) {
      if (event.button !== 0) return;
      dragging = true;
      didDrag = false;
      dragStartX = event.clientX;
      dragStartY = event.clientY;
      dragPanStartX = panX;
      dragPanStartY = panY;
      canvas.setPointerCapture?.(event.pointerId);
    }

    function onPointerMove(event) {
      if (dragging) {
        const dx = event.clientX - dragStartX;
        const dy = event.clientY - dragStartY;
        if (!didDrag && Math.hypot(dx, dy) > CLICK_MOVE_THRESHOLD_PX) {
          didDrag = true;
          setFollowAgent(false);
          clearHover();
        }
        if (didDrag) {
          panX = dragPanStartX + dx;
          panY = dragPanStartY + dy;
          scheduleDraw();
        }
        return;
      }
      setHoverTile(screenToTile(event.clientX, event.clientY));
    }

    function onPointerUp(event) {
      if (!dragging) return;
      dragging = false;
      canvas.releasePointerCapture?.(event.pointerId);
      if (!didDrag) {
        onFieldClick?.();
        const tile = screenToTile(event.clientX, event.clientY);
        if (tile && onTileSelect) {
          const rect = canvas.getBoundingClientRect();
          const bp = worldMeta.block_px;
          const screenX = rect.left + lastCamera.offsetX + (tile.y + 0.5) * bp * zoom;
          const screenY = rect.top + lastCamera.offsetY + (tile.x + 0.5) * bp * zoom;
          onTileSelect({
            x: tile.x,
            y: tile.y,
            blockId: blockIdAt(tile.x, tile.y),
            screenX,
            screenY,
            tileScreenSize: bp * zoom,
          });
        }
      }
    }

    function onPointerLeave() {
      canvas.style.cursor = "";
      clearHover();
    }

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerLeave);

    return {
      applyFrame,
      resize,
      recenter,
      hasBaseMap: () => baseMapReady,
      isFollowing: () => followAgent,
      zoomIn: () => zoomBy(ZOOM_STEP),
      zoomOut: () => zoomBy(1 / ZOOM_STEP),
      scheduleDraw,
      destroy() {
        if (rafId) cancelAnimationFrame(rafId);
        if (hoverDwellTimer) clearTimeout(hoverDwellTimer);
        canvas.removeEventListener("wheel", onWheel);
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
        canvas.removeEventListener("pointerleave", onPointerLeave);
      },
    };
  }

  window.PlayWorldMap = { createWorldMapRenderer };
})();
