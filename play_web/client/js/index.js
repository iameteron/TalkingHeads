const { createApp, ref, computed, onMounted, onUnmounted, nextTick, watch } = Vue;

function playFramesStepByStep(frames, delayMs, setFrame) {
  if (!frames || frames.length === 0) return;
  let i = 0;
  setFrame(frames[0]);
  if (frames.length === 1) return;
  const iv = setInterval(() => {
    i += 1;
    if (i >= frames.length) {
      clearInterval(iv);
      return;
    }
    setFrame(frames[i]);
  }, delayMs);
}

const BENCH_RENDER_SLOTS = 3;

const HF_INFERENCE_PROVIDER_SUFFIXES = new Set([
  "nscale",
  "novita",
  "featherless-ai",
  "featherless",
  "together",
  "fireworks-ai",
  "hyperbolic",
  "groq",
  "cerebras",
  "sambanova",
  "cohere",
  "replicate",
]);

function isHfHubModelId(modelId) {
  const name = String(modelId || "").trim();
  if (!name || name.startsWith("./") || name.startsWith("/")) return false;
  return name.includes("/");
}

function stripHfInferenceProviderSuffix(modelId) {
  const name = String(modelId || "").trim();
  if (!name || !name.includes(":")) return name;
  const idx = name.lastIndexOf(":");
  const base = name.slice(0, idx);
  const suffix = name.slice(idx + 1).toLowerCase();
  if (!base || !HF_INFERENCE_PROVIDER_SUFFIXES.has(suffix)) return name;
  return base;
}

function ensureHfInferenceProviderSuffix(modelId, provider = "nscale") {
  const name = String(modelId || "").trim();
  if (!isHfHubModelId(name) || !name) return name;
  if (name.includes(":")) {
    const idx = name.lastIndexOf(":");
    const suffix = name.slice(idx + 1).toLowerCase();
    if (HF_INFERENCE_PROVIDER_SUFFIXES.has(suffix)) return name;
  }
  return `${name}:${provider}`;
}

function adaptModelIdForGateway(modelId, gateway) {
  const mode = String(gateway || "openrouter").toLowerCase() === "hub" ? "hub" : "openrouter";
  if (mode === "openrouter") {
    const stripped = stripHfInferenceProviderSuffix(modelId);
    if (!isHfHubModelId(stripped)) return stripped;
    const slash = stripped.indexOf("/");
    if (slash < 0) return stripped.toLowerCase();
    return `${stripped.slice(0, slash).toLowerCase()}/${stripped.slice(slash + 1).toLowerCase()}`;
  }
  if (mode === "hub") return ensureHfInferenceProviderSuffix(modelId);
  return String(modelId || "").trim();
}

const CRAFTAX_TO_EXO_MEGAPROMPT = {
  dialog: "exo_reasoning_or_ask_help",
  no_dialog: "exo_no_dialog",
  reasoning_or_ask_path: "exo_reasoning_or_ask_path",
  reasoning_or_ask_help: "exo_reasoning_or_ask_help",
  database_formulation: "exo_database_formulation",
  database_formulation_deployment: "exo_database_formulation_deployment",
};

const EXO_TO_CRAFTAX_MEGAPROMPT = Object.fromEntries(
  Object.entries(CRAFTAX_TO_EXO_MEGAPROMPT).map(([craftax, exo]) => [exo, craftax]),
);

const CRAFTAX_MEGAPROMPT_DEFAULTS = new Set([
  "dialog",
  "no_dialog",
  "reasoning_or_ask_path",
  "reasoning_or_ask_help",
  "database_formulation",
  "database_formulation_deployment",
]);

const EXO_MEGAPROMPT_DEFAULTS = new Set([
  "exo_database_formulation",
  "exo_database_formulation_deployment",
  "exo_reasoning_or_ask_path",
  "exo_reasoning_or_ask_help",
  "exo_no_dialog",
]);

const ARC_MEGAPROMPT_DEFAULTS = new Set([
  "arc_2_image",
  "arc_grid",
  "arc_grid_image",
  "arc_image",
]);

const DEFAULT_CRAFT_AGENT_MODEL = "qwen/qwen3-next-80b-a3b-instruct";
const DEFAULT_COMPANION_MAX_TICKS_PER_TASK = 150;
const DEFAULT_ARC_COMPANION_MAX_TICKS_PER_TASK = 120;

const ACTIVE_AGENT_MODEL_PRESETS_CRAFT = [
  {
    id: DEFAULT_CRAFT_AGENT_MODEL,
    label: "Qwen3 Next 80B",
    note: "Default for Crafter and Exo-planet; faster text baseline for companion campaigns.",
  },
  {
    id: "qwen/qwen3-235b-a22b-2507",
    label: "Qwen3 235B",
    note: "Larger Qwen3 when you need maximum quality and can wait longer per step.",
  },
];

const ACTIVE_AGENT_MODEL_PRESETS_ARC = [
  {
    id: "anthropic/claude-sonnet-4.5",
    label: "Claude Sonnet 4.5",
    note: "Default premium vision agent for ARC.",
  },
  {
    id: "qwen/qwen2.5-vl-72b-instruct",
    label: "Qwen2.5-VL 72B",
    note: "Vision baseline for ARC frame and grid+image observations.",
  },
  {
    id: "openai/gpt-4o-mini",
    label: "GPT-4o mini",
    note: "Cheap OpenAI multimodal baseline for ARC image observations.",
  },
  {
    id: "openai/gpt-5-mini",
    label: "GPT-5 mini",
    note: "Compact reasoning model with vision for harder ARC puzzles.",
  },
  {
    id: "qwen/qwen3-next-80b-a3b-instruct",
    label: "Qwen3 Next 80B",
    note: "Text-only; use only with arc_grid observations.",
  },
];

function activeAgentModelPresetsForGameKind(gameKind) {
  return String(gameKind || "") === "arc_agi"
    ? ACTIVE_AGENT_MODEL_PRESETS_ARC
    : ACTIVE_AGENT_MODEL_PRESETS_CRAFT;
}

function isExoMegapromptConfig(configName) {
  return String(configName || "").startsWith("exo_");
}

function isArcMegapromptConfig(configName) {
  return ARC_MEGAPROMPT_DEFAULTS.has(String(configName || "").trim());
}

function worldModeFromExoEnabled(exoEnabled) {
  return exoEnabled ? "exo-planet" : "craftax";
}

function defaultMegapromptForWorldMode(worldMode, gameKind = "craftax") {
  if (gameKind === "arc_agi") return "arc_grid";
  return worldMode === "exo-planet" ? "exo_database_formulation" : "database_formulation";
}

function coerceMegapromptForWorldMode(configName, worldMode, gameKind = "craftax") {
  let normalized = String(configName || "").trim();
  if (!normalized) normalized = "dialog";
  if (gameKind === "arc_agi") {
    return isArcMegapromptConfig(normalized) ? normalized : "arc_grid";
  }
  const exoMode = worldMode === "exo-planet";
  if (isArcMegapromptConfig(normalized)) {
    return exoMode ? "exo_database_formulation" : "database_formulation";
  }
  const isExoConfig = isExoMegapromptConfig(normalized);
  if (exoMode && !isExoConfig) {
    const mapped = CRAFTAX_TO_EXO_MEGAPROMPT[normalized];
    if (mapped) return mapped;
    const candidate = `exo_${normalized}`;
    if (EXO_MEGAPROMPT_DEFAULTS.has(candidate)) return candidate;
    return "exo_database_formulation";
  }
  if (!exoMode && isExoConfig) {
    const mapped = EXO_TO_CRAFTAX_MEGAPROMPT[normalized];
    if (mapped) return mapped;
    const base = normalized.startsWith("exo_") ? normalized.slice(4) : normalized;
    if (CRAFTAX_MEGAPROMPT_DEFAULTS.has(base)) return base;
    return "database_formulation";
  }
  return normalized;
}

function companionBenchFrameSrc(frame) {
  if (!frame || !frame.png_b64) return "";
  return "data:image/png;base64," + frame.png_b64;
}

function formatLogAnswer(m) {
  if (m.kind === "human_agent") return "Agent: " + (m.answer || "");
  if (m.answer && (m.ok || !m.error)) return m.answer;
  if (m.error) return "ERROR: " + m.error;
  if (m.answer) return m.answer;
  return "ERROR: unknown";
}

function logMessageIsError(m) {
  if (m.pending) return false;
  if (m.kind === "agent_action" || m.kind === "system" || m.kind === "operator_notice") return false;
  if (m.answer && !m.error) return false;
  return !m.ok;
}

function logMessageHeader(m) {
  if (m.kind === "oracle") return "Q:";
  if (m.kind === "agent_message") return m.question && m.question.trim() ? "Agent asks:" : "Agent:";
  return "";
}

function logMessageReasoning(m) {
  return String((m && m.reasoning) || "").trim();
}

const AI_OPERATOR_DISPLAY_NAME = "TalkingHeads";
const AI_OPERATOR_AVATAR_SRC = "./assets/talkingheads-avatar.png";

function showHumanOperatorIdentityForMessage(m, interactionMode) {
  if (interactionMode !== "human") return false;
  if (!m || m.pending) return false;
  if (m.kind === "agent_message" && m.answer) return true;
  if (m.kind === "human_agent" && m.answer) return true;
  return false;
}

function showAiOperatorIdentityForMessage(m, interactionMode) {
  if (interactionMode === "human") return false;
  if (!m || m.pending) return false;
  if (m.kind === "oracle" && m.answer) return true;
  if (m.kind === "agent_message" && m.answer) return true;
  return false;
}

function showOperatorAuthorIdentityForMessage(m, interactionMode) {
  return showHumanOperatorIdentityForMessage(m, interactionMode)
    || showAiOperatorIdentityForMessage(m, interactionMode);
}

function findLastPendingAgentMessageIndex(messagesArr) {
  for (let i = messagesArr.length - 1; i >= 0; i--) {
    const entry = messagesArr[i];
    if (entry.pending && entry.kind === "agent_message") return i;
  }
  return -1;
}

function findAgentMessageIndexByQuestion(messagesArr, question) {
  const normalized = String(question || "").trim();
  if (!normalized) return -1;
  for (let i = messagesArr.length - 1; i >= 0; i--) {
    const entry = messagesArr[i];
    if (
      entry.kind === "agent_message"
      && (entry.question === normalized || entry.rawQuestion === normalized)
    ) return i;
  }
  return -1;
}

// Hide the "Agent is thinking" banner this long after each real agent step.
const AGENT_THINKING_STALL_MS = 2000;

function useWebSocket(url, { onKnowledgeUpdated, onCompanionResearchComplete, onAchievementDiscovered, onAgentPromptChanged } = {}) {
  function notifyKnowledgeUpdated(msg) {
    if (msg && msg.knowledge_updated && typeof onKnowledgeUpdated === "function") {
      onKnowledgeUpdated(msg);
    }
  }
  function notifyAgentPromptChanged() {
    if (typeof onAgentPromptChanged === "function") {
      onAgentPromptChanged();
    }
  }
  const ws = ref(null);
  const status = ref("disconnected");
  const reward = ref(0);
  const done = ref(false);
  const playerPosition = ref("");
  const currentFrame = ref(null);
  const agentObservation = ref("");
  const agentReasoning = ref("");
  const messages = ref([]);
  const campaignState = ref(null);
  const companionResearchActive = ref(false);
  const companionResearchSnapshot = ref(null);

  function applyCampaignState(msg) {
    if (msg && msg.campaign_state) {
      campaignState.value = msg.campaign_state;
    }
  }

  let reconnectTimer = null;
  const RECONNECT_DELAY_MS = 2000;
  const agentWorking = ref(false);
  const agentStopping = ref(false);
  const agentAwaitingResponse = ref(false);
  const agentTickProgress = ref({ done: 0, total: 0, active: false });
  const lastAgentStepAt = ref(0);
  const agentThinkingNow = ref(Date.now());
  let agentThinkingClockTimer = null;
  let agentStopNoticeShown = false;

  function noteAgentStep() {
    lastAgentStepAt.value = Date.now();
    ensureAgentThinkingClock();
  }

  function beginAgentMissionClock() {
    // No step happened yet: the banner must show right away and keep showing
    // until the first real step lands (which then hides it for 2 seconds).
    lastAgentStepAt.value = 0;
    // A stale unanswered question from a previous run must not keep the
    // banner suppressed for the whole new mission.
    pendingAgentQuestion.value = "";
    ensureAgentThinkingClock();
  }

  function resetAgentMissionClock() {
    lastAgentStepAt.value = 0;
    if (agentThinkingClockTimer) {
      clearInterval(agentThinkingClockTimer);
      agentThinkingClockTimer = null;
    }
  }

  function ensureAgentThinkingClock() {
    const missionActive = agentWorking.value
      || companionResearchActive.value
      || agentTickProgress.value.active;
    if (!missionActive) {
      if (agentThinkingClockTimer) {
        clearInterval(agentThinkingClockTimer);
        agentThinkingClockTimer = null;
      }
      return;
    }
    agentThinkingNow.value = Date.now();
    if (!agentThinkingClockTimer) {
      agentThinkingClockTimer = setInterval(() => {
        agentThinkingNow.value = Date.now();
      }, 200);
    }
  }

  watch(
    () => agentWorking.value || companionResearchActive.value || agentTickProgress.value.active,
    () => ensureAgentThinkingClock(),
    { immediate: true },
  );

  watch(agentWorking, (working) => {
    if (!working && !agentTickProgress.value.active) {
      agentStopping.value = false;
      agentAwaitingResponse.value = false;
    }
  });

  function isAgentMissionActive() {
    return Boolean(
      agentWorking.value
      || companionResearchActive.value
      || agentTickProgress.value.active,
    );
  }

  function setAgentAwaitingResponse(next) {
    agentAwaitingResponse.value = !!next;
  }

  function markAwaitingNextStepIfNeeded(msg) {
    if (!agentWorking.value || agentStopping.value) {
      setAgentAwaitingResponse(false);
      return;
    }
    const tick = Number(msg?.tick);
    const total = Number(msg?.total_ticks);
    if (Number.isFinite(tick) && Number.isFinite(total) && tick > 0 && total > 0) {
      setAgentAwaitingResponse(tick < total);
      return;
    }
    setAgentAwaitingResponse(true);
  }

  function applyAgentTickProgress(msg) {
    if (msg == null || msg.tick == null || msg.total_ticks == null) return;
    const total = Math.max(1, Math.round(Number(msg.total_ticks)));
    const done = Math.max(0, Math.min(total, Math.round(Number(msg.tick))));
    agentTickProgress.value = { done, total, active: true };
  }

  function clearAgentTickProgress() {
    agentTickProgress.value = { done: 0, total: 0, active: false };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, RECONNECT_DELAY_MS);
  }

  function connect() {
    try {
      const sock = new WebSocket(url);
      ws.value = sock;
      status.value = "connecting";
      sock.onopen = () => {
        status.value = "connected";
      };
      sock.onclose = () => {
        status.value = "disconnected";
        agentWorking.value = false;
        clearAgentTickProgress();
        resetAgentMissionClock();
        scheduleReconnect();
      };
      sock.onerror = () => {
        status.value = "error";
        agentWorking.value = false;
        clearAgentTickProgress();
        resetAgentMissionClock();
      };
      sock.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        applyCampaignState(msg);
        if (msg.companion_research) {
          companionResearchSnapshot.value = msg.companion_research;
        }
        if (msg.type === "companion_research_started") {
          companionResearchActive.value = true;
          agentWorking.value = true;
          setAgentAwaitingResponse(true);
          beginAgentMissionClock();
          return;
        }
        if (msg.type === "companion_research_complete") {
          companionResearchActive.value = false;
          agentWorking.value = false;
          agentStopping.value = false;
          resetAgentMissionClock();
          if (msg.frame) {
            currentFrame.value = msg.frame;
            if (msg.frame.agent_observation != null) agentObservation.value = msg.frame.agent_observation;
          }
          if (typeof onCompanionResearchComplete === "function") {
            onCompanionResearchComplete(msg);
          }
          return;
        }
        if (msg.type === "frame") {
          if (msg.debug_timing) {
            console.debug("frame timing", msg.debug_timing);
          }
          if (msg.frame) {
            currentFrame.value = msg.frame;
            if (msg.frame.agent_observation != null) agentObservation.value = msg.frame.agent_observation;
          }
          if (msg.reward !== undefined) reward.value = msg.reward;
          if (msg.player_position !== undefined) playerPosition.value = msg.player_position;
          if (msg.done !== undefined) done.value = msg.done;
          notifyAgentPromptChanged();
          return;
        }
        if (msg.type === "oracle_answer") {
          if (msg.error_kind === "model_not_found") showModelError(msg);
          const question = msg.question || "";
          const idx = messages.value.findIndex((m) => m.pending && m.question === question);
          const now = Date.now();
          const entry = {
            kind: "oracle",
            question,
            answer: msg.answer || "",
            ok: msg.ok,
            error: msg.error,
            pending: false,
            responseTimeMs: idx >= 0 && messages.value[idx].sentAt != null ? now - messages.value[idx].sentAt : null,
          };
          if (idx >= 0) messages.value[idx] = entry;
          else messages.value.push(entry);
          if (msg.ok && msg.frame) {
            currentFrame.value = msg.frame;
            if (msg.frame.agent_observation != null) agentObservation.value = msg.frame.agent_observation;
          }
          return;
        }
        if (msg.type === "agent_operator_notice") {
          notifyKnowledgeUpdated(msg);
          applyAgentTickProgress(msg);
          const tickInfo = msg.tick ? `[Tick ${msg.tick}${msg.total_ticks ? `/${msg.total_ticks}` : ""}]` : "";
          const notice = msg.message || "";
          messages.value.push({
            kind: "operator_notice",
            question: tickInfo,
            answer: notice,
            reasoning: msg.reasoning || "",
            ok: true,
            error: null,
            pending: false,
          });
          agentReasoning.value =
            typeof msg.reasoning === "string" && msg.reasoning.trim() ? msg.reasoning : "...";
          if (msg.frame) {
            currentFrame.value = msg.frame;
            if (msg.frame.agent_observation != null) agentObservation.value = msg.frame.agent_observation;
          }
          if (agentWorking.value) setAgentAwaitingResponse(true);
          return;
        }
        if (msg.type === "agent_question_pending") {
          notifyKnowledgeUpdated(msg);
          applyAgentTickProgress(msg);
          const tickInfo = msg.tick ? `[Tick ${msg.tick}${msg.total_ticks ? `/${msg.total_ticks}` : ""}]` : "";
          const rawQuestion = msg.question || "";
          const question = tickInfo && rawQuestion ? `${tickInfo} ${rawQuestion}` : rawQuestion;
          agentReasoning.value =
            typeof msg.reasoning === "string" && msg.reasoning.trim() ? msg.reasoning : "...";
          messages.value.push({
            kind: "agent_message",
            question,
            rawQuestion,
            answer: "",
            reasoning: msg.reasoning || "",
            ok: true,
            error: null,
            pending: true,
            sentAt: Date.now(),
          });
          pendingAgentQuestion.value = rawQuestion;
          agentWorking.value = false;
          setAgentAwaitingResponse(false);
          resetAgentMissionClock();
          notifyAgentPromptChanged();
          return;
        }
        if (msg.type === "agent_stop_ack") {
          applyCampaignState(msg);
          const stillRunning = Boolean(msg.running);
          if (stillRunning) {
            agentStopping.value = true;
            if (companionResearchActive.value || agentTickProgress.value.active) {
              agentWorking.value = true;
            }
          } else {
            agentWorking.value = false;
            agentStopping.value = false;
            setAgentAwaitingResponse(false);
            resetAgentMissionClock();
            const prev = agentTickProgress.value;
            agentTickProgress.value = {
              done: prev.done,
              total: prev.total,
              active: false,
            };
          }
          if (!agentStopNoticeShown) {
            messages.value.push({
              kind: "system",
              question: "",
              answer: msg.message || "Stop requested.",
              ok: true,
              pending: false,
            });
            agentStopNoticeShown = true;
          }
          return;
        }
        if (msg.type === "agent_system_notice") {
          notifyKnowledgeUpdated(msg);
          applyAgentTickProgress(msg);
          const isModelError = msg.error_kind === "model_not_found";
          if (isModelError) showModelError(msg);
          const tickInfo = msg.tick ? `[Tick ${msg.tick}${msg.total_ticks ? `/${msg.total_ticks}` : ""}]` : "";
          messages.value.push({
            kind: "system",
            question: tickInfo,
            answer: msg.message || msg.answer || "Unknown system notice.",
            reasoning: msg.reasoning || "",
            ok: !isModelError,
            error: isModelError ? "model_not_found" : null,
            pending: false,
          });
          if (typeof msg.reasoning === "string" && msg.reasoning.trim()) {
            agentReasoning.value = msg.reasoning;
          }
          if (msg.frame) {
            currentFrame.value = msg.frame;
            if (msg.frame.agent_observation != null) agentObservation.value = msg.frame.agent_observation;
          }
          return;
        }
        if (msg.type === "agent_message") {
          notifyKnowledgeUpdated(msg);
          applyAgentTickProgress(msg);
          const tickInfo = msg.tick ? `[Tick ${msg.tick}${msg.total_ticks ? `/${msg.total_ticks}` : ""}]` : "";
          const questionText = (msg.question || "").trim();
          const question = tickInfo && questionText ? `${tickInfo} ${questionText}` : questionText;
          let idx = -1;
          if (questionText) {
            for (let i = messages.value.length - 1; i >= 0; i--) {
              if (messages.value[i].pending && messages.value[i].kind === "agent_message") {
                idx = i;
                break;
              }
            }
          }
          const now = Date.now();
          const answerText = msg.answer || "";
          const entry = {
            kind: "agent_message",
            question,
            answer: answerText,
            reasoning: msg.reasoning || "",
            ok: msg.ok !== false || Boolean(answerText && !msg.error),
            error: msg.error || null,
            pending: false,
            responseTimeMs: idx >= 0 && messages.value[idx].sentAt != null ? now - messages.value[idx].sentAt : null,
          };
          setAgentAwaitingResponse(false);
          agentReasoning.value =
            typeof msg.reasoning === "string" && msg.reasoning.trim() ? msg.reasoning : "...";
          if (idx >= 0) messages.value[idx] = entry;
          else messages.value.push(entry);
          // The question got answered (e.g. by the AI operator) — it is no
          // longer pending, otherwise the "thinking" banner stays hidden.
          if (idx >= 0 || String(pendingAgentQuestion.value || "").trim() === questionText) {
            pendingAgentQuestion.value = "";
          }
          if (msg.frame) {
            currentFrame.value = msg.frame;
            if (msg.frame.agent_observation != null) agentObservation.value = msg.frame.agent_observation;
          }
          noteAgentStep();
          markAwaitingNextStepIfNeeded(msg);
          return;
        }
        if (msg.type === "operator_answer_ack") {
          const question = msg.question || "";
          const answer = msg.answer || (msg.ok ? "Answer saved." : "");
          let idx = findAgentMessageIndexByQuestion(messages.value, question);
          if (idx < 0) idx = findLastPendingAgentMessageIndex(messages.value);
          const now = Date.now();
          if (idx >= 0) {
            const prev = messages.value[idx];
            messages.value[idx] = {
              kind: "agent_message",
              question: prev.question || question || "",
              rawQuestion: question || prev.rawQuestion || "",
              answer: answer || prev.answer || "",
              reasoning: prev.reasoning || "",
              ok: !!msg.ok,
              error: msg.error || null,
              pending: false,
              sentAt: prev.sentAt,
              responseTimeMs: prev.responseTimeMs != null
                ? prev.responseTimeMs
                : (prev.sentAt != null ? now - prev.sentAt : null),
            };
          } else {
            messages.value.push({
              kind: "agent_message",
              question,
              answer,
              reasoning: "",
              ok: !!msg.ok,
              error: msg.error || null,
              pending: false,
              sentAt: now,
              responseTimeMs: null,
            });
          }
          if (msg.ok && msg.resuming) {
            agentWorking.value = true;
            agentStopping.value = false;
            setAgentAwaitingResponse(true);
            beginAgentMissionClock();
            if (!companionResearchActive.value) {
              agentTickProgress.value = {
                ...agentTickProgress.value,
                active: true,
              };
            }
          }
          notifyAgentPromptChanged();
          return;
        }
        if (msg.type === "agent_action") {
          notifyKnowledgeUpdated(msg);
          applyAgentTickProgress(msg);
          const tickInfo = msg.tick ? `[Tick ${msg.tick}${msg.total_ticks ? `/${msg.total_ticks}` : ""}]` : "";
          messages.value.push({
            kind: "agent_action",
            action: tickInfo ? `${tickInfo} ${msg.action || ""}` : msg.action || "",
            question: "",
            answer: "",
            reasoning: msg.reasoning || "",
            ok: true,
            pending: false,
          });
          setAgentAwaitingResponse(false);
          agentReasoning.value =
            typeof msg.reasoning === "string" && msg.reasoning.trim() ? msg.reasoning : "...";
          if (msg.reward !== undefined) reward.value = msg.reward;
          if (msg.player_position !== undefined) playerPosition.value = msg.player_position;
          if (msg.done !== undefined) done.value = msg.done;
          const frames = msg.frames || [];
          if (frames.length === 0 && msg.frame) {
            currentFrame.value = msg.frame;
            if (msg.frame.agent_observation != null) agentObservation.value = msg.frame.agent_observation;
          } else if (frames.length > 0) {
            playFramesStepByStep(frames, 500, (f) => {
              currentFrame.value = f;
              if (f && f.agent_observation != null) agentObservation.value = f.agent_observation;
            });
          }
          noteAgentStep();
          if (msg.done) {
            agentWorking.value = false;
            resetAgentMissionClock();
          } else {
            markAwaitingNextStepIfNeeded(msg);
          }
          notifyAgentPromptChanged();
          return;
        }
        if (msg.type === "achievement_discovered") {
          applyCampaignState(msg);
          if (typeof onAchievementDiscovered === "function") {
            onAchievementDiscovered(msg.achievement);
          }
          if (msg.frame) {
            currentFrame.value = msg.frame;
            if (msg.frame.agent_observation != null) agentObservation.value = msg.frame.agent_observation;
          }
          return;
        }
        if (msg.type === "agent_tick_complete") {
          agentWorking.value = false;
          agentStopping.value = false;
          resetAgentMissionClock();
          const prev = agentTickProgress.value;
          agentTickProgress.value = {
            done: msg.stopped ? prev.done : (prev.total || prev.done),
            total: prev.total,
            active: false,
          };
          agentHasStepped.value = true;
          if (statsPanelCollapsed.value) statsPanelNeonAlert.value = true;
          if (msg.stopped && !agentStopNoticeShown) {
            messages.value.push({
              kind: "system",
              question: "",
              answer: "Agent stopped after the last completed step.",
              ok: true,
              pending: false,
            });
          }
          agentStopNoticeShown = false;
          return;
        }
        if (msg.type === "campaign_status") {
          if (msg.frame) {
            currentFrame.value = msg.frame;
            if (msg.frame.agent_observation != null) agentObservation.value = msg.frame.agent_observation;
          }
          return;
        }
        if (msg.type === "agent_direct_chat_status") {
          agentWorking.value = false;
          resetAgentMissionClock();
          if (typeof msg.active === "boolean") agentDirectChatActive.value = msg.active;
          if (msg.ended) {
            messages.value.push({
              kind: "system",
              question: "",
              answer: msg.message || "Direct agent chat ended.",
              ok: true,
              pending: false,
            });
          } else if (!msg.ok && msg.error) {
            messages.value.push({
              kind: "error",
              question: "",
              answer: msg.error,
              ok: false,
              pending: false,
            });
          }
          return;
        }
        if (msg.type === "agent_direct_chat_reply") {
          notifyKnowledgeUpdated(msg);
          agentWorking.value = false;
          if (typeof msg.active === "boolean") agentDirectChatActive.value = msg.active;
          let idx = -1;
          for (let i = messages.value.length - 1; i >= 0; i--) {
            if (messages.value[i].pending && messages.value[i].kind === "human_agent") {
              idx = i;
              break;
            }
          }
          const now = Date.now();
          const entry = {
            kind: "human_agent",
            question: msg.human_message || "",
            answer: msg.raw_answer || "",
            ok: !!msg.ok,
            pending: false,
            responseTimeMs: idx >= 0 && messages.value[idx].sentAt != null ? now - messages.value[idx].sentAt : null,
          };
          if (idx >= 0) messages.value[idx] = entry;
          else messages.value.push(entry);
          agentReasoning.value =
            typeof msg.reasoning === "string" && msg.reasoning.trim() ? msg.reasoning : "...";
          return;
        }
        if (msg.type === "error") {
          agentWorking.value = false;
          messages.value.push({
            kind: "error",
            question: "",
            answer: msg.error || "Unknown error",
            ok: false,
            pending: false,
          });
        }
      };
    } catch (e) {
      status.value = "error";
      agentWorking.value = false;
    }
  }

  function send(obj) {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) ws.value.send(JSON.stringify(obj));
  }
  function reset() { send({ type: "reset" }); }
  function step(action) { send({ type: "step", action }); }
  function oracleAsk(question, runCode, forcedExpert = "") {
    const payload = { type: "oracle_ask", question, run_code: runCode };
    if (forcedExpert && forcedExpert !== "auto") payload.forced_expert = forcedExpert;
    send(payload);
  }
  function agentTick(goal, steps = 1, independent = true, runtimeOverrides = null) {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      agentStopNoticeShown = false;
      agentStopping.value = false;
      agentWorking.value = true;
      setAgentAwaitingResponse(true);
      beginAgentMissionClock();
      agentTickProgress.value = {
        done: 0,
        total: Math.max(1, Math.round(Number(steps) || 1)),
        active: true,
      };
      agentReasoning.value = "";
      const payload = {
        type: "agent_tick",
        goal: goal || "Drink a water",
        steps: steps || 1,
        independent_agent: independent,
      };
      if (runtimeOverrides && typeof runtimeOverrides === "object") {
        if (runtimeOverrides.active_agent_model) {
          payload.active_agent_model = String(runtimeOverrides.active_agent_model);
        }
        if (runtimeOverrides.active_agent_mode) {
          payload.active_agent_mode = String(runtimeOverrides.active_agent_mode);
        }
        if (runtimeOverrides.megaprompt_config_name) {
          payload.megaprompt_config_name = String(runtimeOverrides.megaprompt_config_name);
        }
        if (runtimeOverrides.arc_prompt_extra !== undefined) {
          payload.arc_prompt_extra = String(runtimeOverrides.arc_prompt_extra || "");
        }
        if (runtimeOverrides.game_kind) {
          payload.game_kind = String(runtimeOverrides.game_kind);
        }
        if (runtimeOverrides.arc_game_id) {
          payload.arc_game_id = String(runtimeOverrides.arc_game_id);
        }
        if (runtimeOverrides.exo_planet_enabled !== undefined) {
          payload.exo_planet_enabled = Boolean(runtimeOverrides.exo_planet_enabled);
        }
      }
      send(payload);
    }
  }
  function agentStop() {
    if (!isAgentMissionActive()) return;
    agentStopping.value = true;
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      send({ type: "agent_stop" });
    }
  }
  function agentDirectChat(message) {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      agentStopping.value = false;
      agentWorking.value = true;
      agentReasoning.value = "";
      setAgentAwaitingResponse(true);
      beginAgentMissionClock();
      send({ type: "agent_direct_chat", message: message || "" });
    }
  }

  function setCampaignEnabled(enabled) {
    send({ type: "campaign_toggle", enabled: !!enabled });
  }

  function startCampaignPhase2(levelKey) {
    send({ type: "campaign_phase2_start", level_key: String(levelKey || "") });
  }

  return { ws, status, reward, done, playerPosition, currentFrame, agentObservation, agentReasoning, messages, campaignState, companionResearchActive, companionResearchSnapshot, agentWorking, agentStopping, agentAwaitingResponse, agentTickProgress, lastAgentStepAt, agentThinkingNow, beginAgentMissionClock, isAgentMissionActive, connect, send, reset, step, oracleAsk, agentTick, agentStop, agentDirectChat, setCampaignEnabled, startCampaignPhase2 };
}

let pendingAgentQuestion = ref("");
let agentDirectChatActive = ref(false);

createApp({
  setup() {
    const apiBase = window.PlayWebSession?.resolveApiBase?.() || "http://127.0.0.1:8001/api";
    const wsUrl = window.PlayWebSession?.wsUrlWithSession?.(
      window.PlayWebSession?.resolveWsUrl?.() || "ws://127.0.0.1:8001/ws",
    ) || "ws://127.0.0.1:8001/ws";

    function apiFetch(path, options = {}) {
      return window.PlayWebSession.apiFetch(apiBase, path, options);
    }

    function apiSecretsForRequest() {
      const stored = window.PlayWebSession?.readApiSecrets?.() || {};
      return {
        hf: hfTokenInput.value.trim() || String(stored.hf_token || "").trim(),
        openrouter: openrouterApiKeyInput.value.trim() || String(stored.openrouter_api_key || "").trim(),
      };
    }

    function persistApiSecretsToStorage() {
      const { hf, openrouter } = apiSecretsForRequest();
      if (!hf && !openrouter) return;
      window.PlayWebSession.writeApiSecrets({
        hf_token: hf,
        openrouter_api_key: openrouter,
      });
    }

    function hasStoredApiSecrets() {
      const stored = window.PlayWebSession?.readApiSecrets?.() || {};
      return Boolean(
        String(stored.hf_token || "").trim() || String(stored.openrouter_api_key || "").trim(),
      );
    }

    function persistSharedSettingsToStorage() {
      try {
        localStorage.setItem(SESSION_PREFS_STORAGE_KEY, JSON.stringify(collectSettingsSnapshot()));
      } catch (e) {}
    }

    function applySharedSettingsFromStorage() {
      let snapshot;
      try {
        const raw = localStorage.getItem(SESSION_PREFS_STORAGE_KEY);
        if (!raw) return false;
        snapshot = JSON.parse(raw);
      } catch (e) {
        return false;
      }
      applySettingsSnapshot(snapshot);
      alignMegapromptToWorldMode();
      return true;
    }

    async function bootstrapSessionSettings() {
      await loadSessionConfig();
      let desiredExo = exoPlanetEnabled.value;
      let desiredGameKind = gameKind.value;
      let desiredArcGameId = arcGameId.value;
      let pendingWorldMode = null;
      try {
        pendingWorldMode = localStorage.getItem(PENDING_WORLD_MODE_KEY);
      } catch (_e) {
        pendingWorldMode = null;
      }
      if (pendingWorldMode) {
        if (pendingWorldMode.startsWith("arc_agi:")) {
          desiredGameKind = "arc_agi";
          desiredArcGameId = pendingWorldMode.split(":", 2)[1] || "ls20";
          desiredExo = false;
        } else {
          desiredGameKind = "craftax";
          desiredExo = pendingWorldMode === "exo-planet";
        }
        try { localStorage.removeItem(PENDING_WORLD_MODE_KEY); } catch (_e) {}
      }
      applySharedSettingsFromStorage();
      if (pendingWorldMode) {
        gameKind.value = desiredGameKind;
        arcGameId.value = desiredArcGameId;
        exoPlanetEnabled.value = desiredExo;
        restoreAgentModelForCurrentContext();
        ensureDemoAgentModelPreset();
      }
      if (isArcGame.value) {
        ensureArcCompanionMode();
      }
      syncExpertModesToActiveAgent();
      syncExpertModelsToActiveAgent();
      alignMegapromptToWorldMode();
      await loadInventoryIcons(true);
      const saved = await saveSessionConfig(undefined, { silent: true });
      if (!saved.ok && pendingWorldMode) {
        if (desiredGameKind === "arc_agi") {
          gameKind.value = "craftax";
          exoPlanetEnabled.value = false;
          companionModeEnabled.value = false;
          flashSettingsMessage(
            saved.error || "Could not switch to ARC-AGI-3. Install arc-agi from requirements.txt and check local environment files.",
            false,
          );
          await saveSessionConfig(undefined, { silent: true });
          return;
        }
        // The world switch may still be running server-side (e.g. the gateway
        // timed out on a slow texture/JIT warmup). Never fall back to another
        // world: poll until the server reports the desired mode, then retry
        // the save once if it still differs.
        for (let attempt = 0; attempt < 30; attempt++) {
          await new Promise((r) => setTimeout(r, 3000));
          try {
            await loadSessionConfig();
            if (
              gameKind.value === desiredGameKind
              && exoPlanetEnabled.value === desiredExo
            ) break;
          } catch (_e) {}
        }
        if (gameKind.value !== desiredGameKind || exoPlanetEnabled.value !== desiredExo) {
          gameKind.value = desiredGameKind;
          arcGameId.value = desiredArcGameId;
          exoPlanetEnabled.value = desiredExo;
          await saveSessionConfig(undefined, { silent: true });
        }
      }
    }
    const agentKnowledge = ref("");
    const agentKnowledgeLoading = ref(false);
    const knowledgeNeonAlert = ref(false);
    let knowledgeModalInstance = null;
    let settingsModalInstance = null;
    let observationModalInstance = null;
    let agentPromptModalInstance = null;
    let worldModeModalInstance = null;
    let agentPromptRefreshTimer = null;
    let agentPromptAutoRefreshTimer = null;
    const worldModeSwitching = ref(false);
    const PENDING_WORLD_MODE_KEY = "playWebPendingWorldMode";
    const agentPromptGoal = ref("");
    const agentPromptSystem = ref("");
    const agentPromptUser = ref("");
    const agentPromptLoading = ref(false);
    const agentPromptHasPrompt = ref(false);
    const agentPromptError = ref("");
    const agentPromptViewMode = ref("preview");

    const agentPromptSystemHtml = computed(() => {
      const md = window.PlayWebPromptMarkdown;
      return md ? md.renderMarkdown(agentPromptSystem.value) : "";
    });
    const agentPromptUserHtml = computed(() => {
      const md = window.PlayWebPromptMarkdown;
      return md ? md.renderMarkdown(agentPromptUser.value) : "";
    });
    const agentPromptUserSections = computed(() => {
      const md = window.PlayWebPromptMarkdown;
      return md ? md.extractSections(agentPromptUser.value) : [];
    });

    function scrollAgentPromptToSection(sectionId) {
      const container = document.getElementById("agentPromptUserViewMd");
      if (!container || !sectionId) return;
      const target = container.querySelector(`#${CSS.escape(sectionId)}`);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    const arcPromptExtra = ref("");
    const appLoading = ref(false);
    const appLoadingText = ref("Preparing your session...");
    const appLoadingProgress = ref(0);
    const appLoadingSpriteSrc = ref("");
    const launchLoadingMessages = [
      "Preparing your session...",
      "Syncing oracle channels...",
      "Loading world textures...",
      "Almost ready...",
    ];
    let loadingTextTimerId = null;
    let loadingProgressTimerId = null;
    const WIZARD_COMPLETE_KEY = "playWebWizardComplete";
    const COMPANION_START_HINT_KEY = "playWebCompanionStartHint";
    // Bump when wizard steps/content change materially — old completions won't skip the new flow.
    const WIZARD_COMPLETE_VERSION = "4";
    const EXPERT_MODE_KEYS = [
      "map_expert",
      "mechanics_expert",
      "action_expert",
      "question_expert",
      "goal_expert",
      "path_expert_helper",
      "path_expert",
    ];
    const setupWizardSteps = [
      { id: 1, label: "Profile" },
      { id: 2, label: "Realm" },
      { id: 3, label: "Operator" },
      { id: 4, label: "Play mode" },
    ];
    const WIZARD_REALM_OPERATOR_CAPS = {
      craftax: { human_operator: true, ai_operator: true },
      "exo-planet": { human_operator: true, ai_operator: true },
    };
    const WIZARD_OPERATOR_BADGE_DEFS = [
      {
        key: "human",
        icon: "bi bi-headset",
        availableTooltip: "Human Operator — you answer the agent's questions.",
        unavailableTooltip: "Human Operator is not supported for this game.",
      },
      {
        key: "ai",
        icon: "bi bi-stars",
        availableTooltip: "AI Operator — an oracle answers using retrieved knowledge.",
        unavailableTooltip: "AI Operator is not supported for this game.",
      },
    ];
    const wizardStepTouched = ref({
      profile: false,
      realm: false,
      role: false,
      playMode: false,
    });
    const storedPlayerProfile = loadPlayerProfile();
    const playerNickname = ref(storedPlayerProfile.nickname);
    const playerAvatarId = ref(storedPlayerProfile.avatar_id);
    const avatarOptions = AVATAR_OPTIONS;
    const playerAvatarSrc = computed(() => avatarSrcFor(playerAvatarId.value));
    const playerDisplayName = computed(() => playerNickname.value.trim() || "Human Operator");
    function isGameEntryUrl() {
      try {
        return new URLSearchParams(window.location.search).has("play");
      } catch (e) {
        return false;
      }
    }
    function gameEntryUrl() {
      const base = window.location.pathname.replace(/[^/]+$/, "");
      return `${base}game.html`;
    }
    function stripGameEntryUrl() {
      try {
        const url = new URL(window.location.href);
        if (!url.searchParams.has("play")) return;
        url.searchParams.delete("play");
        const next = `${url.pathname}${url.search}${url.hash}`;
        window.history.replaceState(null, "", next || "./index.html");
      } catch (e) {}
    }
    function readWizardComplete() {
      try {
        if (new URLSearchParams(window.location.search).has("setup")) return false;
        if (!isGameEntryUrl()) return false;
        return sessionStorage.getItem(WIZARD_COMPLETE_KEY) === WIZARD_COMPLETE_VERSION;
      } catch (e) {
        return false;
      }
    }
    function markWizardComplete() {
      try {
        sessionStorage.setItem(WIZARD_COMPLETE_KEY, WIZARD_COMPLETE_VERSION);
        localStorage.removeItem(WIZARD_COMPLETE_KEY);
      } catch (e) {}
    }
    function clearWizardComplete() {
      try {
        sessionStorage.removeItem(WIZARD_COMPLETE_KEY);
        localStorage.removeItem(WIZARD_COMPLETE_KEY);
      } catch (e) {}
    }
    const setupComplete = ref(readWizardComplete());
    const setupWizardStep = ref(1);
    const setupWizardGateway = ref("openrouter");
    const setupWizardToken = ref("");
    const wizardTokenError = ref("");
    const wizardLaunching = ref(false);
    const companionBenchStatus = ref(null);
    const companionBenchError = ref("");
    const companionBenchStarting = ref(false);
    const companionBenchStopping = ref(false);
    const companionBenchMaxTicksPerTask = ref(DEFAULT_COMPANION_MAX_TICKS_PER_TASK);
    const companionBenchMaxTicksDirty = ref(false);
    const companionBenchParallelAgents = ref(3);
    const companionBenchParallelAgentsDirty = ref(false);
    const companionBenchCycles = ref(1);
    const companionBenchCyclesDirty = ref(false);
    const companionBenchTestTaskKey = ref("");
    const companionBenchKnowledgeSource = ref("base");
    const companionBenchKnowledgeSourceDirty = ref(false);
    const companionBenchTestKnowledgeSource = ref("base");
    const companionBenchTestKnowledgeSourceDirty = ref(false);
    const companionStripSettingsOpen = ref(false);
    let companionBenchModalInstance = null;
    let companionBenchPollTimerId = null;

    async function fetchAgentKnowledge() {
      agentKnowledgeLoading.value = true;
      try {
        const useModelScope =
          companionModeEnabled.value || companionResearchActive.value;
        const scope = useModelScope ? "model" : "default";
        const resp = await apiFetch(`/agent_knowledge?scope=${scope}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        agentKnowledge.value = String(data.knowledge || "").trim() || "—";
      } catch (e) {
        agentKnowledge.value = "(failed to load knowledge)";
      } finally {
        agentKnowledgeLoading.value = false;
      }
    }

    function onKnowledgeUpdatedFromAgent() {
      const modalEl = document.getElementById("knowledgeModal");
      if (modalEl && modalEl.classList.contains("show")) {
        fetchAgentKnowledge();
        return;
      }
      knowledgeNeonAlert.value = true;
    }

    function achievementLearnMessage(title) {
      const t = String(title || "").trim();
      if (t.startsWith("Collect ")) return `Agent learned how to obtain ${t.slice(8)}`;
      if (t.startsWith("Make ")) return `Agent learned how to make ${t.slice(5)}`;
      if (t.startsWith("Deploy ")) return `Agent learned how to deploy ${t.slice(7)}`;
      if (t.startsWith("Place ")) return `Agent learned how to place ${t.slice(6)}`;
      return `Agent learned: ${t}`;
    }

    const achievementToast = ref(null);
    const achievementToastQueue = [];
    let achievementToastTimerId = null;

    function drainAchievementToastQueue() {
      if (achievementToast.value || achievementToastQueue.length === 0) return;
      const achievement = achievementToastQueue.shift();
      achievementToast.value = {
        key: String(achievement?.key || ""),
        message: achievementLearnMessage(achievement?.title),
      };
      if (achievementToastTimerId) clearTimeout(achievementToastTimerId);
      achievementToastTimerId = setTimeout(() => {
        achievementToast.value = null;
        achievementToastTimerId = null;
        setTimeout(drainAchievementToastQueue, 250);
      }, 4500);
    }

    function showAchievementToast(achievement) {
      if (!achievement || !achievement.key) return;
      achievementToastQueue.push(achievement);
      drainAchievementToastQueue();
    }

    function dismissAchievementToast() {
      achievementToast.value = null;
      if (achievementToastTimerId) clearTimeout(achievementToastTimerId);
      achievementToastTimerId = null;
      setTimeout(drainAchievementToastQueue, 250);
    }

    function clearAchievementToasts() {
      achievementToastQueue.length = 0;
      dismissAchievementToast();
    }

    function openKnowledgeModal() {
      knowledgeNeonAlert.value = false;
      fetchAgentKnowledge();
      const el = document.getElementById("knowledgeModal");
      if (!el) return;
      knowledgeModalInstance = knowledgeModalInstance || new bootstrap.Modal(el);
      knowledgeModalInstance.show();
    }

    function openSettingsModal() {
      const el = document.getElementById("settingsModal");
      if (!el) return;
      settingsModalInstance = settingsModalInstance || new bootstrap.Modal(el);
      settingsModalInstance.show();
    }

    function openObservationModal() {
      const el = document.getElementById("observationModal");
      if (!el) return;
      observationModalInstance = observationModalInstance || new bootstrap.Modal(el);
      observationModalInstance.show();
    }

    function openWorldModeModal() {
      const el = document.getElementById("worldModeModal");
      if (!el) return;
      worldModeModalInstance = worldModeModalInstance || new bootstrap.Modal(el);
      worldModeModalInstance.show();
    }

    async function chooseWorldMode(mode, gameId = "") {
      if (worldModeSwitching.value) return;
      hideWizardCapabilityTip();
      const rawMode = String(mode || "craftax");
      const targetKind = rawMode === "arc_agi" ? "arc_agi" : "craftax";
      const targetArcId = targetKind === "arc_agi" ? String(gameId || arcGameId.value || "ls20") : arcGameId.value;
      const targetExo = targetKind === "craftax" && rawMode === "exo-planet";
      if (
        targetKind === gameKind.value
        && targetExo === exoPlanetEnabled.value
        && (targetKind !== "arc_agi" || targetArcId === arcGameId.value)
      ) {
        worldModeModalInstance?.hide();
        return;
      }
      worldModeSwitching.value = true;
      try {
        localStorage.setItem(
          PENDING_WORLD_MODE_KEY,
          targetKind === "arc_agi" ? `arc_agi:${targetArcId}` : (targetExo ? "exo-planet" : "craftax"),
        );
      } catch (_e) {
        /* ignore storage errors */
      }
      window.location.reload();
    }

    async function openCompanionTestModal() {
      if (!guardApiKeyForAction()) return;
      const el = document.getElementById("companionBenchModal");
      if (!el) return;
      companionBenchModalInstance = companionBenchModalInstance || new bootstrap.Modal(el);
      companionBenchError.value = "";
      await loadCompanionBenchStatus();
      const status = companionBenchStatus.value || {};
      const phase = String(status.phase || "").toLowerCase();
      if (status.running && phase === "test") {
        await stopCompanionBench();
        await loadCompanionBenchStatus();
      }
      if (!companionBenchTestTaskKey.value && campaignTasks.value.length) {
        companionBenchTestTaskKey.value = String(campaignTasks.value[0].key || "");
      }
      if (!companionBenchTestKnowledgeSourceDirty.value) {
        companionBenchTestKnowledgeSource.value = "base";
      }
      if (
        companionBenchTestKnowledgeSource.value === "own"
        && !companionBenchStatus.value?.has_own_knowledge
      ) {
        companionBenchTestKnowledgeSource.value = "base";
      }
      companionBenchModalInstance.show();
    }

    function stopAgentPromptRefresh() {
      if (agentPromptRefreshTimer) {
        clearInterval(agentPromptRefreshTimer);
        agentPromptRefreshTimer = null;
      }
      if (agentPromptAutoRefreshTimer) {
        clearTimeout(agentPromptAutoRefreshTimer);
        agentPromptAutoRefreshTimer = null;
      }
    }

    async function prepareLoadingSprite() {
      const webpSrc = "./assets/loading-character.webp";
      const gifSrc = "./assets/loading-character.gif";
      const pngSrc = "./assets/loading-character.png";
      const webpReady = await new Promise((resolve) => {
        const webp = new Image();
        webp.onload = () => resolve(true);
        webp.onerror = () => resolve(false);
        webp.src = webpSrc;
      });
      if (webpReady) {
        appLoadingSpriteSrc.value = webpSrc;
        return;
      }
      const gifReady = await new Promise((resolve) => {
        const gif = new Image();
        gif.onload = () => resolve(true);
        gif.onerror = () => resolve(false);
        gif.src = gifSrc;
      });
      if (gifReady) {
        appLoadingSpriteSrc.value = gifSrc;
        return;
      }
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
          try {
            const canvas = document.createElement("canvas");
            canvas.width = img.naturalWidth || img.width;
            canvas.height = img.naturalHeight || img.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const px = imageData.data;
            const bgR = px[0];
            const bgG = px[1];
            const bgB = px[2];
            for (let i = 0; i < px.length; i += 4) {
              const r = px[i];
              const g = px[i + 1];
              const b = px[i + 2];
              if (r === bgR && g === bgG && b === bgB) px[i + 3] = 0;
            }
            ctx.putImageData(imageData, 0, 0);
            appLoadingSpriteSrc.value = canvas.toDataURL("image/png");
          } catch (e) {
            appLoadingSpriteSrc.value = pngSrc;
          }
          resolve();
        };
        img.onerror = () => {
          appLoadingSpriteSrc.value = "";
          resolve();
        };
        img.src = pngSrc;
      });
    }

    function startLaunchLoadingVisuals() {
      let messageIndex = 0;
      appLoadingText.value = launchLoadingMessages[messageIndex];
      loadingTextTimerId = setInterval(() => {
        messageIndex = (messageIndex + 1) % launchLoadingMessages.length;
        if (appLoadingProgress.value < 92) {
          appLoadingText.value = launchLoadingMessages[messageIndex];
        }
      }, 2800);
      // A world switch can hold one launch step for a minute or more; keep the
      // bar creeping so the screen never looks stuck while the character walks.
      loadingProgressTimerId = setInterval(() => {
        if (appLoadingProgress.value < 90) {
          appLoadingProgress.value += 1;
        }
      }, 1100);
    }

    function stopLoadingVisuals() {
      if (loadingTextTimerId) {
        clearInterval(loadingTextTimerId);
        loadingTextTimerId = null;
      }
      if (loadingProgressTimerId) {
        clearInterval(loadingProgressTimerId);
        loadingProgressTimerId = null;
      }
    }

    async function fetchAgentPrompt() {
      if (!featureFlags.value.agent_prompt_debug) {
        agentPromptError.value = "Agent prompt debug is disabled in this app profile.";
        agentPromptHasPrompt.value = false;
        return;
      }
      agentPromptLoading.value = true;
      agentPromptError.value = "";
      try {
        const resp = await apiFetch("/agent_prompt");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        agentPromptGoal.value = data.goal || "No goal set";
        agentPromptSystem.value = data.system_message != null ? String(data.system_message) : "";
        agentPromptUser.value = data.prompt != null ? String(data.prompt) : "";
        agentPromptHasPrompt.value = Boolean(data.has_prompt);
      } catch (e) {
        agentPromptError.value = `Failed to load agent prompt: ${e.message}`;
      } finally {
        agentPromptLoading.value = false;
      }
    }

    function scheduleAgentPromptPreviewRefresh() {
      const el = document.getElementById("agentPromptModal");
      if (!el || !el.classList.contains("show")) return;
      if (agentWorking.value) return;
      if (agentPromptAutoRefreshTimer) return;
      agentPromptAutoRefreshTimer = setTimeout(() => {
        agentPromptAutoRefreshTimer = null;
        fetchAgentPrompt();
      }, 150);
    }

    function openAgentPromptModal() {
      if (!featureFlags.value.agent_prompt_debug) return;
      const el = document.getElementById("agentPromptModal");
      if (!el) return;
      agentPromptViewMode.value = "preview";
      fetchAgentPrompt();
      agentPromptModalInstance = agentPromptModalInstance || new bootstrap.Modal(el);
      agentPromptModalInstance.show();
      stopAgentPromptRefresh();
    }

    async function saveArcPromptExtra() {
      await saveSessionConfig("ARC full prompt override saved.");
      fetchAgentPrompt();
    }

    async function clearArcPromptExtra() {
      arcPromptExtra.value = "";
      await saveArcPromptExtra();
    }

    function useCurrentAgentPromptAsArcOverride() {
      arcPromptExtra.value = String(agentPromptUser.value || "");
    }

    const { status, reward, done, playerPosition, currentFrame, agentObservation, agentReasoning, messages, campaignState, companionResearchActive, companionResearchSnapshot, agentWorking, agentStopping, agentAwaitingResponse, agentTickProgress, lastAgentStepAt, agentThinkingNow, beginAgentMissionClock, isAgentMissionActive, connect, reset: resetGame, step, oracleAsk, agentTick, agentStop, agentDirectChat, setCampaignEnabled, startCampaignPhase2, send } = useWebSocket(wsUrl, {
      onKnowledgeUpdated: onKnowledgeUpdatedFromAgent,
      onAchievementDiscovered: showAchievementToast,
      onCompanionResearchComplete: () => {
        loadCompanionBenchStatus();
        loadStatistics();
        loadHumanLeaderboard();
      },
      onAgentPromptChanged: scheduleAgentPromptPreviewRefresh,
    });

    const worldCanvasEl = ref(null);
    const arcFrameShellEl = ref(null);
    const arcFrameCanvasEl = ref(null);
    let worldMapRenderer = null;
    const mapFollowingAgent = ref(true);
    const logEl = ref(null);
    const questionInput = ref("");
    const questionInputEl = ref(null);
    const interactionMode = ref("oracle");
    const allExperts = ref([]);
    const selectedExperts = ref([]);
    const forcedExpert = ref("goal_expert");
    const megapromptConfigName = ref("database_formulation");
    const megapromptOptions = ref(["database_formulation"]);
    const expertButtons = [
      { id: "goal", label: "AI" },
      { id: "human", label: "Human" },
    ];
    pendingAgentQuestion = ref("");
    const AGENT_STEPS_PER_PLAY = 20;
    const agentStepsPerClick = ref(AGENT_STEPS_PER_PLAY);
    const maxAgentStepsPerClick = ref(100);
    const agentStepsRangeMax = computed(() =>
      Math.min(Math.max(1, maxAgentStepsPerClick.value || 1), 500),
    );
    const saveTrajectory = ref(false);
    const PANEL_LAYOUT_STORAGE_KEY = "playWebPanelLayout";
    const STATS_PANEL_WIDTH = { default: 520, min: 320, max: 720 };
    const OPERATOR_PANEL_WIDTH = { default: 420, min: 300, max: 560 };

    function clampPanelWidth(value, bounds) {
      const n = Number(value);
      if (!Number.isFinite(n)) return bounds.default;
      return Math.min(bounds.max, Math.max(bounds.min, Math.round(n)));
    }

    function loadPanelLayout() {
      try {
        const raw = localStorage.getItem(PANEL_LAYOUT_STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : null;
      } catch (_err) {
        return null;
      }
    }

    function savePanelLayout() {
      try {
        localStorage.setItem(
          PANEL_LAYOUT_STORAGE_KEY,
          JSON.stringify({
            statsWidth: statsPanelWidth.value,
            operatorWidth: operatorPanelWidth.value,
            reasoningCollapsed: reasoningPanelCollapsed.value,
            statsCollapsed: statsPanelCollapsed.value,
            operatorCollapsed: operatorPanelCollapsed.value,
          }),
        );
      } catch (_err) {
        /* ignore quota / private mode */
      }
    }

    const savedPanelLayout = loadPanelLayout();
    const statsPanelWidth = ref(
      clampPanelWidth(savedPanelLayout?.statsWidth, STATS_PANEL_WIDTH),
    );
    const operatorPanelWidth = ref(
      clampPanelWidth(savedPanelLayout?.operatorWidth, OPERATOR_PANEL_WIDTH),
    );
    const reasoningPanelCollapsed = ref(
      typeof savedPanelLayout?.reasoningCollapsed === "boolean"
        ? savedPanelLayout.reasoningCollapsed
        : true,
    );
    const statsPanelCollapsed = ref(
      typeof savedPanelLayout?.statsCollapsed === "boolean"
        ? savedPanelLayout.statsCollapsed
        : true,
    );
    const operatorPanelCollapsed = ref(
      typeof savedPanelLayout?.operatorCollapsed === "boolean"
        ? savedPanelLayout.operatorCollapsed
        : false,
    );
    const operatorPanelMessageAlert = ref(false);
    const operatorPanelMessageGlow = ref(false);
    let operatorPanelGlowTimer = null;
    const statsPanelNeonAlert = ref(false);
    const agentHasStepped = ref(false);
    const reasoningNeonAlert = ref(false);
    const oracleStats = ref(null);
    const oracleStatsError = ref("");
    const statsChartEl = ref(null);
    let statsChart = null;
    const campaignBenchmark = ref({ craftax: null, "exo-planet": null });
    const campaignBenchmarkError = ref("");
    const benchmarkSince = ref(window.PlayWebBenchmark?.loadBenchmarkSince?.() || "");
    const benchmarkSinceApplied = ref("");
    const currentTrajStats = ref(null);
    const currentTrajError = ref("");
    const currentQAChartEl = ref(null);
    const currentLenChartEl = ref(null);
    let currentQAChart = null;
    let currentLenChart = null;
    const CHART_FONT_MIN = 8;
    const CHART_FONT_MAX = 22;
    const chartFontSizes = ref({
      stats: { axis: 13, legend: 13 },
      currentLen: { axis: 13 },
    });

    function applyChartFontSizes(instance, sizes, { hasAxes = true, hasLegend = false } = {}) {
      if (!instance?.options) return false;
      let changed = false;
      if (hasAxes && sizes.axis != null) {
        const axisSize = Number(sizes.axis);
        for (const axisId of ["x", "y"]) {
          const scale = instance.options.scales?.[axisId];
          if (!scale) continue;
          scale.ticks = scale.ticks || {};
          const prevFont = scale.ticks.font;
          scale.ticks.font = typeof prevFont === "object" && prevFont
            ? { ...prevFont, size: axisSize }
            : { size: axisSize };
          changed = true;
        }
      }
      if (hasLegend && sizes.legend != null) {
        const legendSize = Number(sizes.legend);
        instance.options.plugins = instance.options.plugins || {};
        instance.options.plugins.legend = instance.options.plugins.legend || {};
        instance.options.plugins.legend.labels = instance.options.plugins.legend.labels || {};
        const prevFont = instance.options.plugins.legend.labels.font;
        instance.options.plugins.legend.labels.font = typeof prevFont === "object" && prevFont
          ? { ...prevFont, size: legendSize }
          : { size: legendSize };
        changed = true;
      }
      if (changed) instance.update();
      return changed;
    }

    function onChartFontChange(chartKey, field, evt) {
      const value = Number(evt?.target?.value);
      if (!Number.isFinite(value)) return;
      const bucket = chartFontSizes.value[chartKey];
      if (bucket && field) bucket[field] = value;
      const sizes = chartFontSizes.value[chartKey];
      if (!sizes) return;
      let applied = false;
      if (chartKey === "stats") {
        applied = applyChartFontSizes(statsChart, sizes, { hasAxes: true, hasLegend: true });
        if (!applied) updateStatsChart();
      } else if (chartKey === "currentLen") {
        applied = applyChartFontSizes(currentLenChart, sizes, { hasAxes: true, hasLegend: false });
        if (!applied) updateCurrentCharts();
      }
    }
    let currentTrajTimerId = null;
    const expertUiRows = [
      { key: "map_expert", label: "Map expert" },
      { key: "mechanics_expert", label: "Mechanics expert" },
      { key: "action_expert", label: "Action expert" },
      { key: "question_expert", label: "Question expert" },
      { key: "goal_expert", label: "Goal expert" },
      { key: "path_expert_helper", label: "Path helper expert" },
      { key: "path_expert", label: "Path expert" },
    ];
    const expertModelFields = ref({
      map_expert: "",
      mechanics_expert: "",
      action_expert: "",
      question_expert: "",
      goal_expert: "",
      path_expert_helper: "",
      path_expert: "",
    });
    const expertModeFields = ref({
      map_expert: "hub",
      mechanics_expert: "hub",
      action_expert: "hub",
      question_expert: "hub",
      goal_expert: "hub",
      path_expert_helper: "hub",
      path_expert: "hub",
    });
    const activeAgentModel = ref(DEFAULT_CRAFT_AGENT_MODEL);
    const activeAgentMode = ref("openrouter");
    const hfTokenInput = ref("");
    const openrouterApiKeyInput = ref("");
    const hfTokenPreview = ref("");
    const openrouterApiKeyPreview = ref("");
    const settingsSaveMessage = ref("");
    const settingsSaveOk = ref(true);
    const appProfile = ref("dev");
    const arcMultiLevel = ref(false);
    const featureFlags = ref({
      settings_api_keys: true,
      model_selection: true,
      expert_model_settings: true,
      observation_format_selection: true,
      arc_prompt_override: true,
      agent_prompt_debug: true,
      setup_wizard: true,
      leaderboard: true,
      human_operator: true,
      companion_bench: true,
    });
    let settingsSaveTimerId = null;
    const apiKeyAlertActive = ref(false);
    const apiKeyMissingKeys = ref([]);
    const apiKeyGuideFloatStyle = ref({ top: "0px", left: "0px" });
    const modelErrorAlert = ref(null);
    const invalidModels = ref([]);
    const saveToast = ref(null);
    let saveToastTimerId = null;
    const arcScore = ref(null);
    const humanLeaderboardRows = ref([]);
    const arcPlayerName = ref(storedPlayerProfile.nickname || "");
    const arcScoreError = ref("");
    const arcScoreSubmitting = ref(false);
    const arcScorePromptShown = ref(false);
    let arcScoreModalInstance = null;

    const activeAgentModelPresets = computed(() => activeAgentModelPresetsForGameKind(gameKind.value));
    const isDemoProfile = computed(() => appProfile.value === "demo");
    const agentModelByContext = ref({ arc_agi: "", craftax: "" });

    function agentModelContextKey() {
      return isArcGame.value ? "arc_agi" : "craftax";
    }

    function restoreAgentModelForCurrentContext() {
      const presets = activeAgentModelPresets.value;
      const key = agentModelContextKey();
      const saved = String(agentModelByContext.value[key] || "").trim();
      if (saved && presets.some((preset) => preset.id === saved)) {
        activeAgentModel.value = saved;
        return;
      }
      const fallback = presets[0]?.id;
      if (fallback) activeAgentModel.value = fallback;
    }

    const selectedAgentModelPreset = computed(() => {
      const current = String(activeAgentModel.value || "").trim();
      return activeAgentModelPresets.value.some((preset) => preset.id === current) ? current : "__custom__";
    });
    const activeAgentModelPresetInfo = computed(() => {
      const current = String(activeAgentModel.value || "").trim();
      return activeAgentModelPresets.value.find((preset) => preset.id === current) || null;
    });

    function ensureDemoAgentModelPreset() {
      if (!isDemoProfile.value) return;
      const presets = activeAgentModelPresets.value;
      const current = String(activeAgentModel.value || "").trim();
      if (presets.some((preset) => preset.id === current)) return;
      activeAgentModel.value = presets[0]?.id || current;
    }

    function ensureDemoArcGameSelection() {
      if (!isDemoProfile.value) return;
      const allowedIds = new Set(
        (Array.isArray(arcGameOptions.value) ? arcGameOptions.value : [])
          .map((option) => String(option?.id || "").trim())
          .filter(Boolean),
      );
      const current = String(arcGameId.value || "").trim();
      if (current && allowedIds.size && !allowedIds.has(current)) {
        arcGameId.value = allowedIds.has("ls20") ? "ls20" : [...allowedIds][0];
      }
    }

    function selectAgentModelPreset(modelId) {
      const normalized = String(modelId || "").trim();
      if (!normalized || normalized === "__custom__") return;
      activeAgentModel.value = normalized;
    }

    function showSaveToast(ok, message) {
      saveToast.value = {
        ok: Boolean(ok),
        message: String(message || (ok ? "Settings updated." : "Failed to update settings.")),
      };
      if (saveToastTimerId) clearTimeout(saveToastTimerId);
      saveToastTimerId = setTimeout(() => {
        saveToast.value = null;
      }, 4500);
    }

    function dismissSaveToast() {
      saveToast.value = null;
      if (saveToastTimerId) clearTimeout(saveToastTimerId);
    }

    function applyAppCapabilities(data) {
      if (!data || typeof data !== "object") return;
      appProfile.value = String(data.app_profile || appProfile.value || "dev");
      arcMultiLevel.value = Boolean(data.arc_multi_level);
      if (data.features && typeof data.features === "object") {
        featureFlags.value = { ...featureFlags.value, ...data.features };
      }
      ensureDemoAgentModelPreset();
    }

    async function loadHumanLeaderboard() {
      if (!companionModeEnabled.value && !isArcGame.value) {
        humanLeaderboardRows.value = [];
        return;
      }
      try {
        const params = new URLSearchParams();
        if (isArcGame.value) {
          params.set("game_kind", "arc_agi");
          params.set("arc_game_id", arcGameId.value);
        } else {
          params.set("game_kind", "craftax");
          params.set("world_mode", exoPlanetEnabled.value ? "exo-planet" : "craftax");
        }
        const resp = await apiFetch(`/human_leaderboard?${params.toString()}`);
        const data = await resp.json();
        if (!resp.ok || data.ok === false) throw new Error(data.error || `HTTP ${resp.status}`);
        humanLeaderboardRows.value = Array.isArray(data.rows) ? data.rows : [];
      } catch (_err) {
        humanLeaderboardRows.value = [];
      }
    }

    let lastCampaignLeaderboardCheckpoint = 0;
    watch(
      () => Number(campaignState.value?.completed_count || 0),
      (completed) => {
        if (!companionModeEnabled.value || completed <= 0) return;
        if (completed === lastCampaignLeaderboardCheckpoint) return;
        lastCampaignLeaderboardCheckpoint = completed;
        loadHumanLeaderboard();
      },
    );

    function humanLeaderboardTypeLabel(row) {
      const kind = String(row?.leaderboard_type || "");
      if (kind === "campaign") return "Companion";
      if (kind === "instruction") return "Instruction";
      return kind || "Run";
    }

    function leaderboardAvatarSrc(row) {
      if (row?.is_ai_operator) return AI_OPERATOR_AVATAR_SRC;
      return avatarSrcFor(row?.player_avatar_id);
    }

    const CAMPAIGN_TASK_ORDER = [
      "collect_wood",
      "place_table",
      "make_wood_pickaxe",
      "collect_stone",
      "make_stone_pickaxe",
      "collect_coal",
      "collect_iron",
      "make_furnace",
      "make_iron_pickaxe",
      "collect_diamond",
    ];
    const LEADERBOARD_TASK_SHORT_TITLES = {
      craftax: {
        collect_wood: "Wood",
        place_table: "Table",
        make_wood_pickaxe: "Wood pick",
        collect_stone: "Stone",
        make_stone_pickaxe: "Stone pick",
        collect_coal: "Coal",
        collect_iron: "Iron",
        make_furnace: "Furnace",
        make_iron_pickaxe: "Iron pick",
        collect_diamond: "Diamond",
      },
      "exo-planet": {
        collect_wood: "Biomass",
        place_table: "Replicator",
        make_wood_pickaxe: "Bone drill",
        collect_stone: "Basalt",
        make_stone_pickaxe: "Rock drill",
        collect_coal: "Energy ore",
        collect_iron: "Titanite",
        make_furnace: "Thermal oven",
        make_iron_pickaxe: "Titan drill",
        collect_diamond: "Core ore",
      },
    };

    function humanLeaderboardTaskShortTitle(taskKey, worldMode) {
      const key = String(taskKey || "").trim();
      const mode = String(worldMode || "craftax").toLowerCase() === "exo-planet" ? "exo-planet" : "craftax";
      return LEADERBOARD_TASK_SHORT_TITLES[mode]?.[key] || key.replaceAll("_", " ");
    }

    function humanLeaderboardLevelsLabel(row) {
      const kind = String(row?.leaderboard_type || "");
      if (kind === "campaign") {
        const completed = Number(row.phase1_completed_levels || 0);
        const total = Number(row.total_levels || 0);
        return total > 0 ? `${completed}/${total}` : String(completed);
      }
      if (kind === "instruction") return "—";
      return "—";
    }

    function humanLeaderboardStepsLabel(row) {
      const kind = String(row?.leaderboard_type || "");
      if (kind === "campaign") {
        const steps = Number(row.agent_steps || 0);
        return steps > 0 ? String(steps) : "—";
      }
      if (kind === "instruction") return "—";
      return "—";
    }

    function humanLeaderboardQuestionsLabel(row) {
      const kind = String(row?.leaderboard_type || "");
      if (kind === "campaign" || kind === "arc") {
        const questions = Number(row.questions || 0);
        return questions > 0 ? String(questions) : "—";
      }
      if (kind === "instruction") {
        const answers = Number(row.human_answers || 0);
        return answers > 0 ? String(answers) : "—";
      }
      return "—";
    }

    function humanLeaderboardLevelBreakdown(row) {
      const kind = String(row?.leaderboard_type || "");
      if (kind !== "campaign") return "";
      const levelSteps = row?.level_steps;
      if (!levelSteps || typeof levelSteps !== "object") return "";
      const worldMode = row.world_mode;
      const parts = CAMPAIGN_TASK_ORDER
        .filter((key) => Object.prototype.hasOwnProperty.call(levelSteps, key))
        .map((key) => `${humanLeaderboardTaskShortTitle(key, worldMode)} ${Number(levelSteps[key] || 0)}`);
      return parts.join(" · ");
    }

    function humanLeaderboardPerLevelCell(row) {
      const breakdown = humanLeaderboardLevelBreakdown(row);
      if (breakdown) return breakdown;
      const kind = String(row?.leaderboard_type || "");
      if (kind === "instruction") {
        const answers = Number(row.human_answers || 0);
        return answers > 0 ? `Instruction · ${answers} answers` : "Instruction";
      }
      return humanLeaderboardDetailsLabel(row);
    }

    function humanLeaderboardProgressLabel(row) {
      const kind = String(row?.leaderboard_type || "");
      if (kind === "campaign") {
        const completed = Number(row.phase1_completed_levels || 0);
        const total = Number(row.total_levels || 0);
        const steps = Number(row.agent_steps || 0);
        const levelsLabel = total > 0 ? `${completed}/${total} levels` : `${completed} levels`;
        return steps > 0 ? `${levelsLabel} · ${steps} steps` : levelsLabel;
      }
      if (kind === "instruction") {
        return `${Number(row.human_answers || 0)} answers`;
      }
      if (kind === "arc") {
        const levels = Number(row.levels_completed || 0);
        if (levels > 0) return `${levels} ${levels === 1 ? "level" : "levels"} completed`;
        return String(row.state || "—");
      }
      return "—";
    }

    function humanLeaderboardDetailsLabel(row) {
      const kind = String(row?.leaderboard_type || "");
      if (kind === "arc") {
        return `${row.actions || 0} actions · ${row.questions || 0}/${row.human_answers || 0} Q/A · ${row.elapsed_seconds || 0}s`;
      }
      if (kind === "campaign") {
        const levelSteps = row?.level_steps;
        const parts = [];
        if (levelSteps && typeof levelSteps === "object") {
          const breakdown = Object.entries(levelSteps)
            .map(([key, value]) => `${key}:${Number(value || 0)}`)
            .join(" ");
          if (breakdown) parts.push(breakdown);
        }
        const questions = Number(row.questions || 0);
        if (questions > 0) parts.push(`${questions} questions`);
        const reason = String(row.finish_reason || "").trim();
        if (reason) parts.push(reason);
        return parts.length ? parts.join(" · ") : "—";
      }
      if (kind === "instruction") {
        return `${row.finish_reason || "reset"}`;
      }
      return "—";
    }

    async function openHumanLeaderboardModal() {
      if (!companionModeEnabled.value && !isArcGame.value) return;
      const el = document.getElementById("arcScoreModal");
      if (!el) return;
      arcScoreModalInstance = arcScoreModalInstance || new bootstrap.Modal(el);
      if (isArcGame.value && done.value) {
        await fetchArcHumanScore();
      }
      if (!arcPlayerName.value.trim() && playerNickname.value.trim()) {
        arcPlayerName.value = playerNickname.value.trim();
      }
      await loadHumanLeaderboard();
      arcScoreModalInstance.show();
    }

    async function openArcScoreModal() {
      await openHumanLeaderboardModal();
    }

    async function loadArcHumanLeaderboard() {
      await loadHumanLeaderboard();
    }

    async function fetchArcHumanScore({ show = false } = {}) {
      if (!isArcGame.value) return null;
      try {
        const resp = await apiFetch("/arc_human_score");
        const data = await resp.json();
        if (!resp.ok || data.ok === false) throw new Error(data.error || `HTTP ${resp.status}`);
        arcScore.value = data.score || null;
        if (show && arcScore.value && !arcScore.value.submitted) {
          await openArcScoreModal();
        }
        return arcScore.value;
      } catch (err) {
        arcScoreError.value = err?.message || "Failed to load ARC score.";
        return null;
      }
    }

    async function submitArcHumanScore() {
      const playerName = (arcPlayerName.value.trim() || playerNickname.value.trim());
      if (!playerName) {
        arcScoreError.value = "Enter a name to submit the score.";
        return;
      }
      arcScoreSubmitting.value = true;
      arcScoreError.value = "";
      try {
        const resp = await apiFetch("/arc_human_score", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            player_name: playerName,
            player_avatar_id: playerAvatarId.value,
          }),
        });
        const data = await resp.json();
        if (!resp.ok || data.ok === false) throw new Error(data.error || `HTTP ${resp.status}`);
        if (data.entry) arcScore.value = { ...(arcScore.value || {}), ...data.entry, submitted: true };
        if (playerName !== playerNickname.value.trim()) {
          playerNickname.value = playerName;
          persistPlayerProfile({ syncServer: true });
        }
        humanLeaderboardRows.value = Array.isArray(data.leaderboard?.rows)
          ? data.leaderboard.rows
          : humanLeaderboardRows.value;
        await loadHumanLeaderboard();
      } catch (err) {
        arcScoreError.value = err?.message || "Failed to submit ARC score.";
      } finally {
        arcScoreSubmitting.value = false;
      }
    }

    function resetArcScoreState() {
      arcScore.value = null;
      arcScoreError.value = "";
      arcScoreSubmitting.value = false;
      arcScorePromptShown.value = false;
      arcPlayerName.value = "";
    }

    function applyModelCheck(data) {
      const check = data && data.model_check;
      invalidModels.value = check && check.checked && Array.isArray(check.invalid) ? check.invalid : [];
    }
    let apiKeyGuidePositionListener = null;
    let lastSessionConfigData = null;
    const companionModeEnabled = ref(false);
    const exoPlanetEnabled = ref(false);
    const gameKind = ref("craftax");
    const arcGameId = ref("ls20");
    const arcGameOptions = ref([
      {
        id: "ar25",
        label: "ar25",
        description: "Interactive ARC-AGI-3 puzzle with human-helper scoring on the ARC leaderboard.",
        human_operator: true,
        ai_operator: false,
      },
      {
        id: "bp35",
        label: "bp35",
        description: "Interactive ARC-AGI-3 puzzle with human-helper scoring on the ARC leaderboard.",
        human_operator: true,
        ai_operator: false,
      },
      {
        id: "lp85",
        label: "lp85",
        description: "Interactive puzzle suite with human-helper scoring on the ARC leaderboard.",
        human_operator: true,
        ai_operator: false,
      },
      {
        id: "ls20",
        label: "ls20",
        description: "Interactive puzzle suite with human-helper scoring on the ARC leaderboard.",
        human_operator: true,
        ai_operator: false,
      },
    ]);
    const arcGamePreviewBroken = ref({});
    const isArcGame = computed(() => gameKind.value === "arc_agi");

    function ensureArcCompanionMode() {
      if (!isArcGame.value) return;
      interactionMode.value = "human";
      companionModeEnabled.value = true;
      if (!companionBenchMaxTicksDirty.value) {
        companionBenchMaxTicksPerTask.value = DEFAULT_ARC_COMPANION_MAX_TICKS_PER_TASK;
      }
    }

    function syncArcCampaignOnConnect() {
      if (!isArcGame.value || !setupComplete.value) return;
      if (status.value !== "connected") return;
      ensureArcCompanionMode();
      setCampaignEnabled(true);
    }

    function syncCompanionCampaignOnConnect() {
      if (isArcGame.value || !companionModeEnabled.value || !setupComplete.value) return;
      if (status.value !== "connected") return;
      if (campaignState.value?.enabled) return;
      setCampaignEnabled(true);
    }
    const arcScoreIsFinal = computed(() => {
      if (arcScore.value?.final) return true;
      const state = String(arcScore.value?.state || "").toUpperCase();
      if (["WIN", "GAME_OVER", "DONE", "LOSE", "LOST"].includes(state)) return true;
      if (!arcMultiLevel.value && Number(arcScore.value?.levels_completed || 0) >= 1) return true;
      return false;
    });
    const isArcImagePrompt = computed(() => isArcGame.value && ["arc_image", "arc_grid_image", "arc_2_image"].includes(megapromptConfigName.value));
    const isArcGridImagePrompt = computed(() => isArcGame.value && megapromptConfigName.value === "arc_grid_image");
    const isArcTwoImagePrompt = computed(() => isArcGame.value && megapromptConfigName.value === "arc_2_image");
    const worldModeLabel = computed(() =>
      isArcGame.value ? `ARC-AGI-3 ${arcGameId.value}` : (exoPlanetEnabled.value ? "Exo-Planet" : "Craftax")
    );
    const operatorPanelTitle = computed(() => isArcGame.value ? "Human helper" : "Operator panel");
    const operatorPanelSubtitle = computed(() => {
      if (isArcGame.value) {
        return playerNickname.value.trim()
          ? `${playerDisplayName.value} — answer the agent's questions or give lightweight hints.`
          : "Answer the agent's questions or give lightweight hints.";
      }
      if (interactionMode.value === "human") {
        return `${playerDisplayName.value} — answer when the agent asks for help.`;
      }
      return "";
    });
    const operatorInputPlaceholder = computed(() => {
      if (agentDirectChatActive.value) return "Message to agent (END. to exit)...";
      if (isArcGame.value) {
        return pendingAgentQuestion.value
          ? "Answer the agent's ARC question..."
          : "Human hint for ARC agent...";
      }
      return interactionMode.value === "human" ? "Answer (or hint: ...)" : "Your question...";
    });
    const operatorEmptyText = computed(() =>
      isArcGame.value
        ? "No ARC helper messages yet. If the agent asks a question, answer below."
        : "No questions yet. Type below and send."
    );
    const composerPanelActive = computed(() => {
      if (agentDirectChatActive.value) return true;
      if (isArcGame.value) return true;
      return interactionMode.value === "human";
    });
    const composerInactiveMessage = computed(() => "The operator is answering the questions");
    const operatorChatExamples = computed(() => {
      if (agentDirectChatActive.value) return [];
      if (isArcGame.value) {
        return [
          { kind: "agent", text: "Should I rotate the red block clockwise?" },
          { kind: "author", name: playerDisplayName.value, avatar: playerAvatarSrc.value, text: "Yes — try rotating it once." },
        ];
      }
      if (interactionMode.value === "human") {
        return [
          { kind: "agent", text: "Should I explore east or craft tools first?" },
          { kind: "author", name: playerDisplayName.value, avatar: playerAvatarSrc.value, text: "Craft a wooden pickaxe, then explore east." },
        ];
      }
      return [
        { kind: "user", text: "Where can I find water?" },
        { kind: "author", name: AI_OPERATOR_DISPLAY_NAME, avatar: AI_OPERATOR_AVATAR_SRC, text: "There is a lake two tiles north of spawn." },
      ];
    });
    const composerResponderIsHuman = computed(() => {
      if (isArcGame.value) return true;
      if (agentDirectChatActive.value) return true;
      return interactionMode.value === "human";
    });
    const composerResponderAvatarSrc = computed(() => {
      if (composerResponderIsHuman.value) return playerAvatarSrc.value;
      return AI_OPERATOR_AVATAR_SRC;
    });
    const composerResponderDisplayName = computed(() => {
      if (composerResponderIsHuman.value) return playerDisplayName.value;
      return AI_OPERATOR_DISPLAY_NAME;
    });
    const composerResponderTooltip = computed(() => {
      if (agentDirectChatActive.value) return "Direct chat with the agent";
      return composerResponderIsHuman.value
        ? "You are answering the questions"
        : "The operator is answering the questions";
    });
    const arcImageWarning = computed(() =>
      isArcImagePrompt.value
        ? `${megapromptConfigName.value} attaches the rendered frame as an image. Use a vision-capable OpenRouter model; text-only or Hub models may ignore the image.`
        : ""
    );
    const worldModeTheme = computed(() =>
      exoPlanetEnabled.value ? "exo-planet" : "craftax"
    );
    const WORLD_MODE_AGENT_ICONS = {
      craftax: "./assets/agent-icon-craftax.png",
      "exo-planet": "./assets/agent-icon-exo-planet.png",
      arc_agi: "./assets/brain-knowledge-icon.png",
    };
    const WORLD_STAT_ICONS = {
      health: "./assets/game-stats/health.png",
      food: "./assets/game-stats/food.png",
      drink: "./assets/game-stats/drink.png",
      energy: "./assets/game-stats/energy.png",
    };
    function worldStatIconSrc(statKey) {
      return WORLD_STAT_ICONS[statKey] || "";
    }
    const INVENTORY_SLOT_ORDER = [
      "wood",
      "stone",
      "coal",
      "iron",
      "diamond",
      "sapling",
      "wood_pickaxe",
      "stone_pickaxe",
      "iron_pickaxe",
      "wood_sword",
      "stone_sword",
      "iron_sword",
    ];
    const inventoryIcons = ref({});
    const inventorySlotOrder = ref(INVENTORY_SLOT_ORDER.slice());
    let inventoryIconsTheme = null;
    async function loadInventoryIcons(force = false) {
      const theme = worldModeTheme.value;
      if (!force && inventoryIconsTheme === theme) return;
      try {
        const resp = await apiFetch(`/inventory_icons?theme=${encodeURIComponent(theme)}`);
        if (!resp.ok) return;
        const data = await resp.json();
        inventoryIcons.value = data.icons || {};
        if (Array.isArray(data.order) && data.order.length) {
          inventorySlotOrder.value = data.order;
        }
        inventoryIconsTheme = theme;
      } catch (err) {
        /* keep previously loaded icons on failure */
      }
    }
    const TILE_INFO = {
      craftax: {
        0: { name: "Unknown", desc: "Undefined tile." },
        1: { name: "World edge", desc: "Map boundary — you can't pass it." },
        2: { name: "Grass", desc: "Walkable ground. You can plant saplings here." },
        3: { name: "Water", desc: "Drink source — restores thirst. You can't walk through it." },
        4: { name: "Stone", desc: "Mine it with a pickaxe to get stone.", mineable: true },
        5: { name: "Tree", desc: "Chop it to get wood.", mineable: true },
        6: { name: "Wood", desc: "A placed wood block.", mineable: true },
        7: { name: "Path", desc: "Walkable ground left after a block was removed." },
        8: { name: "Coal", desc: "Coal ore — mine it with a wood (or better) pickaxe.", mineable: true },
        9: { name: "Iron", desc: "Iron ore — needs a stone pickaxe.", mineable: true },
        10: { name: "Diamond", desc: "Diamond ore — needs an iron pickaxe.", mineable: true },
        11: { name: "Crafting table", desc: "Stand next to it to craft tools." },
        12: { name: "Furnace", desc: "Stand next to it to smelt iron." },
        13: { name: "Sand", desc: "Walkable ground." },
        14: { name: "Lava", desc: "Dangerous! Deals damage and can kill you.", danger: true },
        15: { name: "Sapling", desc: "A growing plant, it will ripen over time." },
        16: { name: "Ripe plant", desc: "Ripe — eat it to restore food.", mineable: true },
      },
      "exo-planet": {
        0: { name: "Unknown", desc: "Undefined tile." },
        1: { name: "World edge", desc: "Map boundary — you can't pass it." },
        2: { name: "Regolith Turf", desc: "Walkable ground cover." },
        3: { name: "Brine Pool", desc: "Dense fluid — restores fluid reserves when you DRINK_BRINE adjacent. Not walkable." },
        4: { name: "Basalt Crust", desc: "Rock outcrop — EXTRACT with a matching drill tier to get Basalt Shard.", mineable: true },
        5: { name: "Xeno-Root Mass", desc: "Organic structure — EXTRACT while facing it to collect Biomass.", mineable: true },
        6: { name: "Biomass block", desc: "A placed biomass block.", mineable: true },
        7: { name: "Survey Trail", desc: "Cleared walkable route." },
        8: { name: "Energy Ore", desc: "Crystallized energy deposit — EXTRACT with a Bone Drill or better.", mineable: true },
        9: { name: "Titanite Ore", desc: "Advanced metal ore — EXTRACT with a Rock Drill or better.", mineable: true },
        10: { name: "Core Ore", desc: "Rare deep-planet crystal — EXTRACT with a Titan Drill.", mineable: true },
        11: { name: "Replicator", desc: "Fabrication station — stand adjacent and face it to MAKE_* tools." },
        12: { name: "Thermal Oven", desc: "Processing chamber — stand adjacent for smelting recipes." },
        13: { name: "Dune Silts", desc: "Loose regolith; walkable ground cover." },
        14: { name: "Magma Vent", desc: "Thermal hazard — deals hull damage. Not walkable.", danger: true },
        15: { name: "Bio-Sprout", desc: "Planted cultivar — it will ripen over time." },
        16: { name: "Mature Bio-Crop", desc: "Ripe harvest — consume to restore nutrient reserves.", mineable: true },
      },
    };
    const MOB_INFO = {
      craftax: {
        zombie: { name: "Zombie", desc: "Hostile mob. It attacks you, especially at night.", danger: true },
        skeleton: { name: "Skeleton", desc: "Hostile mob. It shoots arrows from a distance.", danger: true },
        cow: { name: "Cow", desc: "Passive animal. Hit it to get food." },
      },
      "exo-planet": {
        zombie: { name: "Hostile Scavenger", desc: "Aggressive ground threat — turn to face it and use ENGAGE_HOSTILE.", danger: true },
        skeleton: { name: "Frenzy Stalker", desc: "Aggressive ranged threat — keep cover and ENGAGE_HOSTILE when aligned.", danger: true },
        cow: { name: "Grazer Unit", desc: "Passive creature — interact while facing it to harvest food." },
      },
    };
    const TILE_INFO_FADE_MS = 1000;
    const selectedTile = ref(null);
    const tileInfoFading = ref(false);
    const tileInfoStyle = ref({});
    let tileInfoCloseTimer = 0;
    const selectedTileInfo = computed(() => {
      const tile = selectedTile.value;
      if (!tile) return null;
      if (tile.kind === "arc") {
        return {
          x: tile.x,
          y: tile.y,
          name: tile.name || "ARC object",
          desc: tile.desc || "A visible object in the ARC-AGI-3 frame.",
          isArc: true,
          arcControls: Array.isArray(tile.controls) ? tile.controls : [],
          isCreature: false,
          mineable: false,
          danger: false,
        };
      }
      const theme = exoPlanetEnabled.value ? "exo-planet" : "craftax";
      if (tile.mobType) {
        const mobTable = MOB_INFO[theme] || MOB_INFO.craftax;
        const mob = mobTable[tile.mobType] || { name: "Creature", desc: "A living creature." };
        return {
          x: tile.x,
          y: tile.y,
          name: mob.name,
          desc: mob.desc,
          isCreature: true,
          mineable: false,
          danger: Boolean(mob.danger),
        };
      }
      const table = TILE_INFO[theme] || TILE_INFO.craftax;
      const entry = (tile.blockId != null && table[tile.blockId]) || {
        name: "Unknown",
        desc: "No data for this tile.",
      };
      return {
        x: tile.x,
        y: tile.y,
        name: entry.name,
        desc: entry.desc,
        isCreature: false,
        mineable: Boolean(entry.mineable),
        danger: Boolean(entry.danger),
      };
    });
    function mobTypeAt(x, y) {
      const mobs = currentFrame.value?.world?.mobs;
      if (!Array.isArray(mobs)) return null;
      const hit = mobs.find((m) => m.x === x && m.y === y);
      return hit ? hit.type : null;
    }
    function clearTileInfoTimers() {
      if (tileInfoCloseTimer) { clearTimeout(tileInfoCloseTimer); tileInfoCloseTimer = 0; }
    }
    function computeTileInfoStyle(tile) {
      const margin = 14;
      const cardW = 340;
      const cardH = 210;
      const gap = 18;
      const half = (tile.tileScreenSize || 0) / 2;
      const vw = window.innerWidth;
      const vh = window.innerHeight;

      // The side panels (stats/operator) overlay the map on the right, so the
      // card must stay to the left of whichever panel starts first.
      let rightLimit = vw - margin;
      document.querySelectorAll(".operator-panel-shell, .stats-panel-shell").forEach((el) => {
        const r = el.getBoundingClientRect();
        if (r.width > 0 && r.height > 0 && r.left < rightLimit) {
          rightLimit = r.left - margin;
        }
      });
      const leftLimit = margin;
      const maxLeft = Math.max(leftLimit, rightLimit - cardW);

      const px = tile.screenX ?? vw / 2;
      const py = tile.screenY ?? vh / 2;

      // Prefer placing the card to the right of the tile; flip to the left when
      // it would collide with the side panels.
      let left = px + half + gap;
      if (left + cardW > rightLimit) {
        left = px - half - gap - cardW;
      }
      left = Math.min(Math.max(leftLimit, left), maxLeft);

      let top = py - cardH / 2;
      top = Math.min(Math.max(margin, top), vh - cardH - margin);
      return { left: `${Math.round(left)}px`, top: `${Math.round(top)}px` };
    }
    function showTileInfo(tile) {
      clearTileInfoTimers();
      tileInfoFading.value = false;
      selectedTile.value = { ...tile, mobType: mobTypeAt(tile.x, tile.y) };
      tileInfoStyle.value = computeTileInfoStyle(tile);
    }
    function arcFrameGridRows() {
      const grid = String(currentFrame.value?.arc?.frame_grid || "");
      const rows = grid.split("\n").filter((row) => row.length > 0);
      return rows.length ? rows : [];
    }
    function arcFrameGridSize() {
      const rows = arcFrameGridRows();
      if (!rows.length) return { w: 64, h: 64 };
      return {
        w: Math.max(...rows.map((row) => row.length)),
        h: rows.length,
      };
    }
    function arcCellAt(rows, x, y) {
      if (!rows.length || y < 0 || y >= rows.length) return "";
      const row = rows[y] || "";
      return x >= 0 && x < row.length ? row[x].toLowerCase() : "";
    }
    function arcPointsForCells(rows, cells) {
      const wanted = new Set(cells.map((item) => String(item).toLowerCase()));
      const points = [];
      rows.forEach((row, y) => {
        for (let x = 0; x < row.length; x++) {
          if (wanted.has(String(row[x]).toLowerCase())) points.push({ x, y });
        }
      });
      return points;
    }
    function arcBoundsForCells(rows, cells) {
      const points = arcPointsForCells(rows, cells);
      if (!points.length) return null;
      const xs = points.map((p) => p.x);
      const ys = points.map((p) => p.y);
      return {
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minY: Math.min(...ys),
        maxY: Math.max(...ys),
      };
    }
    function arcPointNearCells(rows, x, y, cells, radius = 1) {
      for (let yy = y - radius; yy <= y + radius; yy++) {
        for (let xx = x - radius; xx <= x + radius; xx++) {
          if (cells.includes(arcCellAt(rows, xx, yy))) return true;
        }
      }
      return false;
    }
    function arcPointInBounds(x, y, bounds, pad = 0) {
      return Boolean(
        bounds
        && x >= bounds.minX - pad
        && x <= bounds.maxX + pad
        && y >= bounds.minY - pad
        && y <= bounds.maxY + pad
      );
    }
    function arcColorName(cell) {
      const names = {
        "0": "white",
        "1": "off-white",
        "2": "light-gray",
        "3": "gray",
        "4": "dark-gray",
        "5": "black",
        "6": "magenta",
        "7": "light-magenta",
        "8": "red",
        "9": "blue",
        a: "light-blue",
        b: "yellow",
        c: "orange",
        d: "maroon",
        e: "green",
        f: "purple",
      };
      return names[String(cell || "").toLowerCase()] || "unknown";
    }
    function arcAvailableActions() {
      const raw = currentFrame.value?.arc?.available_actions;
      if (!Array.isArray(raw)) return new Set();
      return new Set(
        raw.map((action) => String(action || "").trim().toUpperCase()).filter(Boolean),
      );
    }
    function arcControlsFor(kind, x, y) {
      const avail = arcAvailableActions();
      const labels = [];
      const add = (action, label) => {
        if (avail.has(action)) labels.push(label);
      };
      if (kind === "move") {
        add("ACTION1", "ACTION1 ↑ move up");
        add("ACTION2", "ACTION2 ↓ move down");
        add("ACTION3", "ACTION3 ← move left");
        add("ACTION4", "ACTION4 → move right");
      } else if (kind === "step_on") {
        add("ACTION1", "ACTION1 ↑ step on");
        add("ACTION2", "ACTION2 ↓ step on");
        add("ACTION3", "ACTION3 ← step on");
        add("ACTION4", "ACTION4 → step on");
        add("ACTION5", "ACTION5 interact on tile");
      } else if (kind === "click") {
        if (avail.has("ACTION6")) labels.push(`ACTION6 ${x} ${y} click here`);
      } else if (kind === "interact") {
        add("ACTION5", "ACTION5 interact / select");
      } else if (kind === "undo") {
        add("ACTION7", "ACTION7 undo");
      } else if (kind === "view") {
        labels.push("View only");
      }
      return labels;
    }
    function resolveArcControls(controlKeys, x, y) {
      const keys = Array.isArray(controlKeys) && controlKeys.length ? controlKeys : ["view"];
      const out = [];
      const seen = new Set();
      keys.forEach((key) => {
        arcControlsFor(key, x, y).forEach((label) => {
          if (seen.has(label)) return;
          seen.add(label);
          out.push(label);
        });
      });
      return out.length ? out : ["View only"];
    }
    function arcLs20TileHint(x, y, cell, rows) {
      const playerBounds = arcBoundsForCells(rows, ["c"]);
      if (arcPointInBounds(x, y, playerBounds, 2)) {
        return {
          name: "Player character",
          desc: "This is your controllable character. Use the movement actions to navigate it through the maze.",
          controlKeys: ["move"],
        };
      }
      if (arcPointNearCells(rows, x, y, ["0", "1"], 1)) {
        return {
          name: "White cross switch",
          desc: "This cross changes the shape shown in the lower-left corner. Step on it until that shape matches the target shape at the maze exit.",
          controlKeys: ["step_on"],
        };
      }
      if (x <= 11 && y >= 52 && ["9", "5", "3"].includes(cell)) {
        return {
          name: "Current shape indicator",
          desc: "This shows the shape you are currently trying to match. The white cross can rotate or change this indicator.",
          controlKeys: ["view"],
        };
      }
      if (x >= 32 && x <= 41 && y <= 16 && ["9", "5", "3"].includes(cell)) {
        return {
          name: "Target shape",
          desc: "This is the target shape at the end of the maze. Match the lower-left shape to this before navigating to the exit.",
          controlKeys: ["view"],
        };
      }
      if (y < 58 && (cell === "b" || arcPointNearCells(rows, x, y, ["b"], 1))) {
        return {
          name: "Timer refill",
          desc: "A yellow ring with a hole. It helps refill the yellow timer so the player has enough time to finish the level.",
          controlKeys: ["step_on"],
        };
      }
      if (cell === "5") {
        return { name: "Wall", desc: "A solid maze boundary. The player cannot move through this cell.", controlKeys: ["view"] };
      }
      if (cell === "3") {
        return { name: "Maze floor", desc: "Open maze floor. This is the kind of area the player can navigate through.", controlKeys: ["move"] };
      }
      if (cell === "4") {
        return { name: "Outer background", desc: "Dark background outside the active maze path.", controlKeys: ["view"] };
      }
      if (cell === "b") {
        return { name: "Status bar", desc: "A visual status strip for the ARC game, not the player character.", controlKeys: ["view"] };
      }
      if (cell === "8") {
        return { name: "Status marker", desc: "A marker on the game status strip.", controlKeys: ["view"] };
      }
      return null;
    }
    function arcLp85TileHint(x, y, cell) {
      if (cell === "8" && x <= 8) {
        return {
          name: "Red loop button",
          desc: "A red button that rotates the loop of puzzle tiles. Use it to shift tiles around the loop.",
          controlKeys: ["click", "interact"],
        };
      }
      if (cell === "e" && x >= 54) {
        return {
          name: "Green loop button",
          desc: "A green button that rotates the loop of puzzle tiles. It shifts the same tile loop in a game-specific direction.",
          controlKeys: ["click", "interact"],
        };
      }
      if (cell === "e" && x <= 1) {
        return {
          name: "Level timer",
          desc: "The green strip on the left is the level timer.",
          controlKeys: ["view"],
        };
      }
      if (cell === "b") {
        if (x < 32) {
          return {
            name: "Target zone",
            desc: "The left yellow highlighted area is the target zone. Move the yellow tile into this zone.",
            controlKeys: ["view"],
          };
        }
        return {
          name: "Yellow tile",
          desc: "The right yellow highlighted tile is the tile that needs to be moved into the target zone.",
          controlKeys: ["view"],
        };
      }
      if (["1", "2", "8", "9", "a", "e", "f"].includes(cell)) {
        return {
          name: `${arcColorName(cell)} loop tile`,
          desc: "Part of the tile loop that rotates when the red or green loop button is used.",
          controlKeys: ["view"],
        };
      }
      if (cell === "5") {
        return {
          name: "Empty slot",
          desc: "A black slot or empty area in the puzzle layout.",
          controlKeys: ["view"],
        };
      }
      if (["3", "4"].includes(cell)) {
        return {
          name: "Board background",
          desc: "The neutral board/background area around the active puzzle objects.",
          controlKeys: ["view"],
        };
      }
      return null;
    }
    function arcAr25TileHint(x, y, cell) {
      if (cell === "b" && x === 63) {
        return {
          name: "Level timer",
          desc: "The yellow line on the right is the timer for completing the level.",
          controlKeys: ["view"],
        };
      }
      if (cell === "a" && x >= 30 && x <= 32) {
        return {
          name: "Center divider",
          desc: "A light-blue divider between the left and right sides of the puzzle board.",
          controlKeys: ["view"],
        };
      }
      if (cell === "b" && x >= 51 && x <= 59 && y >= 45 && y <= 53) {
        return {
          name: "Target object",
          desc: "The yellow object is the target. Move the gray synchronized object onto this target.",
          controlKeys: ["view"],
        };
      }
      if ((cell === "5" || cell === "0") && x >= 18 && x <= 26 && y >= 15 && y <= 23) {
        return {
          name: "Movable object",
          desc: "The black figure with white holes is the object you can move.",
          controlKeys: ["move"],
        };
      }
      if (cell === "4" && x >= 36 && x <= 44 && y >= 15 && y <= 23) {
        return {
          name: "Synchronized object",
          desc: "The gray object moves synchronously with the black figure with white holes.",
          controlKeys: ["view"],
        };
      }
      if (cell === "9") {
        return {
          name: "Blue board area",
          desc: "The main blue background of the ar25 puzzle board.",
          controlKeys: ["view"],
        };
      }
      if (cell === "b" && y === 63) {
        return {
          name: "Board edge marker",
          desc: "A thin yellow edge marker at the boundary of the rendered ARC frame.",
          controlKeys: ["view"],
        };
      }
      return null;
    }
    function arcBp35TileHint(x, y, cell) {
      if (["9", "b"].includes(cell) && x >= 19 && x <= 23 && y >= 36 && y <= 41) {
        return {
          name: "Player marker",
          desc: "This is the controllable blue/yellow marker. Use movement actions to navigate it through the corridors.",
          controlKeys: ["move"],
        };
      }
      if (cell === "e") {
        return {
          name: "Breakable block",
          desc: "A green block that can be broken with ACTION6 click coordinates to open a path deeper into the maze.",
          controlKeys: ["click"],
        };
      }
      if (cell === "a") {
        return {
          name: "Open corridor",
          desc: "A light-blue open area of the map. This is the main navigable space around the player marker.",
          controlKeys: ["move"],
        };
      }
      if (cell === "5") {
        return {
          name: "Wall",
          desc: "A solid black wall or boundary. The player marker cannot move through this area.",
          controlKeys: ["view"],
        };
      }
      if (cell === "3") {
        return {
          name: "Wall detail",
          desc: "A gray marker embedded in the wall pattern. It helps distinguish the board texture but is not the player marker.",
          controlKeys: ["view"],
        };
      }
      if (cell === "0" && y >= 63) {
        return {
          name: "Frame boundary",
          desc: "The bottom frame boundary of the rendered ARC observation.",
          controlKeys: ["view"],
        };
      }
      return null;
    }
    function genericArcTileHint(gameId, cell) {
      if (!cell) {
        return {
          name: "ARC frame",
          desc: "The rendered game frame. Click visible objects to inspect their coordinates and color.",
          controlKeys: ["view"],
        };
      }
      if (["3", "4"].includes(cell)) {
        return {
          name: "Background",
          desc: "A neutral background or board cell in this ARC-AGI-3 game.",
          controlKeys: ["view"],
        };
      }
      if (cell === "5") {
        return {
          name: "Solid boundary",
          desc: "A dark boundary, wall, or empty slot depending on the game.",
          controlKeys: ["view"],
        };
      }
      if (cell === "7") {
        return {
          name: "Final target",
          desc: "The final target that the agent should reach to complete the level.",
          controlKeys: ["step_on"],
        };
      }
      return {
        name: `${arcColorName(cell)} ARC object`,
        desc: `A visible ${arcColorName(cell)} object in ${gameId}. The exact rule is game-specific; use its coordinate if the agent needs to refer to it.`,
        controlKeys: ["click", "move"],
      };
    }
    function buildArcTileHint(x, y) {
      const rows = arcFrameGridRows();
      const cell = arcCellAt(rows, x, y);
      const gameId = String(currentFrame.value?.arc?.game_id || arcGameId.value || "").toLowerCase();
      const specific = gameId === "ls20"
        ? arcLs20TileHint(x, y, cell, rows)
        : gameId === "lp85"
          ? arcLp85TileHint(x, y, cell)
          : gameId === "ar25"
            ? arcAr25TileHint(x, y, cell)
            : gameId === "bp35"
              ? arcBp35TileHint(x, y, cell)
              : null;
      const hint = specific || genericArcTileHint(gameId || "ARC-AGI-3", cell);
      return {
        ...hint,
        controls: resolveArcControls(hint.controlKeys, x, y),
      };
    }
    function arcImageDrawMetrics(img, boxRect) {
      const frameW = Math.max(1, Number(currentFrame.value?.w || img?.naturalWidth || 1));
      const frameH = Math.max(1, Number(currentFrame.value?.h || img?.naturalHeight || 1));
      const imageAspect = frameW / frameH;
      const boxAspect = boxRect.width / boxRect.height;
      let drawW = boxRect.width;
      let drawH = boxRect.height;
      let drawLeft = boxRect.left;
      let drawTop = boxRect.top;
      if (boxAspect > imageAspect) {
        drawW = boxRect.height * imageAspect;
        drawLeft = boxRect.left + (boxRect.width - drawW) / 2;
      } else if (boxAspect < imageAspect) {
        drawH = boxRect.width / imageAspect;
        drawTop = boxRect.top + (boxRect.height - drawH) / 2;
      }
      return { drawLeft, drawTop, drawW, drawH, frameW, frameH };
    }
    const ARC_HOVER_DWELL_MS = 2000;
    const ARC_HOVER_BLINK_PERIOD_MS = 850;
    const ARC_HOVER_RIPPLE_PERIOD_MS = 2600;
    const ARC_HOVER_RIPPLE_COUNT = 3;
    const ARC_HOVER_RADIUS_CELLS = 5;
    function arcFrameTileMetrics(metrics) {
      const { w: gridW, h: gridH } = arcFrameGridSize();
      const tilePxX = metrics.drawW / gridW;
      const tilePxY = metrics.drawH / gridH;
      return { gridW, gridH, tilePxX, tilePxY, tilePx: Math.min(tilePxX, tilePxY) };
    }
    function arcPointerEventToCoord(event) {
      const shell = arcFrameShellEl.value;
      const img = shell?.querySelector?.(".arc-frame-image");
      if (!shell || !img || !currentFrame.value) return null;
      const shellRect = shell.getBoundingClientRect();
      const metrics = arcImageDrawMetrics(img, shellRect);
      const { gridW, gridH, tilePxX, tilePxY, tilePx } = arcFrameTileMetrics(metrics);
      const rx = (event.clientX - metrics.drawLeft) / metrics.drawW;
      const ry = (event.clientY - metrics.drawTop) / metrics.drawH;
      if (rx < 0 || rx > 1 || ry < 0 || ry > 1) return null;
      const x = Math.max(0, Math.min(gridW - 1, Math.floor(rx * gridW)));
      const y = Math.max(0, Math.min(gridH - 1, Math.floor(ry * gridH)));
      return {
        x,
        y,
        localX: event.clientX - shellRect.left,
        localY: event.clientY - shellRect.top,
        screenX: event.clientX,
        screenY: event.clientY,
        tileScreenSize: Math.max(6, tilePx * ARC_HOVER_RADIUS_CELLS * 2),
        metrics,
        shellRect,
      };
    }
    let arcHoverTile = null;
    let arcHoverActive = false;
    let arcHoverBlinkStart = 0;
    let arcHoverDwellTimer = 0;
    let arcHoverRafId = 0;
    let arcFrameResizeObserver = null;
    function clearArcFrameHover() {
      if (arcHoverDwellTimer) {
        clearTimeout(arcHoverDwellTimer);
        arcHoverDwellTimer = 0;
      }
      const hadTile = arcHoverTile;
      arcHoverTile = null;
      arcHoverActive = false;
      if (hadTile) handleTileHoverChange(null);
      stopArcFrameHoverLoop();
      const canvas = arcFrameCanvasEl.value;
      canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    }
    function stopArcFrameHoverLoop() {
      if (arcHoverRafId) {
        cancelAnimationFrame(arcHoverRafId);
        arcHoverRafId = 0;
      }
    }
    function scheduleArcFrameHoverLoop() {
      if (arcHoverRafId) return;
      const tick = () => {
        arcHoverRafId = requestAnimationFrame(tick);
        if (arcHoverActive && arcHoverTile) drawArcFrameHover();
        else stopArcFrameHoverLoop();
      };
      arcHoverRafId = requestAnimationFrame(tick);
    }
    function syncArcFrameCanvasSize() {
      const canvas = arcFrameCanvasEl.value;
      const shell = arcFrameShellEl.value;
      if (!canvas || !shell) return;
      const width = Math.max(1, Math.round(shell.clientWidth));
      const height = Math.max(1, Math.round(shell.clientHeight));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }
    function drawArcFrameHover() {
      const canvas = arcFrameCanvasEl.value;
      const shell = arcFrameShellEl.value;
      const img = shell?.querySelector?.(".arc-frame-image");
      if (!canvas || !shell || !img || !arcHoverActive || !arcHoverTile) return;
      syncArcFrameCanvasSize();
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const shellRect = shell.getBoundingClientRect();
      const metrics = arcImageDrawMetrics(img, shellRect);
      const { tilePxX, tilePxY, tilePx } = arcFrameTileMetrics(metrics);
      const offsetX = metrics.drawLeft - shellRect.left;
      const offsetY = metrics.drawTop - shellRect.top;
      const cx = offsetX + (arcHoverTile.x + 0.5) * tilePxX;
      const cy = offsetY + (arcHoverTile.y + 0.5) * tilePxY;
      const elapsed = performance.now() - arcHoverBlinkStart;
      const hoverRadiusPx = tilePx * ARC_HOVER_RADIUS_CELLS;
      const pulse = (Math.sin((elapsed / ARC_HOVER_BLINK_PERIOD_MS) * Math.PI * 2) + 1) / 2;
      ctx.fillStyle = `rgba(120, 220, 255, ${0.10 + pulse * 0.12})`;
      ctx.beginPath();
      ctx.arc(cx, cy, hoverRadiusPx, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = Math.max(1.5, tilePx * 0.35);
      for (let i = 0; i < ARC_HOVER_RIPPLE_COUNT; i++) {
        const t = ((elapsed / ARC_HOVER_RIPPLE_PERIOD_MS) + i / ARC_HOVER_RIPPLE_COUNT) % 1;
        const radius = (1 - (1 - t) * (1 - t)) * hoverRadiusPx;
        if (radius <= 0.5) continue;
        const alpha = Math.sin(t * Math.PI) * 0.7;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(170, 235, 255, ${alpha})`;
        ctx.stroke();
      }
    }
    function setArcFrameHoverTile(tile) {
      if (!tile) {
        clearArcFrameHover();
        return;
      }
      if (arcHoverTile && arcHoverTile.x === tile.x && arcHoverTile.y === tile.y) {
        return;
      }
      handleTileHoverChange({ x: tile.x, y: tile.y });
      if (arcHoverDwellTimer) clearTimeout(arcHoverDwellTimer);
      const hadActive = arcHoverActive;
      arcHoverActive = false;
      arcHoverTile = { x: tile.x, y: tile.y };
      if (hadActive) {
        const canvas = arcFrameCanvasEl.value;
        canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      }
      arcHoverDwellTimer = window.setTimeout(() => {
        arcHoverDwellTimer = 0;
        arcHoverActive = true;
        arcHoverBlinkStart = performance.now();
        scheduleArcFrameHoverLoop();
        drawArcFrameHover();
      }, ARC_HOVER_DWELL_MS);
    }
    function onArcFramePointerMove(event) {
      if (!isArcGame.value) return;
      const coord = arcPointerEventToCoord(event);
      if (!coord) {
        clearArcFrameHover();
        return;
      }
      setArcFrameHoverTile(coord);
    }
    function onArcFramePointerLeave() {
      clearArcFrameHover();
    }
    function onArcFrameClick(event) {
      if (!isArcGame.value) return;
      closeReasoningPanel();
      showArcTileInfoFromEvent(event);
    }
    function ensureArcFrameHoverObserver() {
      if (arcFrameResizeObserver || !arcFrameShellEl.value || typeof ResizeObserver === "undefined") return;
      arcFrameResizeObserver = new ResizeObserver(() => {
        syncArcFrameCanvasSize();
        if (arcHoverActive) drawArcFrameHover();
      });
      arcFrameResizeObserver.observe(arcFrameShellEl.value);
    }
    function destroyArcFrameHoverObserver() {
      if (arcFrameResizeObserver) {
        arcFrameResizeObserver.disconnect();
        arcFrameResizeObserver = null;
      }
      clearArcFrameHover();
      stopArcFrameHoverLoop();
      const canvas = arcFrameCanvasEl.value;
      canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    }
    function showArcTileInfoFromEvent(event) {
      const coord = arcPointerEventToCoord(event);
      if (!coord) return;
      const hint = buildArcTileHint(coord.x, coord.y);
      clearTileInfoTimers();
      tileInfoFading.value = false;
      selectedTile.value = {
        kind: "arc",
        ...coord,
        name: hint.name,
        desc: hint.desc,
        controls: hint.controls,
      };
      tileInfoStyle.value = computeTileInfoStyle(selectedTile.value);
    }
    function closeTileInfo() {
      clearTileInfoTimers();
      if (!selectedTile.value) return;
      tileInfoFading.value = true;
      tileInfoCloseTimer = window.setTimeout(() => {
        selectedTile.value = null;
        tileInfoFading.value = false;
      }, TILE_INFO_FADE_MS);
    }
    function handleTileHoverChange(tile) {
      const sel = selectedTile.value;
      if (!sel || tileInfoFading.value) return;
      if (!tile || tile.x !== sel.x || tile.y !== sel.y) {
        closeTileInfo();
      }
    }
    const worldModeAgentIconSrc = computed(
      () => WORLD_MODE_AGENT_ICONS[isArcGame.value ? "arc_agi" : worldModeTheme.value] || WORLD_MODE_AGENT_ICONS.craftax
    );
    const worldModeToggleIconSrc = worldModeAgentIconSrc;
    const worldModeToggleLabel = worldModeLabel;
    const campaignBenchmarkCompact = computed(() => {
      const mode = worldModeTheme.value;
      const block = campaignBenchmark.value?.[mode];
      return Array.isArray(block?.compact) ? block.compact : [];
    });
    const campaignBenchmarkRuns = computed(() => {
      const mode = worldModeTheme.value;
      const block = campaignBenchmark.value?.[mode];
      return Number(block?.exploration_runs ?? block?.kara_runs ?? 0);
    });
    const statsPanelStyle = computed(() => {
      if (statsPanelCollapsed.value) return { "--stats-panel-width": "2.5rem" };
      const width = statsPanelWidth.value;
      return {
        flexBasis: `${width}px`,
        width: `${width}px`,
        minWidth: `${width}px`,
        maxWidth: `${width}px`,
        "--stats-panel-width": `${width}px`,
      };
    });
    const operatorPanelStyle = computed(() => {
      if (operatorPanelCollapsed.value) return { "--operator-panel-width": "0px" };
      const width = operatorPanelWidth.value;
      return {
        flexBasis: `${width}px`,
        width: `${width}px`,
        minWidth: `${width}px`,
        maxWidth: `${width}px`,
        "--operator-panel-width": `${width}px`,
      };
    });
    let themeMediaQuery = null;
    const SESSION_PREFS_STORAGE_KEY = "playWebSessionPrefs";
    const SESSION_PREFS_VERSION = 7;
    const themePref = ref(window.PlayWebTheme?.getStoredThemePref?.() || "system");

    function setThemePref(pref) {
      themePref.value = pref;
      window.PlayWebTheme?.setStoredThemePref?.(pref);
      window.PlayWebTheme?.applyTheme?.(pref);
      nextTick(() => {
        updateStatsChart();
        updateCurrentCharts();
      });
    }

    function chartAxisOptions(axisSize) {
      const colors = window.PlayWebTheme?.chartThemeColors?.() || {};
      return {
        ticks: { color: colors.tick || "#9ca3af", font: { size: axisSize } },
        grid: { color: colors.grid || "rgba(61, 66, 77, 0.5)" },
      };
    }

    function alignMegapromptToWorldMode() {
      const worldMode = worldModeFromExoEnabled(exoPlanetEnabled.value);
      megapromptConfigName.value = coerceMegapromptForWorldMode(
        megapromptConfigName.value,
        worldMode,
        gameKind.value,
      );
    }

    function normalizeOperatorSelection() {
      if (isArcGame.value) {
        interactionMode.value = "human";
        selectedExperts.value = [];
        forcedExpert.value = null;
        return;
      }
      if (interactionMode.value === "human") {
        selectedExperts.value = [];
        return;
      }
      interactionMode.value = "oracle";
      const goalExpert = resolveExpertName("goal") || "goal_expert";
      selectedExperts.value = [goalExpert];
      forcedExpert.value = goalExpert;
    }

    function buildAgentRuntimeOverrides() {
      alignMegapromptToWorldMode();
      const overrides = {
        game_kind: gameKind.value,
        arc_game_id: arcGameId.value,
        exo_planet_enabled: exoPlanetEnabled.value,
      };
      if (featureFlags.value.model_selection) {
        overrides.active_agent_model = activeAgentModel.value;
        overrides.active_agent_mode = activeAgentMode.value;
      }
      if (featureFlags.value.observation_format_selection) {
        overrides.megaprompt_config_name = megapromptConfigName.value;
      }
      if (featureFlags.value.arc_prompt_override) {
        overrides.arc_prompt_extra = arcPromptExtra.value;
      }
      return overrides;
    }

    function collectSettingsSnapshot() {
      return {
        version: SESSION_PREFS_VERSION,
        interaction_mode: interactionMode.value,
        allowed_experts: [...selectedExperts.value],
        forced_expert: interactionMode.value === "human" ? null : forcedExpert.value,
        auto_intent_enabled: false,
        megaprompt_config_name: megapromptConfigName.value,
        agent_goal: agentGoal.value,
        expert_models: { ...expertModelFields.value },
        expert_modes: { ...expertModeFields.value },
        active_agent_model: activeAgentModel.value,
        active_agent_models: { ...agentModelByContext.value },
        active_agent_mode: activeAgentMode.value,
        default_agent_steps: clampAgentStepsPerClick(agentStepsPerClick.value),
        game_kind: gameKind.value,
        arc_game_id: arcGameId.value,
        exo_planet_enabled: exoPlanetEnabled.value,
        companion_mode_enabled: companionModeEnabled.value,
        player_nickname: playerNickname.value,
        player_avatar_id: playerAvatarId.value,
      };
    }

    function applyPlayerProfileFromSnapshot(snapshot) {
      if (!snapshot || typeof snapshot !== "object") return;
      if ("player_nickname" in snapshot) playerNickname.value = String(snapshot.player_nickname || "");
      if ("player_avatar_id" in snapshot) playerAvatarId.value = normalizeAvatarId(snapshot.player_avatar_id);
      savePlayerProfile({
        nickname: playerNickname.value,
        avatar_id: playerAvatarId.value,
      });
    }

    function persistPlayerProfile({ syncServer = false } = {}) {
      const saved = savePlayerProfile({
        nickname: playerNickname.value,
        avatar_id: playerAvatarId.value,
      });
      playerNickname.value = saved.nickname;
      playerAvatarId.value = saved.avatar_id;
      if (!arcPlayerName.value.trim() && saved.nickname) {
        arcPlayerName.value = saved.nickname;
      }
      persistSharedSettingsToStorage();
      if (syncServer && setupComplete.value) {
        saveSessionConfig(undefined, { silent: true });
      }
    }

    function onPlayerNicknameInput() {
      wizardStepTouched.value = {
        ...wizardStepTouched.value,
        profile: isPlayerProfileComplete({ nickname: playerNickname.value }),
      };
      persistPlayerProfile();
    }

    function selectPlayerAvatar(avatarId, { syncServer = false } = {}) {
      playerAvatarId.value = normalizeAvatarId(avatarId);
      wizardStepTouched.value = {
        ...wizardStepTouched.value,
        profile: isPlayerProfileComplete({ nickname: playerNickname.value }),
      };
      persistPlayerProfile({ syncServer });
    }

    function showHumanOperatorIdentity(m) {
      return showHumanOperatorIdentityForMessage(m, interactionMode.value);
    }

    function showAiOperatorIdentity(m) {
      return showAiOperatorIdentityForMessage(m, interactionMode.value);
    }

    function showOperatorAuthorIdentity(m) {
      return showOperatorAuthorIdentityForMessage(m, interactionMode.value);
    }

    function operatorAuthorAvatarSrc(m) {
      if (showHumanOperatorIdentityForMessage(m, interactionMode.value)) return playerAvatarSrc.value;
      if (showAiOperatorIdentityForMessage(m, interactionMode.value)) return AI_OPERATOR_AVATAR_SRC;
      return "";
    }

    function operatorAuthorDisplayName(m) {
      if (showHumanOperatorIdentityForMessage(m, interactionMode.value)) return playerDisplayName.value;
      if (showAiOperatorIdentityForMessage(m, interactionMode.value)) return AI_OPERATOR_DISPLAY_NAME;
      return "";
    }

    function applySettingsSnapshot(snapshot) {
      if (!snapshot || typeof snapshot !== "object") return;
      if (snapshot.interaction_mode) interactionMode.value = snapshot.interaction_mode;
      if (Array.isArray(snapshot.allowed_experts)) selectedExperts.value = snapshot.allowed_experts;
      normalizeOperatorSelection();
      if (snapshot.megaprompt_config_name) {
        megapromptConfigName.value = String(snapshot.megaprompt_config_name);
      }
      if (snapshot.agent_goal) agentGoal.value = snapshot.agent_goal;
      if (snapshot.expert_models && typeof snapshot.expert_models === "object") {
        expertModelFields.value = { ...expertModelFields.value, ...snapshot.expert_models };
      }
      if (snapshot.expert_modes && typeof snapshot.expert_modes === "object") {
        expertModeFields.value = {
          ...expertModeFields.value,
          ...Object.fromEntries(
            Object.entries(snapshot.expert_modes).map(([k, v]) => [
              k,
              String(v || "hub").toLowerCase() === "openrouter" ? "openrouter" : "hub",
            ]),
          ),
        };
      }
      if (snapshot.active_agent_models && typeof snapshot.active_agent_models === "object") {
        agentModelByContext.value = {
          arc_agi: String(snapshot.active_agent_models.arc_agi || ""),
          craftax: String(snapshot.active_agent_models.craftax || ""),
        };
      }
      if (snapshot.active_agent_model) {
        const key = agentModelContextKey();
        if (!String(agentModelByContext.value[key] || "").trim()) {
          agentModelByContext.value = {
            ...agentModelByContext.value,
            [key]: String(snapshot.active_agent_model),
          };
        }
      }
      restoreAgentModelForCurrentContext();
      ensureDemoAgentModelPreset();
      activeAgentMode.value = "openrouter";
      if (snapshot.default_agent_steps != null) {
        agentStepsPerClick.value = clampAgentStepsPerClick(snapshot.default_agent_steps);
      }
      if (snapshot.game_kind) gameKind.value = String(snapshot.game_kind) === "arc_agi" ? "arc_agi" : "craftax";
      if (snapshot.arc_game_id) arcGameId.value = String(snapshot.arc_game_id);
      ensureDemoArcGameSelection();
      if (typeof snapshot.exo_planet_enabled === "boolean") {
        exoPlanetEnabled.value = snapshot.exo_planet_enabled;
      }
      if (typeof snapshot.companion_mode_enabled === "boolean") {
        companionModeEnabled.value = snapshot.companion_mode_enabled;
      }
      applyPlayerProfileFromSnapshot(snapshot);
      if (isArcGame.value) {
        ensureArcCompanionMode();
      }
      alignMegapromptToWorldMode();
    }

    function applyConfigResponse(data) {
      lastSessionConfigData = data;
      applyAppCapabilities(data);
      interactionMode.value = data.interaction_mode || interactionMode.value;
      selectedExperts.value = Array.isArray(data.allowed_experts) ? data.allowed_experts : selectedExperts.value;
      normalizeOperatorSelection();
      alignMegapromptToWorldMode();
      if (data.expert_models && typeof data.expert_models === "object") {
        expertModelFields.value = {
          ...expertModelFields.value,
          ...Object.fromEntries(
            Object.entries(data.expert_models).map(([k, v]) => [k, String(v || "")]),
          ),
        };
      }
      if (data.expert_modes && typeof data.expert_modes === "object") {
        expertModeFields.value = {
          ...expertModeFields.value,
          ...Object.fromEntries(
            Object.entries(data.expert_modes).map(([k, v]) => [
              k,
              String(v || "hub").toLowerCase() === "openrouter" ? "openrouter" : "hub",
            ]),
          ),
        };
      }
      activeAgentModel.value = String(data.active_agent_model || activeAgentModel.value);
      agentModelByContext.value = {
        ...agentModelByContext.value,
        [agentModelContextKey()]: activeAgentModel.value,
      };
      restoreAgentModelForCurrentContext();
      ensureDemoAgentModelPreset();
      activeAgentMode.value = "openrouter";
      gameKind.value = String(data.game_kind || "craftax") === "arc_agi" ? "arc_agi" : "craftax";
      arcGameId.value = String(data.arc_game_id || arcGameId.value || "ls20");
      if (Array.isArray(data.arc_game_options) && data.arc_game_options.length) {
        arcGameOptions.value = data.arc_game_options;
      }
      ensureDemoArcGameSelection();
      exoPlanetEnabled.value = Boolean(data.exo_planet_enabled);
      if (Array.isArray(data.megaprompt_options) && data.megaprompt_options.length) {
        megapromptOptions.value = data.megaprompt_options.map((name) => String(name));
      }
      if (data.megaprompt_config_name) {
        megapromptConfigName.value = String(data.megaprompt_config_name);
      }
      if (isArcGame.value) {
        ensureArcCompanionMode();
      }
      alignMegapromptToWorldMode();
      loadInventoryIcons(true);
      if (data.hf_token_configured !== undefined) {
        hfTokenPreview.value = data.hf_token_configured ? String(data.hf_token_preview || "configured") : "";
        if (hfTokenInput.value.trim()) hfTokenInput.value = "";
      }
      if (data.openrouter_api_key_configured !== undefined) {
        openrouterApiKeyPreview.value = data.openrouter_api_key_configured
          ? String(data.openrouter_api_key_preview || "configured")
          : "";
        if (openrouterApiKeyInput.value.trim()) openrouterApiKeyInput.value = "";
      }
      if (data.campaign_state) {
        campaignState.value = data.campaign_state;
      }
      if (data.world_mode_reset) {
        agentGoal.value = "";
        messages.value = [];
        reward.value = 0;
        done.value = false;
        playerPosition.value = "";
        agentReasoning.value = "";
        agentDirectChatActive.value = false;
        if (worldMapRenderer) {
          worldMapRenderer.destroy();
          worldMapRenderer = null;
        }
      }
      if (data.frame) {
        currentFrame.value = data.frame;
        if (data.frame.agent_observation != null) {
          agentObservation.value = data.frame.agent_observation;
        }
      }
      syncApiKeyGuideAfterConfig(data);
      applyModelCheck(data);
      syncExpertModesToActiveAgent();
      syncExpertModelsToActiveAgent();
      alignMegapromptToWorldMode();
    }

    function showSettingsSaveOutcome(data) {
      settingsSaveOk.value = true;
      if (!featureFlags.value.settings_api_keys) {
        settingsSaveMessage.value = "Settings updated.";
        return;
      }
      const activeMode = String(activeAgentMode.value || "openrouter").toLowerCase();
      const expertModes = data.expert_modes || expertModeFields.value;
      const needsOpenRouter =
        (activeMode === "openrouter" || Object.values(expertModes).some((m) => String(m).toLowerCase() === "openrouter"))
        && !data.openrouter_api_key_configured;
      const needsHf =
        (activeMode === "hub" || Object.values(expertModes).some((m) => String(m).toLowerCase() === "hub"))
        && !data.hf_token_configured;
      if (needsOpenRouter && needsHf) {
        settingsSaveOk.value = false;
        settingsSaveMessage.value =
          "Saved, but HF_TOKEN and OPENROUTER_API_KEY are missing for the selected hub/openrouter modes.";
      } else if (needsOpenRouter) {
        settingsSaveOk.value = false;
        settingsSaveMessage.value =
          "Saved, but OPENROUTER_API_KEY is missing (required for openrouter mode).";
      } else if (needsHf) {
        settingsSaveOk.value = false;
        settingsSaveMessage.value = "Saved, but HF_TOKEN is missing (required for hub mode).";
      }
      applyModelCheck(data);
      if (settingsSaveOk.value && invalidModels.value.length) {
        settingsSaveOk.value = false;
        const list = invalidModels.value.join(", ");
        settingsSaveMessage.value =
          `Saved, but not available on OpenRouter: ${list}. Agent runs will fail until you set a valid model id.`;
      }
      syncApiKeyGuideAfterConfig(data);
    }

    function tokenConfigured(kind, data) {
      const stored = window.PlayWebSession?.readApiSecrets?.() || {};
      if (kind === "openrouter") {
        return Boolean(
          openrouterApiKeyInput.value.trim()
          || stored.openrouter_api_key
          || (data && data.openrouter_api_key_configured)
          || openrouterApiKeyPreview.value,
        );
      }
      return Boolean(
        hfTokenInput.value.trim()
        || stored.hf_token
        || (data && data.hf_token_configured)
        || hfTokenPreview.value,
      );
    }

    function computeMissingApiKeys(data) {
      const expertModes = data?.expert_modes || expertModeFields.value;
      const activeMode = String(data?.active_agent_mode || activeAgentMode.value || "openrouter").toLowerCase();
      const missing = [];
      const needsOpenRouter =
        activeMode === "openrouter"
        || Object.values(expertModes).some((m) => String(m).toLowerCase() === "openrouter");
      const needsHf =
        activeMode === "hub"
        || Object.values(expertModes).some((m) => String(m).toLowerCase() === "hub");
      if (needsOpenRouter && !tokenConfigured("openrouter", data)) missing.push("openrouter");
      if (needsHf && !tokenConfigured("hf", data)) missing.push("hf");
      return missing;
    }

    function apiKeyAlertMessageForKeys(keys) {
      if (!featureFlags.value.settings_api_keys) {
        if (keys.includes("hf") && keys.includes("openrouter")) {
          return "Server API keys required. Set HF_TOKEN and OPENROUTER_API_KEY in the deployment environment.";
        }
        if (keys.includes("openrouter")) {
          return "Server OpenRouter API key required. Set OPENROUTER_API_KEY in the deployment environment.";
        }
        if (keys.includes("hf")) {
          return "Server Hugging Face token required. Set HF_TOKEN in the deployment environment.";
        }
      }
      if (keys.includes("hf") && keys.includes("openrouter")) {
        return "API keys required. Set HF_TOKEN and OPENROUTER_API_KEY in env/.env or enter them in Settings.";
      }
      if (keys.includes("openrouter")) {
        return "OpenRouter API key required. Set OPENROUTER_API_KEY in env/.env or enter it in Settings.";
      }
      if (keys.includes("hf")) {
        return "Hugging Face token required. Set HF_TOKEN in env/.env or enter it in Settings.";
      }
      return "";
    }

    const apiKeyAlertMessage = computed(() => apiKeyAlertMessageForKeys(apiKeyMissingKeys.value));

    function updateApiKeyGuidePosition() {
      const btn = document.getElementById("settingsGearBtn");
      if (!btn) return;
      const rect = btn.getBoundingClientRect();
      apiKeyGuideFloatStyle.value = {
        top: `${Math.round(rect.top)}px`,
        left: `${Math.round(rect.right)}px`,
      };
    }

    function attachApiKeyGuidePositionListeners() {
      detachApiKeyGuidePositionListeners();
      apiKeyGuidePositionListener = () => updateApiKeyGuidePosition();
      window.addEventListener("resize", apiKeyGuidePositionListener);
      window.addEventListener("scroll", apiKeyGuidePositionListener, true);
    }

    function detachApiKeyGuidePositionListeners() {
      if (!apiKeyGuidePositionListener) return;
      window.removeEventListener("resize", apiKeyGuidePositionListener);
      window.removeEventListener("scroll", apiKeyGuidePositionListener, true);
      apiKeyGuidePositionListener = null;
    }

    function showApiKeyGuide(missingKeys) {
      if (!missingKeys.length) return;
      apiKeyMissingKeys.value = missingKeys;
      apiKeyAlertActive.value = true;
      nextTick(() => {
        updateApiKeyGuidePosition();
        attachApiKeyGuidePositionListeners();
      });
    }

    function clearApiKeyGuide() {
      apiKeyMissingKeys.value = [];
      apiKeyAlertActive.value = false;
      detachApiKeyGuidePositionListeners();
    }

    function guardApiKeyForAction() {
      const missing = computeMissingApiKeys();
      if (!missing.length) {
        clearApiKeyGuide();
        return true;
      }
      showApiKeyGuide(missing);
      return false;
    }

    function showModelError(msg) {
      const model = String(msg?.model || activeAgentModel.value || "").trim();
      const details = String(msg?.message || "").trim();
      modelErrorAlert.value = {
        model,
        details,
      };
    }

    function dismissModelError() {
      modelErrorAlert.value = null;
    }

    function openSettingsFromModelError() {
      dismissModelError();
      openSettingsModal();
    }

    function serverMissingApiKeys(data) {
      if (!data) return [];
      const expertModes = data.expert_modes || expertModeFields.value;
      const activeMode = String(data.active_agent_mode || activeAgentMode.value || "openrouter").toLowerCase();
      const needsOpenRouter =
        activeMode === "openrouter"
        || Object.values(expertModes).some((m) => String(m).toLowerCase() === "openrouter");
      const needsHf =
        activeMode === "hub"
        || Object.values(expertModes).some((m) => String(m).toLowerCase() === "hub");
      const missing = [];
      if (needsOpenRouter && !data.openrouter_api_key_configured) missing.push("openrouter");
      if (needsHf && !data.hf_token_configured) missing.push("hf");
      return missing;
    }

    function syncApiKeyGuideAfterConfig(data) {
      if (!setupComplete.value) return;
      // Drive the guide from the server's truth: if the backend reports a key
      // missing for the selected modes, surface the warning even when a stale
      // copy lingers in the browser. Reuse the last server response when called
      // without fresh data so a local-only check can't hide a real gap.
      const cfg = data || lastSessionConfigData;
      const missing = cfg ? serverMissingApiKeys(cfg) : computeMissingApiKeys();
      if (!missing.length) {
        clearApiKeyGuide();
      } else {
        showApiKeyGuide(missing);
      }
    }

    try {
      const storedMessages = localStorage.getItem("oracleChatMessages");
      if (storedMessages) {
        const parsed = JSON.parse(storedMessages);
        if (Array.isArray(parsed)) messages.value = parsed;
      }
    } catch (e) {}

    watch(messages, (newMessages) => {
      try { localStorage.setItem("oracleChatMessages", JSON.stringify(newMessages)); } catch (e) {}
    }, { deep: true });

    let savedGoal = "";
    try { savedGoal = localStorage.getItem("agentGoal") || ""; } catch (e) {}
    const agentGoal = ref(savedGoal);
    watch(agentGoal, (newGoal) => { try { localStorage.setItem("agentGoal", newGoal); } catch (e) {} });

    const agentInstructionReady = computed(() => Boolean(agentGoal.value.trim()));
    const agentInstructionLocked = computed(() => !companionModeEnabled.value && agentWorking.value);

    const INSTRUCTION_EXAMPLES = {
      craftax: [
        "collect wood and place a table",
        "find water nearby",
        "mine stone with a wood pickaxe",
        "craft a furnace from stone",
      ],
      "exo-planet": [
        "collect biomass",
        "deploy two replicators nearby",
        "find a water source",
        "extract basalt with a bone drill",
      ],
      arc_agi: [
        "solve the ARC-AGI-3 game efficiently",
        "explore the rules and complete the level",
      ],
    };
    const INSTRUCTION_EXAMPLE_ROTATE_MS = 3500;
    const instructionExampleIndex = ref(0);
    let instructionExampleTimer = null;

    const instructionExamples = computed(() =>
      isArcGame.value
        ? INSTRUCTION_EXAMPLES.arc_agi
        : exoPlanetEnabled.value
        ? INSTRUCTION_EXAMPLES["exo-planet"]
        : INSTRUCTION_EXAMPLES.craftax,
    );
    const agentInstructionExample = computed(() => {
      const examples = instructionExamples.value;
      if (!examples.length) return "collect wood and place a table";
      return examples[instructionExampleIndex.value % examples.length];
    });
    const agentInstructionPlaceholder = computed(() => {
      if (companionModeEnabled.value || agentInstructionReady.value) {
        return "Enter instruction…";
      }
      return `Enter instruction, e.g. "${agentInstructionExample.value}"`;
    });

    function stopInstructionExampleRotation() {
      if (instructionExampleTimer) {
        clearInterval(instructionExampleTimer);
        instructionExampleTimer = null;
      }
    }

    function startInstructionExampleRotation() {
      stopInstructionExampleRotation();
      if (companionModeEnabled.value || agentInstructionReady.value) return;
      const examples = instructionExamples.value;
      if (examples.length <= 1) return;
      instructionExampleTimer = setInterval(() => {
        if (companionModeEnabled.value || agentInstructionReady.value) {
          stopInstructionExampleRotation();
          return;
        }
        const len = instructionExamples.value.length;
        if (len > 1) {
          instructionExampleIndex.value = (instructionExampleIndex.value + 1) % len;
        }
      }, INSTRUCTION_EXAMPLE_ROTATE_MS);
    }

    watch(exoPlanetEnabled, () => {
      instructionExampleIndex.value = 0;
    });
    watch([companionModeEnabled, agentInstructionReady], ([companion, ready]) => {
      if (!companion && !ready) startInstructionExampleRotation();
      else stopInstructionExampleRotation();
    }, { immediate: true });

    const instructionPlayNoticeVisible = ref(false);
    const instructionPlayNoticeFading = ref(false);
    const instructionPlayNoticeSteps = ref(AGENT_STEPS_PER_PLAY);
    let instructionPlayNoticeTimer = null;
    const INSTRUCTION_PLAY_NOTICE_FADE_MS = 7000;
    const INSTRUCTION_PLAY_NOTICE_HOLD_MS = 4000;

    const companionStartHighlight = ref(false);
    let companionStartHighlightTimer = null;
    const COMPANION_START_HIGHLIGHT_MS = 10000;

    function hideCompanionStartHighlight() {
      companionStartHighlight.value = false;
      if (companionStartHighlightTimer) {
        clearTimeout(companionStartHighlightTimer);
        companionStartHighlightTimer = null;
      }
    }

    function showCompanionStartHighlight() {
      hideCompanionStartHighlight();
      companionStartHighlight.value = true;
      companionStartHighlightTimer = setTimeout(
        hideCompanionStartHighlight,
        COMPANION_START_HIGHLIGHT_MS,
      );
    }

    function markCompanionStartHintPending() {
      try {
        sessionStorage.setItem(COMPANION_START_HINT_KEY, "1");
      } catch (e) {}
    }

    function tryConsumeCompanionStartHint() {
      try {
        if (sessionStorage.getItem(COMPANION_START_HINT_KEY) !== "1") return;
        sessionStorage.removeItem(COMPANION_START_HINT_KEY);
      } catch (e) {
        return;
      }
      if (!companionModeEnabled.value) return;
      showCompanionStartHighlight();
    }

    const companionPlayHighlightVisible = computed(() => {
      if (!companionStartHighlight.value) return false;
      if (companionResearchActive.value) return false;
      if (isArcGame.value && agentMissionActive.value) return false;
      if (!isArcGame.value && agentWorking.value) return false;
      return true;
    });

    function hideInstructionPlayNotice() {
      instructionPlayNoticeVisible.value = false;
      instructionPlayNoticeFading.value = false;
      if (instructionPlayNoticeTimer) {
        clearTimeout(instructionPlayNoticeTimer);
        instructionPlayNoticeTimer = null;
      }
    }

    function showInstructionPlayNotice(steps) {
      hideInstructionPlayNotice();
      instructionPlayNoticeSteps.value = Math.max(1, Math.round(Number(steps) || AGENT_STEPS_PER_PLAY));
      instructionPlayNoticeVisible.value = true;
      instructionPlayNoticeFading.value = false;
      instructionPlayNoticeTimer = setTimeout(() => {
        instructionPlayNoticeFading.value = true;
        instructionPlayNoticeTimer = setTimeout(hideInstructionPlayNotice, INSTRUCTION_PLAY_NOTICE_FADE_MS);
      }, INSTRUCTION_PLAY_NOTICE_HOLD_MS);
    }

    const INSTRUCTION_STEP_TIMER_RADIUS = 20;
    const instructionStepTimerCircumference = 2 * Math.PI * INSTRUCTION_STEP_TIMER_RADIUS;
    const instructionStepBadge = computed(() => {
      const progress = agentTickProgress.value || {};
      const limit = Math.max(
        1,
        Math.round(Number(progress.total) || AGENT_STEPS_PER_PLAY),
      );
      const done = Math.max(0, Math.min(limit, Math.round(Number(progress.done) || 0)));
      const active = Boolean(
        !companionModeEnabled.value
        && agentWorking.value
        && progress.active
        && limit > 0,
      );
      const pct = Math.max(0, Math.min(100, Math.round((done / limit) * 100)));
      return { active, done, limit, pct };
    });
    const instructionStepTimerOffset = computed(() => {
      const pct = instructionStepBadge.value.pct;
      return instructionStepTimerCircumference * (1 - pct / 100);
    });

    const campaignProgressPct = computed(() => {
      const state = campaignState.value;
      if (!state || !state.total_count) return 0;
      return Math.round((100 * (state.completed_count || 0)) / state.total_count);
    });

    const campaignProgressText = computed(() => {
      const state = campaignState.value;
      if (!state) return "Campaign mode is off";
      const completed = state.completed_count || 0;
      const total = state.total_count || 0;
      if (!state.enabled) return `Campaign mode is off (${completed}/${total})`;
      if (state.is_finished) return `Campaign complete (${completed}/${total})`;
      return `Current task: ${state.current_task_title || "—"} (${completed}/${total})`;
    });

    const campaignTasks = computed(() => {
      const state = campaignState.value;
      const stateTasks = Array.isArray(state?.tasks) ? state.tasks : [];
      if (isArcGame.value && stateTasks.length) return stateTasks;
      const bench = companionBenchStatus.value;
      const benchTasks = bench?.campaign_tasks;
      const expectedWorld = isArcGame.value
        ? "arc_agi"
        : (exoPlanetEnabled.value ? "exo-planet" : "craftax");
      const benchWorld = String(bench?.world_mode || "").trim().toLowerCase();
      if (
        Array.isArray(benchTasks)
        && benchTasks.length
        && benchWorld === expectedWorld
      ) {
        return benchTasks;
      }
      return stateTasks;
    });

    const campaignPhase2CompletedKeys = computed(() => {
      const keys = campaignState.value?.phase2?.completed_keys;
      return Array.isArray(keys) ? keys : [];
    });

    const campaignPhase2SelectedKey = computed(() => {
      return String(campaignState.value?.phase2?.selected_task_key || "");
    });

    function clampAgentStepsPerClick(steps) {
      const maxSteps = Math.min(Math.max(1, maxAgentStepsPerClick.value || 1), 500);
      const n = Number(steps);
      if (!Number.isFinite(n)) return 1;
      return Math.max(1, Math.min(Math.round(n), maxSteps));
    }

    function reset() {
      clearAchievementToasts();
      if (isArcGame.value) {
        resetArcScoreState();
      }
      resetGame();
    }

    function companionStripStepBudget() {
      return isArcGame.value
        ? companionBenchMaxTicksPerTask.value
        : agentStepsPerClick.value;
    }

    function agentStep() {
      if (!guardApiKeyForAction()) return;
      const campaignGoal = campaignState.value?.enabled
        ? String(campaignState.value.current_task_goal || "")
        : "";
      const instruction = agentGoal.value.trim();
      if (!companionModeEnabled.value && !campaignGoal && !instruction) return;
      const goal = campaignGoal || instruction;
      const steps = clampAgentStepsPerClick(companionStripStepBudget());
      if (isArcGame.value) {
        companionBenchMaxTicksPerTask.value = steps;
      } else {
        agentStepsPerClick.value = steps;
      }
      if (!companionModeEnabled.value) {
        showInstructionPlayNotice(steps);
      } else {
        hideCompanionStartHighlight();
      }
      agentTick(goal, steps, true, buildAgentRuntimeOverrides());
    }

    const agentMissionActive = computed(() => isAgentMissionActive());

    function agentPlayToggle() {
      if (agentMissionActive.value) {
        agentStop();
      } else {
        agentStep();
      }
    }

    function onAgentInstructionEnter() {
      if (companionModeEnabled.value || agentInstructionLocked.value || !agentInstructionReady.value || agentWorking.value) {
        return;
      }
      agentStep();
    }

    async function toggleCompanionMode() {
      if (isArcGame.value) {
        ensureArcCompanionMode();
        return;
      }
      const next = !companionModeEnabled.value;
      companionModeEnabled.value = next;
      if (!next) {
        if (companionResearchActive.value || agentWorking.value) {
          agentStop();
        }
        if (companionBenchStatus.value?.running) {
          await stopCompanionBench();
        }
        setCampaignEnabled(false);
        companionResearchActive.value = false;
        persistSharedSettingsToStorage();
        return;
      }
      await loadCompanionBenchStatus();
      if (!companionBenchKnowledgeSourceDirty.value) {
        companionBenchKnowledgeSource.value = "base";
      }
      if (
        companionBenchKnowledgeSource.value === "own"
        && !companionBenchStatus.value?.has_own_knowledge
      ) {
        companionBenchKnowledgeSource.value = "base";
      }
      setCampaignEnabled(true);
      persistSharedSettingsToStorage();
    }

    function startCompanionResearch() {
      if (!guardApiKeyForAction()) return;
      if (!send) return;
      hideCompanionStartHighlight();
      companionBenchError.value = "";
      agentStopping.value = false;
      agentWorking.value = true;
      agentAwaitingResponse.value = true;
      beginAgentMissionClock();
      const runtime = buildAgentRuntimeOverrides();
      send({
        type: "companion_research_start",
        knowledge_source: companionBenchKnowledgeSource.value,
        max_ticks_per_task: Math.max(
          1,
          Math.round(Number(companionBenchMaxTicksPerTask.value) || DEFAULT_COMPANION_MAX_TICKS_PER_TASK),
        ),
        active_agent_model: runtime.active_agent_model,
        active_agent_mode: runtime.active_agent_mode,
        megaprompt_config_name: runtime.megaprompt_config_name,
        exo_planet_enabled: runtime.exo_planet_enabled,
        player_name: playerNickname.value.trim(),
        player_avatar_id: playerAvatarId.value,
      });
    }

    function stopCompanionResearch() {
      agentStop();
    }

    const companionKnowledgeOptions = computed(() => {
      const status = companionBenchStatus.value || {};
      const baseFile = String(status.base_knowledge_file || "knowledge_data.json");
      const modelFile = String(status.model_knowledge_file || "knowlage_data_<model>.json");
      const hasOwn = Boolean(status.has_own_knowledge);
      return {
        baseLabel: baseFile,
        modelLabel: hasOwn ? modelFile : `${modelFile} (not yet)`,
        modelExists: hasOwn,
      };
    });

    const companionTestKnowledgeOptions = companionKnowledgeOptions;

    watch(activeAgentModel, () => {
      if (companionModeEnabled.value) {
        loadCompanionBenchStatus();
      }
    });

    watch(exoPlanetEnabled, () => {
      loadInventoryIcons();
      if (companionModeEnabled.value) {
        loadCompanionBenchStatus();
      }
    });

    // The companion strip grows the top overlay; re-measure so the floating
    // control buttons (play/reset/record) and the map drop below it.
    watch(companionModeEnabled, () => {
      if (!companionModeEnabled.value) {
        companionStripSettingsOpen.value = false;
        hideCompanionTip();
      }
      nextTick(() => {
        syncGameTopbarHeight();
        worldMapRenderer?.resize();
      });
    });

    watch(status, (next) => {
      if (next === "connected") {
        syncArcCampaignOnConnect();
        syncCompanionCampaignOnConnect();
      }
    });

    watch(isArcGame, (arc) => {
      if (arc) {
        ensureArcCompanionMode();
        nextTick(() => ensureArcFrameHoverObserver());
      } else {
        destroyArcFrameHoverObserver();
      }
    });

    watch(gameKind, () => {
      restoreAgentModelForCurrentContext();
      ensureDemoAgentModelPreset();
    });

    watch(activeAgentModel, (model) => {
      const normalized = String(model || "").trim();
      if (!normalized) return;
      const key = agentModelContextKey();
      agentModelByContext.value = { ...agentModelByContext.value, [key]: normalized };
    });

    function scheduleCompanionBenchPoll() {
      if (companionBenchPollTimerId) {
        clearTimeout(companionBenchPollTimerId);
        companionBenchPollTimerId = null;
      }
      const delayMs = companionBenchStatus.value?.running ? 700 : 3000;
      companionBenchPollTimerId = setTimeout(() => {
        loadCompanionBenchStatus();
      }, delayMs);
    }

    function stopCompanionBenchPoll() {
      if (companionBenchPollTimerId) {
        clearTimeout(companionBenchPollTimerId);
        companionBenchPollTimerId = null;
      }
    }

    async function loadCompanionBenchStatus() {
      companionBenchError.value = "";
      try {
        const resp = await apiFetch("/companion_bench/status");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        companionBenchStatus.value = data.status || null;
        if (!data?.status?.running && data?.status?.completed) {
          loadStatistics();
        }
        if (!companionBenchMaxTicksDirty.value && data?.status?.max_ticks_per_task != null) {
          const n = Number(data.status.max_ticks_per_task);
          if (Number.isFinite(n) && n > 0) {
            companionBenchMaxTicksPerTask.value = Math.max(1, Math.round(n));
          }
        }
        if (!companionBenchParallelAgentsDirty.value && data?.status?.parallel_agents != null) {
          const n = Number(data.status.parallel_agents);
          if (Number.isFinite(n) && n > 0) {
            companionBenchParallelAgents.value = Math.max(1, Math.min(3, Math.round(n)));
          }
        }
        if (!companionBenchCyclesDirty.value && data?.status?.cycles != null) {
          const n = Number(data.status.cycles);
          if (Number.isFinite(n) && n > 0) {
            companionBenchCycles.value = Math.max(1, Math.round(n));
          }
        }
        if (!companionBenchTestTaskKey.value && data?.status?.test_task_key) {
          companionBenchTestTaskKey.value = String(data.status.test_task_key);
        }
        if (
          companionBenchKnowledgeSource.value === "own"
          && data?.status
          && !data.status.has_own_knowledge
        ) {
          companionBenchKnowledgeSource.value = "base";
        }
        if (
          companionBenchTestKnowledgeSource.value === "own"
          && data?.status
          && !data.status.has_own_knowledge
        ) {
          companionBenchTestKnowledgeSource.value = "base";
        }
      } catch (e) {
        companionBenchError.value = "Failed to load companion bench status";
      } finally {
        scheduleCompanionBenchPoll();
      }
    }

    async function startCompanionBenchRequest(phase, extra = {}) {
      companionBenchStarting.value = true;
      companionBenchError.value = "";
      try {
        alignMegapromptToWorldMode();
        companionBenchMaxTicksDirty.value = false;
        companionBenchParallelAgentsDirty.value = false;
        companionBenchCyclesDirty.value = false;
        const resp = await apiFetch("/companion_bench/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            phase,
            model: activeAgentModel.value,
            mode: activeAgentMode.value,
            megaprompt_config_name: megapromptConfigName.value,
            parallel_agents: Math.max(
              1,
              Math.min(3, Math.round(Number(companionBenchParallelAgents.value) || 3)),
            ),
            max_ticks_per_task: Math.max(
              1,
              Math.round(Number(companionBenchMaxTicksPerTask.value) || DEFAULT_COMPANION_MAX_TICKS_PER_TASK),
            ),
            cycles: Math.max(1, Math.round(Number(companionBenchCycles.value) || 1)),
            task_key: companionBenchTestTaskKey.value,
            knowledge_source:
              phase === "test"
                ? companionBenchTestKnowledgeSource.value
                : companionBenchKnowledgeSource.value,
            ...extra,
          }),
        });
        const data = await resp.json();
        if (!resp.ok || data.ok === false) throw new Error(data.error || `HTTP ${resp.status}`);
        companionBenchStatus.value = data.status || companionBenchStatus.value;
        scheduleCompanionBenchPoll();
        loadStatistics();
      } catch (e) {
        companionBenchError.value = `Failed to start ${phase}${e?.message ? `: ${e.message}` : "."}`;
      } finally {
        companionBenchStarting.value = false;
      }
    }

    async function startCompanionTest() {
      if (!guardApiKeyForAction()) return;
      if (!companionBenchTestTaskKey.value) {
        companionBenchError.value = "Select a level for testing.";
        return;
      }
      await startCompanionBenchRequest("test");
    }

    async function stopCompanionBench() {
      companionBenchStopping.value = true;
      companionBenchError.value = "";
      try {
        const resp = await apiFetch("/companion_bench/stop", {
          method: "POST",
        });
        const data = await resp.json();
        if (!resp.ok || data.ok === false) throw new Error(data.error || `HTTP ${resp.status}`);
        companionBenchStatus.value = data.status || companionBenchStatus.value;
        scheduleCompanionBenchPoll();
        loadStatistics();
      } catch (e) {
        companionBenchError.value = `Failed to stop benchmark${e?.message ? `: ${e.message}` : "."}`;
      } finally {
        companionBenchStopping.value = false;
      }
    }

    function companionResearchLevelClass(index) {
      const completed = Number(campaignState.value?.completed_count || 0);
      if (index < completed) return "completed";
      if (index === completed && companionResearchActive.value) return "current";
      if (campaignState.value?.is_finished && index === completed) return "completed";
      return "pending";
    }

    // Maps each campaign achievement to an inventory icon key (rendered from the
    // texture bundle) or, when no item icon exists, a Bootstrap icon class.
    const COMPANION_TASK_ICON_MAP = {
      collect_wood: { inv: "wood" },
      place_table: { inv: "crafting_table", bi: "bi-grid-3x3-gap-fill" },
      make_wood_pickaxe: { inv: "wood_pickaxe" },
      collect_stone: { inv: "stone" },
      make_stone_pickaxe: { inv: "stone_pickaxe" },
      collect_coal: { inv: "coal" },
      collect_iron: { inv: "iron" },
      make_furnace: { inv: "furnace", bi: "bi-fire" },
      make_iron_pickaxe: { inv: "iron_pickaxe" },
      collect_diamond: { inv: "diamond" },
    };
    // Short "how to get this" hints shown on hover over a strip node.
    const COMPANION_TASK_HINTS_CRAFTAX = {
      collect_wood: "Face a tree and DO to gather — no tool needed (1 wood each).",
      place_table: "On grass or path, place a crafting table (costs 2 wood).",
      make_wood_pickaxe: "Adjacent to a table, facing it: craft a wood pickaxe (1 wood).",
      collect_stone: "DO on stone while holding a wood pickaxe (mines 1 stone).",
      make_stone_pickaxe: "At a table: craft a stone pickaxe (1 wood + 1 stone).",
      collect_coal: "DO on coal while holding a stone pickaxe (mines 1 coal).",
      collect_iron: "DO on iron while holding a stone pickaxe (mines 1 iron).",
      make_furnace: "On grass or path, place a furnace (costs 1 stone + 1 coal).",
      make_iron_pickaxe: "At a table with furnace adjacent: craft an iron pickaxe (1 wood + 1 iron).",
      collect_diamond: "DO on diamond while holding an iron pickaxe (mines 1 diamond).",
    };
    const COMPANION_TASK_HINTS_ARC = {
      level_1: "Solve level 1 — ask the human helper if the rules are unclear.",
      level_2: "Progress to level 2 by completing the current puzzle.",
      level_3: "Each level teaches a new rule — use human hints sparingly.",
      level_4: "Watch levels_completed in the observation to track progress.",
      level_5: "Try ACTION6 with coordinates when the puzzle needs a click.",
      level_6: "Combine actions from earlier levels.",
      level_7: "Final level — finish the ARC game run.",
    };
    const COMPANION_TASK_HINTS_EXO = {
      collect_wood: "Face a Xeno-Root Mass and EXTRACT — no drill needed (1 Biomass each).",
      place_table: "On Regolith Turf or Survey Trail, deploy a Replicator (costs 2 Biomass).",
      make_wood_pickaxe: "Adjacent to Replicator, facing it: MAKE_BONE_DRILL (1 Biomass).",
      collect_stone: "EXTRACT Basalt Crust with a Bone Drill (1 Basalt Shard).",
      make_stone_pickaxe: "At Replicator: MAKE_ROCK_DRILL (1 Biomass + 1 Basalt Shard).",
      collect_coal: "EXTRACT Energy Ore with a Rock Drill (1 Energy Ore).",
      collect_iron: "EXTRACT Titanite Ore with a Rock Drill (1 Titanite Ore).",
      make_furnace: "On turf or trail, deploy a Thermal Oven (costs 1 Basalt Shard + 1 Energy Ore).",
      make_iron_pickaxe: "At Replicator with Thermal Oven adjacent: MAKE_TITAN_DRILL (1 Basalt Shard + 1 Titanite Ore).",
      collect_diamond: "EXTRACT Core Ore with a Titan Drill (1 Core Ore).",
    };

    const companionStripFailedIcons = ref({});
    function onCompanionIconError(key) {
      if (!key || companionStripFailedIcons.value[key]) return;
      companionStripFailedIcons.value = {
        ...companionStripFailedIcons.value,
        [key]: true,
      };
    }

    const companionStripNodes = computed(() => {
      const tasks = campaignTasks.value;
      const icons = inventoryIcons.value || {};
      const failed = companionStripFailedIcons.value;
      const completed = Number(campaignState.value?.completed_count || 0);
      const finished = Boolean(campaignState.value?.is_finished);
      const running = companionResearchActive.value;
      const hints = isArcGame.value
        ? COMPANION_TASK_HINTS_ARC
        : exoPlanetEnabled.value
        ? COMPANION_TASK_HINTS_EXO
        : COMPANION_TASK_HINTS_CRAFTAX;
      return tasks.map((task, index) => {
        const key = String(task.key || "");
        const levelNum = index + 1;
        const iconDef = isArcGame.value
          ? {}
          : (COMPANION_TASK_ICON_MAP[key] || {});
        const invKey = iconDef.inv || "";
        let stateClass;
        if (index < completed) stateClass = "is-done";
        else if (!finished && index === completed) stateClass = "is-current";
        else stateClass = "is-pending";
        return {
          key: key || `task-${index}`,
          index,
          title: isArcGame.value
            ? `Level-${levelNum}`
            : String(task.title || `Level ${levelNum}`),
          hint: hints[key] || "",
          levelNumber: isArcGame.value ? levelNum : null,
          iconKey: invKey,
          icon: invKey && !failed[invKey] ? icons[invKey] || "" : "",
          biClass: iconDef.bi || "bi-question-lg",
          stateClass,
          isCurrentRunning: running && !finished && index === completed,
        };
      });
    });

    const companionStripFillPct = computed(() => {
      const total = companionStripNodes.value.length;
      if (total <= 1) return campaignState.value?.is_finished ? 100 : 0;
      if (campaignState.value?.is_finished) return 100;
      const completed = Number(campaignState.value?.completed_count || 0);
      const frac = Math.min(completed, total - 1) / (total - 1);
      return Math.round(frac * 100);
    });

    // Live "steps done / limit" gauge for companion research (whole-run budget).
    const COMPANION_GAUGE_RADIUS = 52;
    const companionStepGaugeCircumference = 2 * Math.PI * COMPANION_GAUGE_RADIUS;
    const companionStepBadge = computed(() => {
      const snap = companionResearchSnapshot.value || null;
      const progress = agentTickProgress.value || {};
      const arcManualRun = Boolean(
        isArcGame.value
        && !companionResearchActive.value
        && progress.active
        && Math.max(1, Math.round(Number(progress.total) || 0)) > 0,
      );
      const researchRun = Boolean(companionResearchActive.value && snap && snap.active);
      const active = researchRun || arcManualRun;
      let limit;
      let done;
      let taskIndex;
      let taskTotal;
      let taskTitle;
      if (arcManualRun) {
        limit = Math.max(
          1,
          Math.round(Number(progress.total) || Number(companionBenchMaxTicksPerTask.value) || 1),
        );
        done = Math.max(0, Math.min(limit, Math.round(Number(progress.done) || 0)));
        const completed = Number(campaignState.value?.completed_count || 0);
        taskIndex = completed + 1;
        taskTotal = Math.round(Number(campaignState.value?.total_count || campaignTasks.value.length || 0));
        taskTitle = String(campaignState.value?.current_task_title || `Level-${taskIndex}`);
      } else {
        limit = Math.max(
          1,
          Math.round(
            Number(snap?.max_ticks_per_task)
            || Number(companionBenchMaxTicksPerTask.value)
            || 1,
          ),
        );
        done = Math.max(0, Math.min(limit, Math.round(Number(snap?.task_ticks) || 0)));
        taskIndex = Math.round(Number(snap?.task_index) || 0);
        taskTotal = Math.round(Number(snap?.task_total) || 0);
        taskTitle = snap?.task_title || "";
      }
      const pct = Math.max(0, Math.min(100, Math.round((done / limit) * 100)));
      return {
        active,
        done,
        limit,
        pct,
        taskIndex,
        taskTotal,
        taskTitle,
      };
    });
    const companionStripAgentRunning = computed(() =>
      Boolean(isArcGame.value && !companionResearchActive.value && agentTickProgress.value.active),
    );
    const companionStepsLocked = computed(() => {
      if (!companionModeEnabled.value) return false;
      if (agentMissionActive.value || companionResearchActive.value) return true;
      const snap = companionResearchSnapshot.value;
      if (Number(snap?.task_ticks || 0) > 0) return true;
      const progress = agentTickProgress.value || {};
      if (Number(progress.done || 0) > 0) return true;
      if (Number(campaignState.value?.completed_count || 0) > 0) return true;
      return false;
    });
    const companionStepGaugeOffset = computed(() => {
      const pct = companionStepBadge.value.pct;
      return companionStepGaugeCircumference * (1 - pct / 100);
    });

    // The popover is teleported to <body> (a full-viewport WebGL canvas otherwise
    // composites over it when it overflows the top overlay's box), so we position
    // it from the gear button's screen rect.
    const companionStripPopoverStyle = ref({});

    function positionCompanionStripPopover() {
      const gear = document.querySelector(".companion-strip-icon-btn--gear");
      if (!gear) return;
      const rect = gear.getBoundingClientRect();
      const width = 272;
      const left = Math.min(
        Math.max(rect.left, 8),
        Math.max(8, window.innerWidth - width - 8),
      );
      companionStripPopoverStyle.value = {
        left: `${left}px`,
        top: `${rect.bottom + 8}px`,
      };
    }

    function toggleCompanionStripSettings() {
      const next = !companionStripSettingsOpen.value;
      if (next) positionCompanionStripPopover();
      companionStripSettingsOpen.value = next;
    }

    function closeCompanionStripSettings() {
      companionStripSettingsOpen.value = false;
    }

    // Hover hint is teleported to <body> and positioned from the icon's screen
    // rect, so it can never be trapped by the strip's stacking context.
    const companionTip = ref({
      visible: false,
      title: "",
      hint: "",
      index: 0,
      style: {},
    });

    function showCompanionTip(event, node) {
      const el = event.currentTarget;
      if (!el || !el.getBoundingClientRect) return;
      const rect = el.getBoundingClientRect();
      const half = 150;
      const centerX = rect.left + rect.width / 2;
      const clampedX = Math.min(
        Math.max(centerX, half + 8),
        window.innerWidth - half - 8,
      );
      companionTip.value = {
        visible: true,
        title: node.title,
        hint: node.hint,
        index: node.index + 1,
        style: { left: `${clampedX}px`, top: `${rect.bottom + 10}px` },
      };
    }

    function hideCompanionTip() {
      if (!companionTip.value.visible) return;
      companionTip.value = { ...companionTip.value, visible: false };
    }

    function phase2LevelClass(taskKey) {
      const completedKeys = campaignPhase2CompletedKeys.value;
      const selectedKey = campaignPhase2SelectedKey.value;
      if (completedKeys.includes(taskKey)) return "phase2-completed";
      if (selectedKey === taskKey) return "phase2-current";
      return "phase2-pending";
    }

    function startPhase2Level(taskKey) {
      if (!companionModeEnabled.value || !taskKey) return;
      clearAchievementToasts();
      startCampaignPhase2(taskKey);
    }

    function toggleSaveTrajectory() {
      if (send) {
        send({
          type: "trajectory_save_toggle",
          enabled: saveTrajectory.value,
        });
      }
    }
    function toggleStatsPanel() {
      statsPanelCollapsed.value = !statsPanelCollapsed.value;
      savePanelLayout();
      nextTick(() => {
        worldMapRenderer?.resize();
      });
    }

    function toggleOperatorPanel() {
      operatorPanelCollapsed.value = !operatorPanelCollapsed.value;
      if (!operatorPanelCollapsed.value) operatorPanelMessageAlert.value = false;
      savePanelLayout();
      nextTick(() => {
        worldMapRenderer?.resize();
        worldMapRenderer?.recenter?.();
      });
    }

    function openOperatorPanel() {
      operatorPanelMessageAlert.value = false;
      if (!operatorPanelCollapsed.value) return;
      operatorPanelCollapsed.value = false;
      savePanelLayout();
      nextTick(() => {
        worldMapRenderer?.resize();
        worldMapRenderer?.recenter?.();
      });
    }

    function startPanelResize(event, getWidth, setWidth, bounds) {
      const startX = event.clientX;
      const startWidth = getWidth();
      document.body.classList.add("panel-resize-active");

      function onMove(moveEvent) {
        const next = clampPanelWidth(startWidth + (startX - moveEvent.clientX), bounds);
        setWidth(next);
      }

      function onUp() {
        document.body.classList.remove("panel-resize-active");
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        savePanelLayout();
        nextTick(() => {
          worldMapRenderer?.resize();
        });
      }

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    }

    function startStatsPanelResize(event) {
      if (statsPanelCollapsed.value) return;
      startPanelResize(
        event,
        () => statsPanelWidth.value,
        (value) => { statsPanelWidth.value = value; },
        STATS_PANEL_WIDTH,
      );
    }

    function startOperatorPanelResize(event) {
      startPanelResize(
        event,
        () => operatorPanelWidth.value,
        (value) => { operatorPanelWidth.value = value; },
        OPERATOR_PANEL_WIDTH,
      );
    }

    function closeReasoningPanel() {
      if (reasoningPanelCollapsed.value) return;
      reasoningPanelCollapsed.value = true;
      savePanelLayout();
      nextTick(() => {
        worldMapRenderer?.resize();
      });
    }

    function toggleReasoningPanel() {
      reasoningPanelCollapsed.value = !reasoningPanelCollapsed.value;
      savePanelLayout();
      nextTick(() => {
        worldMapRenderer?.resize();
      });
    }

    const wsStatus = computed(() => status.value);
    const agentReasoningThinking = computed(() => {
      if (!agentWorking.value) return false;
      if (agentAwaitingResponse.value) return true;
      const text = String(agentReasoning.value || "").trim();
      return !text || text === "...";
    });
    const agentWaitingForOperator = computed(() => {
      // Only a human operator actually blocks the agent; with the AI operator
      // the server answers by itself, so the agent is still "thinking".
      if (interactionMode.value !== "human") return false;
      if (String(pendingAgentQuestion.value || "").trim()) return true;
      return messages.value.some((m) => m.pending && m.kind === "agent_message");
    });
    const agentStatusBannerVisible = computed(() => {
      if (agentStopping.value) return true;
      if (agentWaitingForOperator.value) return false;
      if (!isAgentMissionActive()) return false;
      const lastStep = lastAgentStepAt.value;
      const fresh = lastStep > 0
        && agentThinkingNow.value - lastStep < AGENT_THINKING_STALL_MS;
      return !fresh;
    });
    const agentReasoningDisplay = computed(() => {
      const text = String(agentReasoning.value || "").trim();
      return text || "...";
    });
    const reasoningToggleLive = computed(() => {
      if (agentReasoningThinking.value) return true;
      const text = String(agentReasoning.value || "").trim();
      return Boolean(text && text !== "...");
    });
    const reasoningPanelSize = computed(() => {
      if (agentReasoningThinking.value) return "compact";
      const text = String(agentReasoning.value || "").trim();
      if (!text || text === "...") return "compact";
      const longestLine = text
        .split("\n")
        .reduce((max, line) => Math.max(max, line.length), 0);
      if (text.length > 900 || longestLine > 80) return "xl";
      if (text.length > 380 || longestLine > 52) return "wide";
      return "compact";
    });
    watch(agentReasoning, (nextTextRaw, prevTextRaw) => {
      const nextText = String(nextTextRaw || "").trim();
      const prevText = String(prevTextRaw || "").trim();
      if (!nextText || nextText === "..." || nextText === prevText) return;
      if (reasoningPanelCollapsed.value) {
        reasoningNeonAlert.value = true;
      }
    });
    watch(reasoningPanelCollapsed, (collapsed) => {
      if (!collapsed) {
        reasoningNeonAlert.value = false;
      }
    });
    watch(agentReasoningThinking, (isThinking, wasThinking) => {
      if (wasThinking && !isThinking && reasoningPanelCollapsed.value) {
        const text = String(agentReasoning.value || "").trim();
        if (text && text !== "...") {
          reasoningNeonAlert.value = true;
        }
      }
    });
    watch(agentStatusBannerVisible, () => {
      nextTick(() => syncGameTopbarHeight());
    });
    watch(statsPanelCollapsed, (collapsed) => {
      if (!collapsed) {
        statsPanelNeonAlert.value = false;
      } else if (agentHasStepped.value) {
        statsPanelNeonAlert.value = true;
      }
      savePanelLayout();
    });
    const actions = [
      { a: 0, label: "No-op" },
      { a: 1, label: "Left" },
      { a: 2, label: "Right" },
      { a: 3, label: "Up" },
      { a: 4, label: "Down" },
    ];

    function syncAgentReasoningHeight() {
      /* reasoning panel uses fixed layout in world-map mode */
    }

    function syncGameTopbarHeight() {
      const panel = document.querySelector(".game-panel--overlay");
      if (!panel) {
        document.documentElement.style.removeProperty("--game-topbar-height");
        return;
      }
      const bottom = Math.ceil(panel.getBoundingClientRect().bottom + 6);
      document.documentElement.style.setProperty("--game-topbar-height", `${bottom}px`);
    }

    let lastRequestedMapEpoch = null;
    function requestFullWorldMap(epoch) {
      if (epoch != null && lastRequestedMapEpoch === epoch) return;
      lastRequestedMapEpoch = epoch != null ? epoch : null;
      send?.({ type: "request_full_map" });
    }

    function ensureWorldMapRenderer() {
      if (isArcGame.value) return null;
      if (worldMapRenderer || !worldCanvasEl.value || !window.PlayWorldMap?.createWorldMapRenderer) {
        return worldMapRenderer;
      }
      worldMapRenderer = window.PlayWorldMap.createWorldMapRenderer(worldCanvasEl.value, {
        onFieldClick: () => {
          closeReasoningPanel();
        },
        onTileSelect: (tile) => {
          showTileInfo(tile);
        },
        onHoverChange: (tile) => {
          handleTileHoverChange(tile);
        },
        onFollowChange: (following) => {
          mapFollowingAgent.value = following;
        },
        onMapEpochMismatch: (epoch) => {
          requestFullWorldMap(epoch);
        },
      });
      mapFollowingAgent.value = worldMapRenderer.isFollowing?.() ?? true;
      if (typeof ResizeObserver !== "undefined") {
        canvasResizeObserver = new ResizeObserver(() => {
          worldMapRenderer?.resize();
        });
        canvasResizeObserver.observe(worldCanvasEl.value);
      }
      if (currentFrame.value) applyWorldFrame(currentFrame.value);
      worldMapRenderer?.resize();
      return worldMapRenderer;
    }

    function waitForFrameChange(prevFrame, { timeoutMs = 6000 } = {}) {
      return new Promise((resolve) => {
        const started = Date.now();
        const check = () => {
          if (currentFrame.value !== prevFrame || Date.now() - started >= timeoutMs) {
            resolve();
            return;
          }
          setTimeout(check, 120);
        };
        setTimeout(check, 120);
      });
    }

    async function syncWorldMapFromFrame(frame = currentFrame.value) {
      if (!frame || frame.arc || isArcGame.value) return;
      ensureWorldMapRenderer();
      if (!worldMapRenderer) return;
      await worldMapRenderer.applyFrame(frame);
      if (!worldMapRenderer.hasBaseMap?.()) {
        const prevFrame = currentFrame.value;
        resetGame();
        await waitForFrameChange(prevFrame);
        if (currentFrame.value) {
          await worldMapRenderer.applyFrame(currentFrame.value);
        }
      }
      worldMapRenderer.recenter?.();
      worldMapRenderer.resize?.();
    }

    function applyWorldFrame(frame) {
      if (!frame) return;
      if (frame.arc || isArcGame.value) return;
      void syncWorldMapFromFrame(frame);
    }

    const worldHud = computed(() => {
      if (isArcGame.value) return null;
      const world = currentFrame.value?.world;
      if (!world) return null;
      return {
        stats: world.stats || {},
        inventory_items: Array.isArray(world.inventory_items) ? world.inventory_items : [],
      };
    });

    const inventorySlots = computed(() => {
      const world = currentFrame.value?.world;
      const counts = (world && world.inventory) || {};
      const icons = inventoryIcons.value;
      const order = inventorySlotOrder.value.length
        ? inventorySlotOrder.value
        : INVENTORY_SLOT_ORDER;
      return order.map((key) => {
        const count = Number(counts[key] || 0);
        return {
          key,
          count,
          icon: icons[key] || "",
          hasItem: count > 0,
        };
      });
    });

    const agentObservationImageSrc = computed(() => {
      const frame = currentFrame.value;
      if (!frame?.png_b64) return "";
      return `data:image/png;base64,${frame.png_b64}`;
    });

    const agentObservationDisplay = computed(() => {
      if (!isArcImagePrompt.value) return agentObservation.value || "";
      if (isArcGridImagePrompt.value) return agentObservation.value || "";
      const arc = currentFrame.value?.arc || {};
      const frame = currentFrame.value || {};
      const actionsList = Array.isArray(arc.available_actions) ? arc.available_actions : [];
      return [
        `Game: ${arc.game_id || arcGameId.value}`,
        `State: ${arc.state || "unknown"}`,
        `Levels completed: ${arc.levels_completed || 0}`,
        `Available actions: ${actionsList.length ? actionsList.join(", ") : "none"}`,
        `Frame image: ${frame.w || 0}x${frame.h || 0} PNG attached as image input.`,
        "Coordinate convention: x=column 0..63 left-to-right, y=row 0..63 top-to-bottom.",
      ].join("\n");
    });

    const arcFrameImageSrc = agentObservationImageSrc;

    function formatInventoryKey(key) {
      return String(key || "").replace(/_/g, " ");
    }

    function recenterWorldMap() {
      worldMapRenderer?.recenter();
    }

    function zoomWorldIn() {
      worldMapRenderer?.zoomIn();
    }

    function zoomWorldOut() {
      worldMapRenderer?.zoomOut();
    }

    function triggerOperatorPanelGlow() {
      operatorPanelMessageGlow.value = false;
      if (operatorPanelGlowTimer) clearTimeout(operatorPanelGlowTimer);
      requestAnimationFrame(() => {
        operatorPanelMessageGlow.value = true;
        operatorPanelGlowTimer = setTimeout(() => {
          operatorPanelMessageGlow.value = false;
        }, 1800);
      });
    }
    watch(() => messages.value.length, (next, prev) => {
      if (next <= prev) return;
      scheduleAgentPromptPreviewRefresh();
      const last = messages.value[messages.value.length - 1];
      if (last && last.kind === "human_agent") return;
      if (last && last.kind === "oracle" && last.pending) return;
      if (operatorPanelCollapsed.value) {
        operatorPanelMessageAlert.value = true;
      } else {
        triggerOperatorPanelGlow();
      }
    });

    watch([operatorPanelWidth, statsPanelWidth, statsPanelCollapsed, operatorPanelCollapsed], () => {
      document.documentElement.style.setProperty(
        "--operator-panel-width",
        operatorPanelCollapsed.value ? "0px" : `${operatorPanelWidth.value}px`,
      );
      document.documentElement.style.setProperty(
        "--stats-panel-width",
        statsPanelCollapsed.value ? "2.5rem" : `${statsPanelWidth.value}px`,
      );
      nextTick(() => {
        worldMapRenderer?.resize();
        worldMapRenderer?.recenter?.();
      });
    }, { immediate: true });

    watch(setupComplete, (ready) => {
      if (!ready) return;
      nextTick(() => {
        void syncWorldMapFromFrame();
        syncGameTopbarHeight();
      });
    });

    watch(companionModeEnabled, () => {
      nextTick(() => syncGameTopbarHeight());
    });

    watch(currentFrame, (frame) => {
      if (frame) applyWorldFrame(frame);
      if (frame) scheduleAgentPromptPreviewRefresh();
    }, { immediate: true });

    let lastArcLeaderboardLevelCheckpoint = 0;
    watch(
      () => Number(currentFrame.value?.arc?.levels_completed || 0),
      async (levelsCompleted) => {
        if (!isArcGame.value || levelsCompleted <= 0) return;
        if (levelsCompleted === lastArcLeaderboardLevelCheckpoint) return;
        lastArcLeaderboardLevelCheckpoint = levelsCompleted;
        await fetchArcHumanScore({ show: true });
        await loadHumanLeaderboard();
      },
    );

    watch(done, async (isDone) => {
      if (!isArcGame.value) return;
      if (!isDone) {
        arcScorePromptShown.value = false;
        return;
      }
      if (arcScorePromptShown.value) return;
      arcScorePromptShown.value = true;
      await fetchArcHumanScore({ show: true });
    });

    function logIsNearBottom(el, thresholdPx = 96) {
      if (!el) return true;
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      return distance <= thresholdPx;
    }

    function toggleMessageReasoning(message) {
      if (!message) return;
      const el = logEl.value;
      const previousTop = el ? el.scrollTop : 0;
      const wasNearBottom = logIsNearBottom(el);
      message.reasoningOpen = !message.reasoningOpen;
      nextTick(() => {
        if (!logEl.value) return;
        if (wasNearBottom) {
          logEl.value.scrollTop = logEl.value.scrollHeight;
        } else {
          logEl.value.scrollTop = previousTop;
        }
      });
    }

    watch(() => messages.value.length, () => {
      const el = logEl.value;
      const shouldStickToBottom = logIsNearBottom(el);
      nextTick(() => {
        if (shouldStickToBottom && logEl.value) {
          logEl.value.scrollTop = logEl.value.scrollHeight;
        }
      });
    });

    const nowRef = ref(Date.now());
    let tickInterval;
    let canvasResizeObserver = null;
    let topbarResizeObserver = null;
    let worldMapResizeHandler = null;
    let companionStripDocClickHandler = null;
    onMounted(() => {
      themeMediaQuery = window.matchMedia("(prefers-color-scheme: light)");
      const onSystemThemeChange = () => {
        if (themePref.value !== "system") return;
        window.PlayWebTheme?.applyTheme?.("system");
        nextTick(() => {
          updateStatsChart();
          updateCurrentCharts();
        });
      };
      if (themeMediaQuery.addEventListener) {
        themeMediaQuery.addEventListener("change", onSystemThemeChange);
      } else if (themeMediaQuery.addListener) {
        themeMediaQuery.addListener(onSystemThemeChange);
      }
      tickInterval = setInterval(() => { nowRef.value = Date.now(); }, 500);
      loadInventoryIcons();
      nextTick(() => {
        ensureWorldMapRenderer();
        syncGameTopbarHeight();
        if (isArcGame.value) ensureArcFrameHoverObserver();
        const topbarPanel = document.querySelector(".game-panel--overlay");
        if (topbarPanel && typeof ResizeObserver !== "undefined") {
          topbarResizeObserver = new ResizeObserver(() => syncGameTopbarHeight());
          topbarResizeObserver.observe(topbarPanel);
        }
      });
      worldMapResizeHandler = () => {
        syncGameTopbarHeight();
        worldMapRenderer?.resize();
      };
      window.addEventListener("resize", worldMapResizeHandler);
      companionStripDocClickHandler = (event) => {
        if (!companionStripSettingsOpen.value) return;
        const target = event.target;
        const wrap = document.querySelector(".companion-strip-settings-wrap");
        // The popover lives in <body> via teleport, so it is not inside `wrap`.
        const popover = document.querySelector(".companion-strip-popover--floating");
        if (wrap && wrap.contains(target)) return;
        if (popover && popover.contains(target)) return;
        companionStripSettingsOpen.value = false;
      };
      document.addEventListener("click", companionStripDocClickHandler);
    });
    onUnmounted(() => {
      if (topbarResizeObserver) {
        topbarResizeObserver.disconnect();
        topbarResizeObserver = null;
      }
      if (worldMapResizeHandler) {
        window.removeEventListener("resize", worldMapResizeHandler);
        worldMapResizeHandler = null;
      }
      if (companionStripDocClickHandler) {
        document.removeEventListener("click", companionStripDocClickHandler);
        companionStripDocClickHandler = null;
      }
      destroyArcFrameHoverObserver();
      worldMapRenderer?.destroy();
      worldMapRenderer = null;
      if (canvasResizeObserver) {
        canvasResizeObserver.disconnect();
        canvasResizeObserver = null;
      }
      if (tickInterval) clearInterval(tickInterval);
      if (statsIntervalId) clearInterval(statsIntervalId);
      if (currentTrajTimerId) clearInterval(currentTrajTimerId);
      stopInstructionExampleRotation();
      stopCompanionBenchPoll();
      hideInstructionPlayNotice();
      if (settingsSaveTimerId) clearTimeout(settingsSaveTimerId);
      stopAgentPromptRefresh();
      stopLoadingVisuals();
      detachApiKeyGuidePositionListeners();
      if (statsChart) statsChart.destroy();
      if (currentQAChart) currentQAChart.destroy();
      if (currentLenChart) currentLenChart.destroy();
    });

    function elapsedSeconds(sentAt) { return Math.floor((nowRef.value - sentAt) / 1000); }
    function formatResponseTime(ms) { return ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : `${ms}ms`; }
    function resizeQuestionInput() {
      const el = questionInputEl.value;
      if (!el) return;
      el.style.height = "auto";
      const maxHeight = parseFloat(getComputedStyle(el).maxHeight) || 120;
      const nextHeight = Math.min(el.scrollHeight, maxHeight);
      el.style.height = `${nextHeight}px`;
      el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
    }

    function clearQuestionInput() {
      questionInput.value = "";
      nextTick(() => {
        const el = questionInputEl.value;
        if (!el) return;
        el.style.height = "auto";
        el.style.overflowY = "hidden";
        resizeQuestionInput();
      });
    }

    function onQuestionInputEnter(event) {
      if (event.shiftKey) return;
      event.preventDefault();
      sendOracle();
    }

    function sendOracle() {
      if (!composerPanelActive.value) return;
      const q = questionInput.value.trim();
      if (!q) return;
      if (q.toUpperCase() === "END.") {
        clearQuestionInput();
        agentDirectChat("END.");
        return;
      }
      if (agentDirectChatActive.value) {
        messages.value.push({ kind: "human_agent", question: q, answer: "", ok: true, pending: true, sentAt: Date.now() });
        clearQuestionInput();
        agentDirectChat(q);
        return;
      }
      if (interactionMode.value === "human") {
        const question = pendingAgentQuestion.value || "Operator message";
        const now = Date.now();
        const pendingIdx = findLastPendingAgentMessageIndex(messages.value);
        if (pendingIdx >= 0) {
          const prev = messages.value[pendingIdx];
          messages.value[pendingIdx] = {
            ...prev,
            rawQuestion: prev.rawQuestion || question,
            answer: q,
            ok: true,
            error: null,
            pending: false,
            responseTimeMs: prev.sentAt != null ? now - prev.sentAt : null,
          };
        } else {
          messages.value.push({
            kind: "agent_message",
            question,
            answer: q,
            ok: true,
            error: null,
            pending: false,
            sentAt: now,
            responseTimeMs: null,
          });
        }
        clearQuestionInput();
        send({ type: "operator_answer", question, answer: q });
        pendingAgentQuestion.value = "";
        return;
      }
      messages.value.push({ kind: "oracle", question: q, pending: true, sentAt: Date.now() });
      clearQuestionInput();
      oracleAsk(q, true, forcedExpert.value);
    }

    function toggleAgentDirectChat() {
      if (agentDirectChatActive.value) {
        agentDirectChat("END.");
        return;
      }
      agentDirectChatActive.value = true;
      messages.value.push({
        kind: "system",
        question: "",
        answer: "Direct agent chat started. Messages go to the agent as-is (with prior tick history). Send END. to exit.",
        ok: true,
        pending: false,
      });
    }

    function clearChat() {
      messages.value = [];
      try { localStorage.removeItem("oracleChatMessages"); } catch (e) {}
    }

    async function loadSessionConfig() {
      try {
        const resp = await apiFetch("/session_config");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        applyAppCapabilities(data);
        interactionMode.value = data.interaction_mode || "oracle";
        allExperts.value = Array.isArray(data.all_experts) ? data.all_experts : [];
        selectedExperts.value = Array.isArray(data.allowed_experts) ? data.allowed_experts : [];
        normalizeOperatorSelection();
        alignMegapromptToWorldMode();
        if (data.max_agent_steps_per_tick != null) {
          maxAgentStepsPerClick.value = Math.max(1, Number(data.max_agent_steps_per_tick) || 1);
        }
        if (data.default_agent_steps != null) {
          agentStepsPerClick.value = clampAgentStepsPerClick(data.default_agent_steps);
        } else {
          agentStepsPerClick.value = clampAgentStepsPerClick(agentStepsPerClick.value);
        }
        if (data.expert_models && typeof data.expert_models === "object") {
          expertModelFields.value = {
            ...expertModelFields.value,
            ...Object.fromEntries(
              Object.entries(data.expert_models).map(([k, v]) => [k, String(v || "")]),
            ),
          };
        }
        if (data.expert_modes && typeof data.expert_modes === "object") {
          expertModeFields.value = {
            ...expertModeFields.value,
            ...Object.fromEntries(
              Object.entries(data.expert_modes).map(([k, v]) => [
                k,
                String(v || "hub").toLowerCase() === "openrouter" ? "openrouter" : "hub",
              ]),
            ),
          };
        }
        activeAgentModel.value = String(data.active_agent_model || "");
        activeAgentMode.value = "openrouter";
        applyModelCheck(data);
        hfTokenPreview.value = data.hf_token_configured ? String(data.hf_token_preview || "configured") : "";
        openrouterApiKeyPreview.value = data.openrouter_api_key_configured
          ? String(data.openrouter_api_key_preview || "configured")
          : "";
        if (typeof data.agent_direct_chat_active === "boolean") {
          agentDirectChatActive.value = data.agent_direct_chat_active;
        }
        gameKind.value = String(data.game_kind || "craftax") === "arc_agi" ? "arc_agi" : "craftax";
        arcGameId.value = String(data.arc_game_id || arcGameId.value || "ls20");
        if (Array.isArray(data.arc_game_options) && data.arc_game_options.length) {
          arcGameOptions.value = data.arc_game_options;
        }
        ensureDemoArcGameSelection();
        exoPlanetEnabled.value = Boolean(data.exo_planet_enabled);
        if (Array.isArray(data.megaprompt_options) && data.megaprompt_options.length) {
          megapromptOptions.value = data.megaprompt_options.map((name) => String(name));
        }
        if (data.megaprompt_config_name) {
          megapromptConfigName.value = String(data.megaprompt_config_name);
        }
        const loadedArcPromptExtra = String(data.arc_prompt_extra || "");
        arcPromptExtra.value = "";
        if (isArcGame.value) {
          ensureArcCompanionMode();
        }
        alignMegapromptToWorldMode();
        if (data.campaign_state) {
          campaignState.value = data.campaign_state;
        }
        if (loadedArcPromptExtra.trim()) {
          try {
            await apiFetch("/session_config", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(buildSessionConfigPayload()),
            });
          } catch (_e) {}
        }
      } catch (e) {}
    }

    function buildSessionConfigPayload() {
      syncExpertModesToActiveAgent();
      syncExpertModelsToActiveAgent();
      normalizeOperatorSelection();
      alignMegapromptToWorldMode();
      const { hf, openrouter } = apiSecretsForRequest();
      const payload = {
        interaction_mode: interactionMode.value,
        allowed_experts: selectedExperts.value,
        forced_expert: interactionMode.value === "human" ? null : forcedExpert.value,
        default_agent_steps: clampAgentStepsPerClick(agentStepsPerClick.value),
        game_kind: gameKind.value,
        arc_game_id: arcGameId.value,
        exo_planet_enabled: exoPlanetEnabled.value,
        ...((companionModeEnabled.value || isArcGame.value) ? { campaign_enabled: true } : {}),
        player_name: playerNickname.value.trim(),
        player_avatar_id: playerAvatarId.value,
      };
      if (featureFlags.value.observation_format_selection) {
        payload.megaprompt_config_name = megapromptConfigName.value;
      }
      if (featureFlags.value.expert_model_settings) {
        payload.expert_models = expertModelFields.value;
        payload.expert_modes = expertModeFields.value;
      }
      if (featureFlags.value.model_selection) {
        payload.active_agent_model = activeAgentModel.value;
        payload.active_agent_mode = activeAgentMode.value;
      }
      if (featureFlags.value.arc_prompt_override) {
        payload.arc_prompt_extra = arcPromptExtra.value;
      }
      if (featureFlags.value.settings_api_keys) {
        if (hf) payload.hf_token = hf;
        if (openrouter) payload.openrouter_api_key = openrouter;
      }
      return payload;
    }

    async function toggleExoPlanet() {
      const previousExo = exoPlanetEnabled.value;
      exoPlanetEnabled.value = !previousExo;
      alignMegapromptToWorldMode();
      const result = await saveSessionConfig("World mode updated — game reset.");
      if (!result.ok) {
        exoPlanetEnabled.value = previousExo;
        alignMegapromptToWorldMode();
        return;
      }
      nextTick(() => {
        ensureWorldMapRenderer();
        worldMapRenderer?.recenter?.();
        syncGameTopbarHeight();
      });
    }

    function flashSettingsMessage(message, ok = true) {
      settingsSaveOk.value = ok;
      settingsSaveMessage.value = message;
      if (settingsSaveTimerId) clearTimeout(settingsSaveTimerId);
      settingsSaveTimerId = setTimeout(() => {
        settingsSaveMessage.value = "";
      }, 4000);
    }

    async function saveSessionConfig(successMessage, { silent = false } = {}) {
      const okMessage = typeof successMessage === "string" ? successMessage : "Settings updated.";
      if (!silent) {
        settingsSaveMessage.value = "";
        if (settingsSaveTimerId) clearTimeout(settingsSaveTimerId);
      }
      try {
        const resp = await apiFetch("/session_config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(buildSessionConfigPayload()),
        });
        const data = await resp.json();
        if (!resp.ok || data.ok === false) throw new Error(data.error || `HTTP ${resp.status}`);
        applyConfigResponse(data);
        persistApiSecretsToStorage();
        persistSharedSettingsToStorage();
        if (!silent) {
          showSettingsSaveOutcome(data);
          if (settingsSaveOk.value) settingsSaveMessage.value = okMessage;
          showSaveToast(settingsSaveOk.value, settingsSaveMessage.value || okMessage);
        }
        return { ok: true };
      } catch (e) {
        if (!silent) {
          settingsSaveOk.value = false;
          settingsSaveMessage.value = `Failed to update settings${e?.message ? `: ${e.message}` : "."}`;
          showSaveToast(false, settingsSaveMessage.value);
        }
        return { ok: false, error: e?.message || "Failed to update settings." };
      }
      if (!silent) {
        settingsSaveTimerId = setTimeout(() => {
          settingsSaveMessage.value = "";
        }, 4000);
      }
    }

    async function saveSettingsForNextSession() {
      try {
        persistSharedSettingsToStorage();
        await saveSessionConfig("Saved for this session and the next time you open the page.");
      } catch (e) {
        flashSettingsMessage(`Failed to save preferences${e?.message ? `: ${e.message}` : "."}`, false);
      }
    }

    async function resetToDefaultSettings() {
      if (!window.confirm("Reset all settings to defaults from config/oracle_config.yaml? Saved browser preferences will be cleared.")) {
        return;
      }
      settingsSaveMessage.value = "";
      if (settingsSaveTimerId) clearTimeout(settingsSaveTimerId);
      try {
        const resp = await apiFetch("/session_config/reset", { method: "POST" });
        const data = await resp.json();
        if (!resp.ok || data.ok === false) throw new Error(data.error || `HTTP ${resp.status}`);
        try { localStorage.removeItem(SESSION_PREFS_STORAGE_KEY); } catch (e) {}
        clearWizardComplete();
        window.PlayWebSession?.clearApiSecrets?.();
        applyConfigResponse(data);
        agentGoal.value = "";
        hfTokenInput.value = "";
        openrouterApiKeyInput.value = "";
        showSettingsSaveOutcome(data);
        settingsSaveMessage.value = settingsSaveOk.value
          ? "Defaults restored."
          : settingsSaveMessage.value || "Defaults restored.";
        window.location.replace("./index.html");
      } catch (e) {
        flashSettingsMessage(`Failed to reset settings${e?.message ? `: ${e.message}` : "."}`, false);
      }
    }

    function resolveExpertName(expertKey) {
      const defaults = {
        map: "map_expert",
        action: "action_expert",
        mechanics: "mechanics_expert",
        goal: "goal_expert",
        path: "path_expert",
      };
      const fallback = defaults[expertKey];
      if (!fallback) return null;
      const candidates = Array.isArray(allExperts.value) ? allExperts.value : [];
      const exact = candidates.find((name) => name === fallback);
      if (exact) return exact;
      const partial = candidates.find((name) => name.includes(expertKey));
      return partial || fallback;
    }

    function isExpertButtonActive(buttonId) {
      if (buttonId === "human") return interactionMode.value === "human";
      const expertName = resolveExpertName(buttonId);
      if (!expertName) return false;
      return interactionMode.value !== "human" && selectedExperts.value.includes(expertName);
    }

    function expertButtonClass(buttonId) {
      const active = isExpertButtonActive(buttonId);
      return active ? "btn-primary" : "btn-outline-primary";
    }

    async function toggleExpertButton(buttonId) {
      if (buttonId === "human") {
        interactionMode.value = "human";
        selectedExperts.value = [];
        await saveSessionConfig();
        return;
      }
      const expertName = resolveExpertName(buttonId);
      if (!expertName) return;
      if (interactionMode.value === "human") interactionMode.value = "oracle";
      selectedExperts.value = [expertName];
      forcedExpert.value = expertName;
      await saveSessionConfig();
    }

    async function selectSettingsAiOperator() {
      if (isArcGame.value) return;
      await toggleExpertButton("goal");
    }

    async function selectSettingsHumanOperator() {
      if (interactionMode.value === "human") return;
      await toggleExpertButton("human");
    }

    async function selectSettingsHumanAvatar(avatarId) {
      selectPlayerAvatar(avatarId, { syncServer: true });
      if (interactionMode.value !== "human") {
        await toggleExpertButton("human");
      }
    }

    function onSettingsNicknameInput() {
      onPlayerNicknameInput();
      if (interactionMode.value !== "human" && playerNickname.value.trim()) {
        interactionMode.value = "human";
        selectedExperts.value = [];
        saveSessionConfig(undefined, { silent: true });
      }
    }

    function syncExpertModesToActiveAgent() {
      // This deployment is locked to OpenRouter: every expert always uses it.
      activeAgentMode.value = "openrouter";
      const next = { ...expertModeFields.value };
      for (const key of EXPERT_MODE_KEYS) {
        next[key] = "openrouter";
      }
      expertModeFields.value = next;
    }

    function syncExpertModelsToActiveAgent() {
      const model = String(activeAgentModel.value || "").trim();
      if (!model) return;
      const next = { ...expertModelFields.value };
      for (const key of EXPERT_MODE_KEYS) {
        next[key] = model;
      }
      expertModelFields.value = next;
    }

    function applyUnifiedGateway(gateway) {
      const mode = String(gateway || "openrouter").toLowerCase() === "hub" ? "hub" : "openrouter";
      activeAgentMode.value = mode;
      activeAgentModel.value = adaptModelIdForGateway(activeAgentModel.value, mode);
      syncExpertModesToActiveAgent();
      syncExpertModelsToActiveAgent();
      setupWizardGateway.value = mode;
    }

    const wizardKeyOnServer = computed(() =>
      setupWizardGateway.value === "openrouter"
        ? Boolean(openrouterApiKeyPreview.value)
        : Boolean(hfTokenPreview.value),
    );

    function wizardTokenAvailable() {
      const token = setupWizardToken.value.trim();
      if (token) return true;
      if (wizardKeyOnServer.value) return true;
      const stored = apiSecretsForRequest();
      return setupWizardGateway.value === "openrouter" ? Boolean(stored.openrouter) : Boolean(stored.hf);
    }

    function wizardOperatorBadges(caps) {
      const human = Boolean(caps?.human_operator);
      const ai = Boolean(caps?.ai_operator);
      return WIZARD_OPERATOR_BADGE_DEFS.map((def) => {
        const available = def.key === "human" ? human : ai;
        return {
          ...def,
          available,
          tooltip: available ? def.availableTooltip : def.unavailableTooltip,
        };
      });
    }

    function wizardOperatorBadgesForRealm(realmKey) {
      return wizardOperatorBadges(WIZARD_REALM_OPERATOR_CAPS[realmKey] || {});
    }

    function wizardOperatorBadgesForArcOption(option) {
      return wizardOperatorBadges(option || {});
    }

    const wizardCapabilityTip = ref({
      visible: false,
      text: "",
      style: {},
    });

    function showWizardCapabilityTip(event, text) {
      const el = event?.currentTarget;
      if (!el?.getBoundingClientRect) return;
      const rect = el.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      wizardCapabilityTip.value = {
        visible: true,
        text: String(text || "").trim(),
        style: {
          left: `${centerX}px`,
          top: `${rect.top - 8}px`,
        },
      };
    }

    function hideWizardCapabilityTip() {
      if (!wizardCapabilityTip.value.visible) return;
      wizardCapabilityTip.value = { ...wizardCapabilityTip.value, visible: false };
    }

    function arcGamePreviewSrc(gameId) {
      const id = String(gameId || "").trim().toLowerCase();
      if (!/^[a-z0-9]{4}$/.test(id)) return "";
      return `./assets/arc-games/${id}.png`;
    }

    function arcGamePreviewVisible(gameId) {
      const id = String(gameId || "").trim().toLowerCase();
      return Boolean(arcGamePreviewSrc(id) && !arcGamePreviewBroken.value[id]);
    }

    function onArcGamePreviewError(gameId) {
      const id = String(gameId || "").trim().toLowerCase();
      if (!id || arcGamePreviewBroken.value[id]) return;
      arcGamePreviewBroken.value = { ...arcGamePreviewBroken.value, [id]: true };
    }

    function arcGameAvailable(option) {
      if (!option || typeof option !== "object") return false;
      if (Object.prototype.hasOwnProperty.call(option, "available_locally")) {
        return Boolean(option.available_locally);
      }
      return true;
    }

    function syncWizardArcDefaults() {
      if (!isArcGame.value) return;
      ensureArcCompanionMode();
      wizardStepTouched.value = {
        ...wizardStepTouched.value,
        role: interactionMode.value === "human",
        playMode: companionModeEnabled.value,
      };
    }

    watch(setupWizardStep, (step) => {
      hideWizardCapabilityTip();
      if (step === 3 || step === 4) syncWizardArcDefaults();
    });

    const wizardCanContinue = computed(() => {
      if (setupWizardStep.value === 1) {
        return isPlayerProfileComplete({ nickname: playerNickname.value });
      }
      if (setupWizardStep.value === 2) return wizardStepTouched.value.realm;
      if (setupWizardStep.value === 3) return wizardStepTouched.value.role;
      if (setupWizardStep.value === 4) return wizardStepTouched.value.playMode;
      return false;
    });

    function selectWizardRealm(theme, arcId = null) {
      const normalized = String(theme || "craftax");
      gameKind.value = normalized === "arc_agi" ? "arc_agi" : "craftax";
      exoPlanetEnabled.value = gameKind.value === "craftax" && normalized === "exo-planet";
      if (gameKind.value === "arc_agi") {
        const nextArcId = String(arcId || arcGameId.value || "ls20").trim().toLowerCase();
        arcGameId.value = nextArcId || "ls20";
        ensureArcCompanionMode();
      }
      alignMegapromptToWorldMode();
      normalizeOperatorSelection();
      restoreAgentModelForCurrentContext();
      ensureDemoAgentModelPreset();
      wizardStepTouched.value = { ...wizardStepTouched.value, realm: true };
    }

    function selectWizardRole(role) {
      if (isArcGame.value && role !== "human") return;
      interactionMode.value = role === "human" ? "human" : "oracle";
      normalizeOperatorSelection();
      wizardStepTouched.value = { ...wizardStepTouched.value, role: true };
    }

    function selectWizardPlayMode(mode) {
      if (isArcGame.value && mode !== "companion") return;
      companionModeEnabled.value = mode === "companion";
      wizardStepTouched.value = { ...wizardStepTouched.value, playMode: true };
      persistSharedSettingsToStorage();
    }

    function wizardBack() {
      wizardTokenError.value = "";
      if (setupWizardStep.value > 1) setupWizardStep.value -= 1;
    }

    function applyWizardTokenToInputs() {
      const token = setupWizardToken.value.trim();
      if (!token) return;
      if (setupWizardGateway.value === "openrouter") {
        openrouterApiKeyInput.value = token;
        return;
      }
      hfTokenInput.value = token;
    }

    function resetWorldDuringLoad({ timeoutMs = 6000 } = {}) {
      return new Promise((resolve) => {
        if (status.value !== "connected") {
          resolve();
          return;
        }
        const prevFrame = currentFrame.value;
        resetGame();
        waitForFrameChange(prevFrame, { timeoutMs }).then(resolve);
      });
    }

    function waitForGameReady({ timeoutMs = 20000, minMs = 1200 } = {}) {
      return new Promise((resolve) => {
        const started = Date.now();
        const tick = () => {
          const elapsed = Date.now() - started;
          const connected = status.value === "connected";
          let progress = 45;
          if (connected) progress += 30;
          if (currentFrame.value) progress += 20;
          appLoadingProgress.value = Math.max(appLoadingProgress.value, Math.min(95, progress));
          if (connected && elapsed >= minMs) {
            resolve();
            return;
          }
          if (elapsed >= timeoutMs) {
            resolve();
            return;
          }
          setTimeout(tick, 150);
        };
        tick();
      });
    }

    function bumpLoadingProgress(target) {
      appLoadingProgress.value = Math.max(appLoadingProgress.value, target);
    }

    async function launchGameSession() {
      appLoading.value = true;
      appLoadingProgress.value = 0;
      appLoadingText.value = launchLoadingMessages[0];
      startLaunchLoadingVisuals();
      try {
        bumpLoadingProgress(10);
        appLoadingText.value = "Applying configuration...";
        await bootstrapSessionSettings();
        syncApiKeyGuideAfterConfig();
        connect();
        bumpLoadingProgress(35);
        appLoadingText.value = "Connecting to the world...";
        await waitForGameReady();
        syncArcCampaignOnConnect();
        syncCompanionCampaignOnConnect();
        bumpLoadingProgress(96);
        appLoadingText.value = "Generating the world...";
        await resetWorldDuringLoad();
        appLoadingProgress.value = 100;
        appLoadingText.value = "Ready";
        loadStatistics();
        loadCompanionBenchStatus();
        statsIntervalId = setInterval(loadStatistics, 5000);
        loadCurrentTrajectoryStats();
        currentTrajTimerId = setInterval(loadCurrentTrajectoryStats, 4000);
        const promptModalEl = document.getElementById("agentPromptModal");
        if (promptModalEl) {
          promptModalEl.addEventListener("hidden.bs.modal", stopAgentPromptRefresh);
        }
        stopLoadingVisuals();
        appLoading.value = false;
        await nextTick();
        tryConsumeCompanionStartHint();
        await syncWorldMapFromFrame();
      } catch (e) {
        stopLoadingVisuals();
        appLoading.value = false;
      }
    }

    async function finalizeWizardAndLaunch() {
      wizardTokenError.value = "";
      wizardLaunching.value = true;
      try {
        applyUnifiedGateway(setupWizardGateway.value);
        applyWizardTokenToInputs();
        agentGoal.value = "";
        alignMegapromptToWorldMode();
        // Do NOT save the config here: switching the world theme server-side
        // can take a minute and would freeze the wizard on "Entering...".
        // Persist everything locally, jump to the loading screen right away,
        // and let bootstrapSessionSettings apply the config under it.
        persistApiSecretsToStorage();
        try {
          localStorage.setItem(
            PENDING_WORLD_MODE_KEY,
            gameKind.value === "arc_agi"
              ? `arc_agi:${arcGameId.value}`
              : (exoPlanetEnabled.value ? "exo-planet" : "craftax"),
          );
        } catch (_e) {}
        markWizardComplete();
        if (companionModeEnabled.value) {
          markCompanionStartHintPending();
        }
        persistSharedSettingsToStorage();
        if (!isGameEntryUrl()) {
          window.location.replace(gameEntryUrl());
          return;
        }
        setupComplete.value = true;
        await launchGameSession();
      } catch (e) {
        wizardTokenError.value = `Failed to start session${e?.message ? `: ${e.message}` : "."}`;
        setupComplete.value = false;
        clearWizardComplete();
      } finally {
        wizardLaunching.value = false;
      }
    }

    async function wizardContinue() {
      if (setupWizardStep.value === 1) {
        persistPlayerProfile();
        wizardStepTouched.value = {
          ...wizardStepTouched.value,
          profile: isPlayerProfileComplete({ nickname: playerNickname.value }),
        };
      }
      if (setupWizardStep.value === 3 && isArcGame.value) {
        syncWizardArcDefaults();
      }
      if (setupWizardStep.value === 4) {
        await finalizeWizardAndLaunch();
        return;
      }
      if (setupWizardStep.value < 4) setupWizardStep.value += 1;
    }

    function openSetupWizard() {
      if (!featureFlags.value.setup_wizard) return;
      if (settingsModalInstance) {
        settingsModalInstance.hide();
      }
      const settingsEl = document.getElementById("settingsModal");
      if (settingsEl) {
        const modal = bootstrap.Modal.getInstance(settingsEl);
        if (modal) modal.hide();
      }
      clearWizardComplete();
      stripGameEntryUrl();
      setupWizardStep.value = 1;
      setupWizardToken.value = "";
      wizardTokenError.value = "";
      wizardStepTouched.value = { profile: false, realm: false, role: false, playMode: false };
      setupWizardGateway.value = activeAgentMode.value || "openrouter";
      setupComplete.value = false;
      appLoading.value = false;
    }

    onMounted(async () => {
      await prepareLoadingSprite();
      if (!setupComplete.value) {
        if (isGameEntryUrl()) stripGameEntryUrl();
        appLoading.value = false;
        try {
          await loadSessionConfig();
          applySharedSettingsFromStorage();
          alignMegapromptToWorldMode();
          setupWizardGateway.value = activeAgentMode.value || "openrouter";
        } catch (e) {}
        if (!featureFlags.value.setup_wizard) {
          markWizardComplete();
          setupComplete.value = true;
          await launchGameSession();
        }
        return;
      }
      await launchGameSession();
    });
    let statsIntervalId = null;

    async function loadStatistics() {
      oracleStatsError.value = "";
      campaignBenchmarkError.value = "";
      try {
        const resp = await apiFetch("/oracle_statistics");
        if (!resp.ok) throw new Error();
        const data = await resp.json();
        oracleStats.value = data.experts || null;
      } catch (err) { oracleStatsError.value = "Failed to load statistics"; }
      try {
        const since = String(benchmarkSince.value || "").trim();
        const boardPath = since
          ? `/campaign_benchmark?since=${encodeURIComponent(since)}`
          : "/campaign_benchmark";
        const respBoard = await apiFetch(boardPath);
        if (!respBoard.ok) throw new Error();
        const dataBoard = await respBoard.json();
        campaignBenchmark.value = window.PlayWebBenchmark
          ?.normalizeCampaignBenchmarkPayload?.(dataBoard)?.world_modes || dataBoard.world_modes || {};
        benchmarkSinceApplied.value = dataBoard.since ? String(dataBoard.since).slice(0, 10) : "";
      } catch (err) {
        campaignBenchmarkError.value = "Failed to load campaign benchmark";
        benchmarkSinceApplied.value = "";
      }
    }
    function onBenchmarkSinceChange() {
      window.PlayWebBenchmark?.saveBenchmarkSince?.(benchmarkSince.value);
      loadStatistics();
    }
    function clearBenchmarkSince() {
      benchmarkSince.value = "";
      window.PlayWebBenchmark?.saveBenchmarkSince?.("");
      loadStatistics();
    }
    function refreshStatistics() { loadStatistics(); }
    const oracleStatsPerExpert = computed(() => (!oracleStats.value ? {} : (({ total, ...rest }) => rest)(oracleStats.value)));
    function prettyExpertName(name) {
      if (!name) return "";
      const base = name.replace(/_expert$/, "");
      return base.charAt(0).toUpperCase() + base.slice(1);
    }

    function updateStatsChart() {
      if (!oracleStats.value || !statsChartEl.value) return;
      const stats = oracleStatsPerExpert.value;
      const expertNames = Object.keys(stats);
      if (!expertNames.length) {
        if (statsChart) statsChart.destroy();
        statsChart = null;
        return;
      }
      const labels = expertNames.map((name) => prettyExpertName(name));
      const messagesData = expertNames.map((name) => stats[name].messages || 0);
      const failuresData = expertNames.map((name) => stats[name].failures || 0);
      const data = {
        labels,
        datasets: [
          { label: "Calls", data: messagesData, backgroundColor: "rgba(232, 165, 75, 0.6)", borderColor: "rgba(232, 165, 75, 1)", borderWidth: 1 },
          { label: "Failures", data: failuresData, backgroundColor: "rgba(248, 113, 113, 0.6)", borderColor: "rgba(248, 113, 113, 1)", borderWidth: 1 },
        ],
      };
      const axisSize = chartFontSizes.value.stats.axis;
      const legendSize = chartFontSizes.value.stats.legend;
      const colors = window.PlayWebTheme?.chartThemeColors?.() || {};
      const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: colors.legend || "#e8e6e3", font: { size: legendSize } } } },
        scales: { x: chartAxisOptions(axisSize), y: { beginAtZero: true, ...chartAxisOptions(axisSize) } },
      };
      if (statsChart) { statsChart.data = data; statsChart.options = options; statsChart.update(); } else statsChart = new Chart(statsChartEl.value.getContext("2d"), { type: "bar", data, options });
    }

    async function loadCurrentTrajectoryStats() {
      currentTrajError.value = "";
      try {
        const resp = await apiFetch("/trajectory_current");
        if (!resp.ok) throw new Error();
        currentTrajStats.value = await resp.json();
      } catch (e) { currentTrajError.value = "Failed to load current trajectory stats"; }
    }

    function updateCurrentCharts() {
      const s = currentTrajStats.value;
      if (!s || !s.active) {
        if (currentQAChart) currentQAChart.destroy();
        if (currentLenChart) currentLenChart.destroy();
        currentQAChart = null;
        currentLenChart = null;
        return;
      }
      if (currentQAChartEl.value) {
        const denom = (s.questions || 0) + (s.actions || 0) || 1;
        const data = { labels: ["Questions", "Actions"], datasets: [{ label: "Share of steps (%)", data: [((s.questions || 0) / denom) * 100, ((s.actions || 0) / denom) * 100], backgroundColor: ["rgba(56, 189, 248, 0.75)", "rgba(110, 231, 183, 0.75)"], borderColor: ["rgba(56, 189, 248, 1)", "rgba(110, 231, 183, 1)"], borderWidth: 1 }] };
        const options = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label(context) { return `${context.label}: ${context.parsed.toFixed(1)}%`; } } } } };
        if (currentQAChart) { currentQAChart.data = data; currentQAChart.options = options; currentQAChart.update(); } else currentQAChart = new Chart(currentQAChartEl.value.getContext("2d"), { type: "doughnut", data, options });
      }
      if (currentLenChartEl.value) {
        const data2 = { labels: ["Actions", "Answer length (chars)"], datasets: [{ label: "Current episode", data: [s.actions || 0, s.mean_answer_len_chars || 0], backgroundColor: ["rgba(234, 179, 8, 0.75)", "rgba(244, 114, 182, 0.75)"], borderColor: ["rgba(234, 179, 8, 1)", "rgba(244, 114, 182, 1)"], borderWidth: 1 }] };
        const axisSize = chartFontSizes.value.currentLen.axis;
        const colors = window.PlayWebTheme?.chartThemeColors?.() || {};
        const gridAlt = colors.gridAlt || colors.grid || "rgba(31, 41, 55, 0.8)";
        const axisOpts = {
          ticks: { color: colors.tick || "#9ca3af", font: { size: axisSize } },
          grid: { color: gridAlt },
        };
        const options2 = {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label(context) {
                  const value = context.parsed.y ?? context.parsed;
                  return `${context.label}: ${value.toFixed(2)}`;
                },
              },
            },
          },
          scales: { x: axisOpts, y: { beginAtZero: true, ...axisOpts } },
        };
        if (currentLenChart) { currentLenChart.data = data2; currentLenChart.options = options2; currentLenChart.update(); } else currentLenChart = new Chart(currentLenChartEl.value.getContext("2d"), { type: "bar", data: data2, options: options2 });
      }
    }

    watch(oracleStats, () => nextTick(updateStatsChart), { deep: true });
    watch(currentTrajStats, () => nextTick(updateCurrentCharts), { deep: true });

    const companionBenchRows = computed(() => {
      const rows = companionBenchStatus.value?.rows;
      return Array.isArray(rows) ? rows : [];
    });

    const companionBenchResearchComplete = computed(() => {
      return Boolean(
        companionResearchSnapshot.value?.complete
        || campaignState.value?.is_finished,
      );
    });

    function meanQuestionsFromBenchRow(row) {
      const raw = row?.mean_questions ?? row?.mean_q;
      if (raw != null && raw !== "" && Number.isFinite(Number(raw))) {
        return Number(raw);
      }
      const perAgent = Array.isArray(row?.per_agent) ? row.per_agent : [];
      if (!perAgent.length) return 0;
      const total = perAgent.reduce(
        (acc, entry) => acc + Number(entry?.questions_count || 0),
        0,
      );
      return total / perAgent.length;
    }

    function formatBenchSuccessRate(sr) {
      const n = Number(sr);
      if (!Number.isFinite(n)) return "—";
      return `${Math.round(n * 100)}%`;
    }

    function formatBenchMeanQuestions(meanQ) {
      const n = Number(meanQ);
      if (!Number.isFinite(n)) return "—";
      if (Math.abs(n - Math.round(n)) < 1e-6) return String(Math.round(n));
      return n.toFixed(1);
    }

    const companionBenchTestResultRows = computed(() => {
      return companionBenchRows.value.map((row) => ({
        model: String(row.model || "unknown"),
        task_title: String(row.task_title || row.task_key || "—"),
        sr: Number(row.sr || 0),
        mean_q: meanQuestionsFromBenchRow(row),
        runs: Number(row.runs || 0),
        limit_violation: Boolean(row.limit_violation) || Number(row.violation_runs || 0) > 0,
      }));
    });

    const companionBenchTableRows = computed(() => {
      const byModel = new Map();
      for (const row of companionBenchRows.value) {
        const model = String(row.model || "unknown");
        if (!byModel.has(model)) byModel.set(model, { model, tasks: {} });
        byModel.get(model).tasks[String(row.task_key || "")] = {
          sr: Number(row.sr || 0),
          mean_q: meanQuestionsFromBenchRow(row),
          runs: Number(row.runs || 0),
          successes: Number(row.successes || 0),
          limit_violation: Boolean(row.limit_violation) || Number(row.violation_runs || 0) > 0,
        };
      }
      return Array.from(byModel.values()).sort((a, b) => a.model.localeCompare(b.model));
    });

    const companionBenchStage = computed(() => String(companionBenchStatus.value?.current_stage || ""));
    const companionBenchInResearch = computed(() => companionResearchActive.value);
    const companionBenchInTest = computed(() => {
      if (!companionBenchStatus.value?.running) return false;
      return companionBenchStage.value.startsWith("parallel:");
    });

    const companionBenchResearchProgressPct = computed(() => {
      const snap = companionResearchSnapshot.value;
      if (snap?.progress_pct != null && companionResearchActive.value) {
        return Number(snap.progress_pct);
      }
      if (campaignState.value?.is_finished || companionBenchResearchComplete.value) return 100;
      const total = Number(campaignState.value?.total_count || 0);
      const completed = Number(campaignState.value?.completed_count || 0);
      if (total > 0) return Math.round((100 * completed) / total);
      return 0;
    });

    const companionBenchResearchStatusText = computed(() => {
      if (campaignState.value?.is_finished && !companionResearchActive.value) {
        return "Research complete";
      }
      return "";
    });

    const companionStripStatusText = computed(() => {
      const researchStatus = companionBenchResearchStatusText.value;
      if (researchStatus) return researchStatus;
      if (campaignState.value?.is_finished) return "All levels done";
      const title = String(campaignState.value?.current_task_title || "").trim();
      if (title) return title;
      return "Ready";
    });

    const companionBenchAgentSlotCount = computed(() => {
      const n = Number(companionBenchStatus.value?.parallel_agents ?? companionBenchParallelAgents.value);
      if (!Number.isFinite(n) || n <= 0) return BENCH_RENDER_SLOTS;
      return Math.max(1, Math.min(BENCH_RENDER_SLOTS, Math.round(n)));
    });

    const companionBenchTestProgressRows = computed(() => {
      const source = Array.isArray(companionBenchStatus.value?.agents_progress)
        ? companionBenchStatus.value.agents_progress
        : [];
      const byId = new Map(source.map((row) => [Number(row.agent_id || 0), row]));
      const inTest = companionBenchInTest.value;
      const slotCount = companionBenchAgentSlotCount.value;
      const defaultMaxTicks = Number(companionBenchStatus.value?.max_ticks_per_task || 0);
      return Array.from({ length: BENCH_RENDER_SLOTS }, (_, idx) => {
        const agent_id = idx + 1;
        const row = byId.get(agent_id);
        const slotEnabled = agent_id <= slotCount;
        const active = inTest && slotEnabled;
        return {
          agent_id,
          tick: Number(row?.tick || 0),
          max_ticks: Number(row?.max_ticks || defaultMaxTicks),
          progress_pct: active ? Number(row?.progress_pct || 0) : 0,
          done: Boolean(row?.done),
          active,
          slotEnabled,
        };
      });
    });

    const companionBenchRenderCards = computed(() => {
      const inTest = companionBenchInTest.value;
      const slotCount = companionBenchAgentSlotCount.value;
      const liveById = new Map();
      for (const agent of Array.isArray(companionBenchStatus.value?.agents_live)
        ? companionBenchStatus.value.agents_live
        : []) {
        liveById.set(Number(agent.agent_id || 0), agent);
      }
      return Array.from({ length: BENCH_RENDER_SLOTS }, (_, idx) => {
        const agent_id = idx + 1;
        const slotEnabled = agent_id <= slotCount;
        const active = inTest && slotEnabled;
        const frame = active ? liveById.get(agent_id)?.frame : null;
        return {
          agent_id,
          title: `Agent ${agent_id}`,
          active,
          slotEnabled,
          frameSrc: active ? companionBenchFrameSrc(frame) : "",
        };
      });
    });

    return { wsStatus, reward, done, playerPosition, agentObservation, agentObservationDisplay, agentObservationImageSrc, agentReasoning, agentReasoningThinking, agentReasoningDisplay, agentStatusBannerVisible, agentStopping, messages, campaignState, companionModeEnabled, exoPlanetEnabled, gameKind, isArcGame, isArcImagePrompt, isArcGridImagePrompt, isArcTwoImagePrompt, arcImageWarning, operatorPanelTitle, operatorPanelSubtitle, operatorInputPlaceholder, operatorEmptyText, composerPanelActive, composerInactiveMessage, operatorChatExamples, composerResponderIsHuman, composerResponderAvatarSrc, composerResponderDisplayName, composerResponderTooltip, arcGameId, arcGameOptions, arcGamePreviewSrc, arcGamePreviewVisible, onArcGamePreviewError, arcGameAvailable, wizardOperatorBadgesForRealm, wizardOperatorBadgesForArcOption, wizardCapabilityTip, showWizardCapabilityTip, hideWizardCapabilityTip, arcFrameImageSrc, arcFrameShellEl, arcFrameCanvasEl, onArcFramePointerMove, onArcFramePointerLeave, onArcFrameClick, arcScore, arcScoreIsFinal, humanLeaderboardRows, arcPlayerName, arcScoreError, arcScoreSubmitting, openHumanLeaderboardModal, openArcScoreModal, loadHumanLeaderboard, loadArcHumanLeaderboard, submitArcHumanScore, humanLeaderboardTypeLabel, leaderboardAvatarSrc, humanLeaderboardProgressLabel, humanLeaderboardLevelsLabel, humanLeaderboardStepsLabel, humanLeaderboardQuestionsLabel, humanLeaderboardLevelBreakdown, humanLeaderboardPerLevelCell, humanLeaderboardDetailsLabel, playerNickname, playerAvatarId, playerAvatarSrc, playerDisplayName, aiOperatorDisplayName: AI_OPERATOR_DISPLAY_NAME, aiOperatorAvatarSrc: AI_OPERATOR_AVATAR_SRC, avatarOptions, avatarSrcFor, selectPlayerAvatar, onPlayerNicknameInput, onSettingsNicknameInput, selectSettingsAiOperator, selectSettingsHumanOperator, selectSettingsHumanAvatar, persistPlayerProfile, showHumanOperatorIdentity, showAiOperatorIdentity, showOperatorAuthorIdentity, operatorAuthorAvatarSrc, operatorAuthorDisplayName, worldModeLabel, worldModeAgentIconSrc, worldModeToggleIconSrc, worldModeToggleLabel, campaignProgressPct, campaignProgressText, campaignTasks, campaignPhase2CompletedKeys, campaignPhase2SelectedKey, companionResearchActive, companionKnowledgeOptions, companionTestKnowledgeOptions, nowRef, worldCanvasEl, worldHud, inventorySlots, worldStatIconSrc, selectedTileInfo, tileInfoStyle, tileInfoFading, closeTileInfo, formatInventoryKey, recenterWorldMap, mapFollowingAgent, zoomWorldIn, zoomWorldOut, operatorPanelMessageGlow, logEl, questionInput, questionInputEl, resizeQuestionInput, onQuestionInputEnter, interactionMode, allExperts, selectedExperts, forcedExpert, expertButtons, isExpertButtonActive, expertButtonClass, toggleExpertButton, toggleCompanionMode, toggleExoPlanet, openWorldModeModal, chooseWorldMode, worldModeSwitching, companionResearchLevelClass, companionStripNodes, companionStripFillPct, companionStripAgentRunning, companionStepsLocked, companionStepBadge, companionStepGaugeCircumference, companionStepGaugeOffset, companionStripSettingsOpen, companionStripPopoverStyle, toggleCompanionStripSettings, closeCompanionStripSettings, onCompanionIconError, companionTip, showCompanionTip, hideCompanionTip, phase2LevelClass, startPhase2Level, actions, agentStepsPerClick, maxAgentStepsPerClick, agentStepsRangeMax, clampAgentStepsPerClick, agentStop, agentGoal, agentInstructionReady, agentInstructionLocked, agentInstructionPlaceholder, agentWorking, agentMissionActive, agentDirectChatActive, instructionPlayNoticeVisible, instructionPlayNoticeFading, instructionPlayNoticeSteps, companionPlayHighlightVisible, instructionStepBadge, instructionStepTimerCircumference, instructionStepTimerOffset, saveTrajectory, statsPanelCollapsed, statsPanelNeonAlert, statsPanelStyle, startStatsPanelResize, operatorPanelStyle, operatorPanelCollapsed, operatorPanelMessageAlert, toggleOperatorPanel, openOperatorPanel, startOperatorPanelResize, reasoningPanelCollapsed, reasoningToggleLive, reasoningNeonAlert, knowledgeNeonAlert, reasoningPanelSize, toggleReasoningPanel, closeReasoningPanel, oracleStats, oracleStatsError, campaignBenchmarkCompact, campaignBenchmarkError, campaignBenchmarkRuns, benchmarkSince, benchmarkSinceApplied, onBenchmarkSinceChange, clearBenchmarkSince, formatBenchSuccessRate, formatBenchMeanQuestions, statsChartEl, currentTrajStats, currentTrajError, currentQAChartEl, currentLenChartEl, CHART_FONT_MIN, CHART_FONT_MAX, chartFontSizes, onChartFontChange, expertUiRows, expertModelFields, expertModeFields, megapromptConfigName, megapromptOptions, arcPromptExtra, saveArcPromptExtra, clearArcPromptExtra, useCurrentAgentPromptAsArcOverride, activeAgentModel, activeAgentModelPresets, selectedAgentModelPreset, activeAgentModelPresetInfo, selectAgentModelPreset, activeAgentMode, hfTokenInput, openrouterApiKeyInput, hfTokenPreview, openrouterApiKeyPreview, settingsSaveMessage, settingsSaveOk, appProfile, isDemoProfile, featureFlags, themePref, setThemePref, setupComplete, setupWizardStep, setupWizardSteps, setupWizardGateway, setupWizardToken, wizardTokenError, wizardLaunching, wizardCanContinue, wizardKeyOnServer, wizardStepTouched, selectWizardRealm, selectWizardRole, selectWizardPlayMode, wizardBack, wizardContinue, openSetupWizard, apiKeyAlertActive, apiKeyAlertMessage, apiKeyMissingKeys, apiKeyGuideFloatStyle, modelErrorAlert, dismissModelError, openSettingsFromModelError, invalidModels, saveToast, dismissSaveToast, achievementToast, dismissAchievementToast, appLoading, appLoadingText, appLoadingProgress, appLoadingSpriteSrc, reset, step, sendOracle, toggleAgentDirectChat, saveSessionConfig, saveSettingsForNextSession, resetToDefaultSettings, agentStep, agentPlayToggle, onAgentInstructionEnter, toggleSaveTrajectory, toggleStatsPanel, refreshStatistics, elapsedSeconds, formatResponseTime, formatLogAnswer, logMessageIsError, logMessageHeader, logMessageReasoning, toggleMessageReasoning, clearChat, agentKnowledge, agentKnowledgeLoading, fetchAgentKnowledge, openKnowledgeModal, openSettingsModal, openObservationModal, agentPromptGoal, agentPromptSystem, agentPromptUser, agentPromptSystemHtml, agentPromptUserHtml, agentPromptUserSections, agentPromptViewMode, agentPromptLoading, agentPromptHasPrompt, agentPromptError, fetchAgentPrompt, openAgentPromptModal, scrollAgentPromptToSection, companionBenchStatus, companionBenchError, companionBenchStarting, companionBenchStopping, companionBenchMaxTicksPerTask, companionBenchMaxTicksDirty, companionBenchParallelAgents, companionBenchParallelAgentsDirty, companionBenchCycles, companionBenchCyclesDirty, companionBenchTestTaskKey, companionBenchKnowledgeSource, companionBenchTestKnowledgeSourceDirty, companionBenchTestKnowledgeSource, companionBenchTableRows, companionBenchTestResultRows, companionBenchResearchComplete, companionBenchInResearch, companionBenchInTest, companionBenchResearchProgressPct, companionBenchResearchStatusText, companionStripStatusText, companionBenchAgentSlotCount, companionBenchTestProgressRows, companionBenchRenderCards, openCompanionTestModal, startCompanionResearch, stopCompanionResearch, startCompanionTest, stopCompanionBench };
  },
}).mount("#app");
