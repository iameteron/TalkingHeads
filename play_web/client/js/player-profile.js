const PLAYER_PROFILE_STORAGE_KEY = "playWebPlayerProfile";
const AVATAR_COUNT = 10;

const AVATAR_OPTIONS = [
  { id: 0, label: "Spiky blue" },
  { id: 1, label: "Green ponytail" },
  { id: 2, label: "Helmet pilot" },
  { id: 3, label: "Grey veteran" },
  { id: 4, label: "Purple bun" },
  { id: 5, label: "Hooded" },
  { id: 6, label: "Pink hair" },
  { id: 7, label: "Glasses" },
  { id: 8, label: "Heavy soldier" },
  { id: 9, label: "Blue bob" },
];

function defaultPlayerProfile() {
  return { nickname: "", avatar_id: 0 };
}

function normalizeAvatarId(value) {
  const id = Number.parseInt(String(value ?? "0"), 10);
  if (!Number.isFinite(id) || id < 0 || id >= AVATAR_COUNT) return 0;
  return id;
}

function normalizeNickname(value) {
  return String(value || "").trim().slice(0, 40);
}

function loadPlayerProfile() {
  try {
    const raw = localStorage.getItem(PLAYER_PROFILE_STORAGE_KEY);
    if (!raw) return defaultPlayerProfile();
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return defaultPlayerProfile();
    return {
      nickname: normalizeNickname(parsed.nickname),
      avatar_id: normalizeAvatarId(parsed.avatar_id),
    };
  } catch (_err) {
    return defaultPlayerProfile();
  }
}

function savePlayerProfile(profile) {
  const normalized = {
    nickname: normalizeNickname(profile?.nickname),
    avatar_id: normalizeAvatarId(profile?.avatar_id),
  };
  try {
    localStorage.setItem(PLAYER_PROFILE_STORAGE_KEY, JSON.stringify(normalized));
  } catch (_err) {}
  return normalized;
}

function avatarSrcFor(id) {
  const avatarId = normalizeAvatarId(id);
  return `./assets/avatars/avatar-${avatarId}.png`;
}

function isPlayerProfileComplete(profile) {
  return normalizeNickname(profile?.nickname).length > 0;
}
