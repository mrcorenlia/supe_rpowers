import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const videoEl = document.getElementById("inputVideo");
const canvasEl = document.getElementById("stage");
const statusEl = document.getElementById("status");
const punchSelect = document.getElementById("punchEffect");
const eyeSelect = document.getElementById("eyeEffect");
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const downloadLink = document.getElementById("downloadLink");

const ctx = canvasEl.getContext("2d");

let webcamStream = null;
let faceLandmarker = null;
let handLandmarker = null;
let latestFaceResult = null;
let latestHandResult = null;
let rafId = 0;
let lastDetectMs = 0;

let mediaRecorder = null;
let recordingStream = null;
let recordedChunks = [];
let activeRecordingUrl = null;

const handMotionState = new Map();
const punchBursts = [];
const detectIntervalMs = 34;
const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const randomBetween = (min, max) => Math.random() * (max - min) + min;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.borderColor = isError
    ? "rgba(255, 120, 120, 0.9)"
    : "rgba(255, 255, 255, 0.2)";
}

function resizeCanvasToDisplaySize() {
  const rect = canvasEl.getBoundingClientRect();
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));

  if (canvasEl.width !== width || canvasEl.height !== height) {
    canvasEl.width = width;
    canvasEl.height = height;
  }
}

function toCanvasPoint(landmark) {
  return {
    x: (1 - landmark.x) * canvasEl.width,
    y: landmark.y * canvasEl.height
  };
}

function averageLandmarks(points) {
  if (!points.length) {
    return null;
  }

  let sumX = 0;
  let sumY = 0;

  points.forEach((point) => {
    sumX += point.x;
    sumY += point.y;
  });

  const avgX = sumX / points.length;
  const avgY = sumY / points.length;

  return {
    x: (1 - avgX) * canvasEl.width,
    y: avgY * canvasEl.height
  };
}

function normalizeVector(x, y) {
  const magnitude = Math.hypot(x, y) || 1;
  return {
    x: x / magnitude,
    y: y / magnitude
  };
}

function buildEyeFrame(eyes, faceLandmarks) {
  const dx = eyes.right.x - eyes.left.x;
  const dy = eyes.right.y - eyes.left.y;
  const eyeGap = Math.max(Math.hypot(dx, dy), 1);
  const axisX = normalizeVector(dx, dy);
  let axisY = normalizeVector(-axisX.y, axisX.x);
  const eyeCenter = {
    x: (eyes.left.x + eyes.right.x) * 0.5,
    y: (eyes.left.y + eyes.right.y) * 0.5
  };

  if (faceLandmarks?.[10] && faceLandmarks?.[152]) {
    const forehead = toCanvasPoint(faceLandmarks[10]);
    const chin = toCanvasPoint(faceLandmarks[152]);
    axisY = normalizeVector(forehead.x - chin.x, forehead.y - chin.y);
  }

  // Keep the forward axis pointing upward on the screen.
  if (axisY.y > 0) {
    axisY = {
      x: -axisY.x,
      y: -axisY.y
    };
  }

  let gaze = axisY;
  if (faceLandmarks?.[1]) {
    const nose = toCanvasPoint(faceLandmarks[1]);
    const awayFromNose = normalizeVector(eyeCenter.x - nose.x, eyeCenter.y - nose.y);
    gaze = normalizeVector(axisY.x * 0.62 + awayFromNose.x * 0.38, axisY.y * 0.62 + awayFromNose.y * 0.38);
  }

  return {
    eyeGap,
    axisX,
    axisY,
    gaze,
    eyeCenter
  };
}

function getFaceBounds(faceLandmarks) {
  const idx = [10, 152, 234, 454, 127, 356];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  idx.forEach((landmarkIndex) => {
    const landmark = faceLandmarks?.[landmarkIndex];
    if (!landmark) {
      return;
    }
    const point = toCanvasPoint(landmark);
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  });

  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
    return null;
  }

  return {
    x: minX,
    y: minY,
    width: Math.max(1, maxX - minX),
    height: Math.max(1, maxY - minY)
  };
}

function getEyeCenters(faceLandmarks) {
  const leftIris = [468, 469, 470, 471, 472]
    .map((idx) => faceLandmarks[idx])
    .filter(Boolean);
  const rightIris = [473, 474, 475, 476, 477]
    .map((idx) => faceLandmarks[idx])
    .filter(Boolean);

  let left = averageLandmarks(leftIris);
  let right = averageLandmarks(rightIris);

  if (!left || !right) {
    const leftFallback = [33, 133, 159, 145]
      .map((idx) => faceLandmarks[idx])
      .filter(Boolean);
    const rightFallback = [362, 263, 386, 374]
      .map((idx) => faceLandmarks[idx])
      .filter(Boolean);

    left = averageLandmarks(leftFallback);
    right = averageLandmarks(rightFallback);
  }

  if (!left || !right) {
    return null;
  }

  return { left, right };
}

function drawMirroredVideo() {
  ctx.save();
  ctx.translate(canvasEl.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
  ctx.restore();
}

function createPunchParticles(type, intensity, direction) {
  const profiles = {
    sparks: {
      countBase: 24,
      countScale: 16,
      minSpeed: 470,
      maxSpeed: 1080,
      gravity: 1900,
      drag: 1.7,
      lifeMin: 260,
      lifeMax: 620,
      sizeMin: 1.3,
      sizeMax: 3.6,
      streakMin: 0.7,
      streakMax: 1.25,
      opacityMin: 0.55,
      opacityMax: 0.95,
      cone: Math.PI * 0.95,
      directionalChance: 0.6,
      lateral: 1.2,
      forwardPush: 18,
      colors: ["255,245,214", "255,185,84", "255,116,32", "255,72,20"]
    },
    shockwave: {
      countBase: 18,
      countScale: 11,
      minSpeed: 280,
      maxSpeed: 760,
      gravity: 1200,
      drag: 2.3,
      lifeMin: 240,
      lifeMax: 520,
      sizeMin: 1.6,
      sizeMax: 4.2,
      streakMin: 0.45,
      streakMax: 0.95,
      opacityMin: 0.3,
      opacityMax: 0.65,
      cone: Math.PI * 0.85,
      directionalChance: 0.7,
      lateral: 1.5,
      forwardPush: 35,
      colors: ["226,240,255", "176,213,255", "121,170,220", "150,161,180"]
    },
    plasma: {
      countBase: 22,
      countScale: 14,
      minSpeed: 380,
      maxSpeed: 980,
      gravity: 800,
      drag: 1.35,
      lifeMin: 220,
      lifeMax: 540,
      sizeMin: 1.2,
      sizeMax: 3.1,
      streakMin: 0.7,
      streakMax: 1.5,
      opacityMin: 0.55,
      opacityMax: 0.9,
      cone: Math.PI * 0.8,
      directionalChance: 0.72,
      lateral: 1.35,
      forwardPush: 26,
      colors: ["223,255,255", "123,246,255", "95,186,255", "70,135,255"]
    },
    debris: {
      countBase: 20,
      countScale: 12,
      minSpeed: 220,
      maxSpeed: 700,
      gravity: 2150,
      drag: 2.6,
      lifeMin: 320,
      lifeMax: 760,
      sizeMin: 1.8,
      sizeMax: 5,
      streakMin: 0.2,
      streakMax: 0.65,
      opacityMin: 0.3,
      opacityMax: 0.7,
      cone: Math.PI * 0.7,
      directionalChance: 0.78,
      lateral: 1.8,
      forwardPush: 42,
      colors: ["214,198,171", "166,147,120", "128,112,89", "92,82,65"]
    },
    metal: {
      countBase: 28,
      countScale: 16,
      minSpeed: 640,
      maxSpeed: 1400,
      gravity: 2350,
      drag: 1.9,
      lifeMin: 180,
      lifeMax: 460,
      sizeMin: 1.1,
      sizeMax: 2.8,
      streakMin: 0.85,
      streakMax: 1.6,
      opacityMin: 0.6,
      opacityMax: 0.95,
      cone: Math.PI * 0.48,
      directionalChance: 0.9,
      lateral: 1.1,
      forwardPush: 58,
      colors: ["255,252,236", "255,219,140", "255,177,82", "255,112,52"]
    },
    airblast: {
      countBase: 26,
      countScale: 16,
      minSpeed: 240,
      maxSpeed: 760,
      gravity: 760,
      drag: 2.9,
      lifeMin: 180,
      lifeMax: 420,
      sizeMin: 2.1,
      sizeMax: 5.2,
      streakMin: 0.3,
      streakMax: 0.95,
      opacityMin: 0.18,
      opacityMax: 0.48,
      cone: Math.PI * 0.36,
      directionalChance: 0.95,
      lateral: 2.4,
      forwardPush: 65,
      colors: ["234,245,255", "195,216,235", "172,188,205", "152,169,184"]
    },
    embertrail: {
      countBase: 20,
      countScale: 14,
      minSpeed: 210,
      maxSpeed: 620,
      gravity: -220,
      drag: 1.45,
      lifeMin: 340,
      lifeMax: 920,
      sizeMin: 1.6,
      sizeMax: 4.6,
      streakMin: 0.45,
      streakMax: 1.1,
      opacityMin: 0.4,
      opacityMax: 0.8,
      cone: Math.PI * 0.72,
      directionalChance: 0.82,
      lateral: 1.7,
      forwardPush: 22,
      colors: ["255,242,209", "255,176,90", "242,108,46", "170,59,26"]
    }
  };
  const profile = profiles[type] || profiles.sparks;
  const particleCount = Math.min(95, Math.round(profile.countBase + intensity * profile.countScale));
  const forward = normalizeVector(direction?.x || 0, direction?.y || -1);
  const tangent = {
    x: -forward.y,
    y: forward.x
  };
  const forwardAngle = Math.atan2(forward.y, forward.x);

  return Array.from({ length: particleCount }, () => {
    const directional = Math.random() < profile.directionalChance;
    const angle = directional
      ? forwardAngle + randomBetween(-profile.cone, profile.cone)
      : randomBetween(-Math.PI, Math.PI);
    const speed = randomBetween(profile.minSpeed, profile.maxSpeed) * (0.85 + intensity * 0.45);
    const lateralKick = randomBetween(-profile.lateral, profile.lateral) * speed * 0.08;

    return {
      vx:
        Math.cos(angle) * speed +
        tangent.x * lateralKick +
        forward.x * profile.forwardPush * intensity,
      vy:
        Math.sin(angle) * speed +
        tangent.y * lateralKick +
        forward.y * profile.forwardPush * intensity,
      gravity: profile.gravity,
      drag: profile.drag,
      lifeMs: randomBetween(profile.lifeMin, profile.lifeMax),
      delayMs: randomBetween(0, 35),
      size: randomBetween(profile.sizeMin, profile.sizeMax),
      streak: randomBetween(profile.streakMin, profile.streakMax),
      opacity: randomBetween(profile.opacityMin, profile.opacityMax),
      color: profile.colors[Math.floor(Math.random() * profile.colors.length)]
    };
  });
}

function spawnPunchEffect(point, nowMs, speed, direction) {
  const type = punchSelect.value;
  if (type === "none") {
    return;
  }

  const intensity = clamp(speed / 1800, 0.7, 2.2);
  const dir = normalizeVector(direction?.x || 0, direction?.y || -1);
  const lifeBaseByType = {
    sparks: 760,
    shockwave: 860,
    plasma: 740,
    debris: 980,
    metal: 720,
    airblast: 620,
    embertrail: 1080
  };

  punchBursts.push({
    x: point.x,
    y: point.y,
    type,
    bornAt: nowMs,
    lifeMs:
      (lifeBaseByType[type] || lifeBaseByType.sparks) *
      clamp(0.88 + intensity * 0.2, 0.88, 1.32),
    intensity,
    coreRadius: 12 + intensity * 9,
    dirX: dir.x,
    dirY: dir.y,
    normalX: -dir.y,
    normalY: dir.x,
    seed: Math.random() * Math.PI * 2,
    particles: createPunchParticles(type, intensity, dir)
  });

  if (punchBursts.length > 30) {
    punchBursts.shift();
  }
}

function detectPunches(nowMs) {
  const landmarksByHand = latestHandResult?.landmarks;
  if (!landmarksByHand?.length) {
    return;
  }

  landmarksByHand.forEach((landmarks, index) => {
    const handednessLabel =
      latestHandResult.handedness?.[index]?.[0]?.categoryName || `hand-${index}`;
    const strikePoint = landmarks[9] || landmarks[0];

    if (!strikePoint) {
      return;
    }

    const point = toCanvasPoint(strikePoint);
    const previous = handMotionState.get(handednessLabel);

    if (!previous) {
      handMotionState.set(handednessLabel, {
        x: point.x,
        y: point.y,
        t: nowMs,
        vx: 0,
        vy: 0,
        speed: 0,
        lastPunchMs: 0
      });
      return;
    }

    const dtSeconds = Math.max((nowMs - previous.t) / 1000, 0.001);
    const dx = point.x - previous.x;
    const dy = point.y - previous.y;
    const travel = Math.hypot(dx, dy);
    const rawSpeed = travel / dtSeconds;
    const speed = previous.speed * 0.58 + rawSpeed * 0.42;
    const vx = previous.vx * 0.55 + (dx / dtSeconds) * 0.45;
    const vy = previous.vy * 0.55 + (dy / dtSeconds) * 0.45;
    const cooldownElapsed = nowMs - previous.lastPunchMs;

    if (speed > 1320 && cooldownElapsed > 220) {
      spawnPunchEffect(point, nowMs, speed, { x: vx, y: vy });
      previous.lastPunchMs = nowMs;
    }

    previous.x = point.x;
    previous.y = point.y;
    previous.t = nowMs;
    previous.vx = vx;
    previous.vy = vy;
    previous.speed = speed;
    handMotionState.set(handednessLabel, previous);
  });
}

function drawImpactFlash(effect, progress, alpha) {
  const toneByType = {
    sparks: ["255,251,234", "255,176,74", "255,74,26"],
    shockwave: ["232,242,255", "140,204,255", "88,141,194"],
    plasma: ["237,255,255", "120,232,255", "67,120,255"],
    debris: ["255,228,190", "185,161,124", "97,83,62"],
    metal: ["255,253,240", "255,214,126", "255,96,38"],
    airblast: ["241,248,255", "189,214,238", "126,157,184"],
    embertrail: ["255,247,223", "255,164,88", "191,64,27"]
  };

  const [innerTone, midTone, outerTone] = toneByType[effect.type] || toneByType.sparks;
  const flashRadius = effect.coreRadius * (1 + progress * 3.6);
  const gradient = ctx.createRadialGradient(
    effect.x,
    effect.y,
    effect.coreRadius * 0.08,
    effect.x,
    effect.y,
    flashRadius
  );
  gradient.addColorStop(0, `rgba(${innerTone}, ${alpha * 0.9})`);
  gradient.addColorStop(0.42, `rgba(${midTone}, ${alpha * 0.55})`);
  gradient.addColorStop(1, `rgba(${outerTone}, 0)`);

  ctx.save();
  if (effect.type !== "debris") {
    ctx.globalCompositeOperation = "lighter";
  }
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(effect.x, effect.y, flashRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawShockwaveShell(effect, progress, alpha) {
  if (effect.type !== "shockwave") {
    return;
  }

  const radius = effect.coreRadius + progress * canvasEl.width * 0.18 * effect.intensity;
  const line = Math.max(1.1, (1 - progress) * 13 * effect.intensity);

  ctx.save();
  ctx.strokeStyle = `rgba(193, 225, 255, ${alpha * 0.45})`;
  ctx.lineWidth = line;
  ctx.beginPath();
  ctx.arc(effect.x, effect.y, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.setLineDash([line * 0.8, line * 1.2]);
  ctx.strokeStyle = `rgba(146, 190, 232, ${alpha * 0.25})`;
  ctx.lineWidth = Math.max(0.9, line * 0.55);
  ctx.beginPath();
  ctx.arc(effect.x, effect.y, radius + line * 0.75, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawPlasmaFilaments(effect, progress, alpha) {
  if (effect.type !== "plasma") {
    return;
  }

  const branches = Math.round(4 + effect.intensity * 1.5);
  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  for (let branch = 0; branch < branches; branch += 1) {
    const angleBase = effect.seed + (Math.PI * 2 * branch) / branches;
    const maxLength = effect.coreRadius * 2.8 + progress * canvasEl.width * 0.04;
    let px = effect.x;
    let py = effect.y;

    ctx.beginPath();
    ctx.moveTo(px, py);

    for (let step = 1; step <= 7; step += 1) {
      const t = step / 7;
      const bend =
        Math.sin(progress * 10 + step * 1.7 + branch * 0.9) *
        (effect.coreRadius * 0.3 * (1 - t));
      const radial = maxLength * t;
      px = effect.x + Math.cos(angleBase) * radial + Math.cos(angleBase + Math.PI / 2) * bend;
      py = effect.y + Math.sin(angleBase) * radial + Math.sin(angleBase + Math.PI / 2) * bend;
      ctx.lineTo(px, py);
    }

    ctx.strokeStyle = `rgba(125, 242, 255, ${alpha * 0.55})`;
    ctx.lineWidth = 2.2 - progress * 1.1;
    ctx.stroke();
  }

  ctx.restore();
}

function drawDebrisCloud(effect, progress, alpha) {
  if (effect.type !== "debris") {
    return;
  }

  const radius = effect.coreRadius * 1.1 + progress * canvasEl.width * 0.05;
  const gradient = ctx.createRadialGradient(
    effect.x,
    effect.y,
    effect.coreRadius * 0.2,
    effect.x,
    effect.y,
    radius
  );
  gradient.addColorStop(0, `rgba(198, 180, 146, ${alpha * 0.35})`);
  gradient.addColorStop(0.6, `rgba(144, 126, 102, ${alpha * 0.2})`);
  gradient.addColorStop(1, "rgba(80, 73, 64, 0)");

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(effect.x, effect.y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawAirBlastCone(effect, progress, alpha) {
  if (effect.type !== "airblast") {
    return;
  }

  const reach = effect.coreRadius * 2.2 + progress * canvasEl.width * 0.18 * effect.intensity;
  const nearWidth = effect.coreRadius * 0.65;
  const farWidth = effect.coreRadius * (1.5 + progress * 2.6);
  const sx = effect.x;
  const sy = effect.y;
  const tx = sx + effect.dirX * reach;
  const ty = sy + effect.dirY * reach;

  const leftNearX = sx + effect.normalX * nearWidth;
  const leftNearY = sy + effect.normalY * nearWidth;
  const rightNearX = sx - effect.normalX * nearWidth;
  const rightNearY = sy - effect.normalY * nearWidth;
  const leftFarX = tx + effect.normalX * farWidth;
  const leftFarY = ty + effect.normalY * farWidth;
  const rightFarX = tx - effect.normalX * farWidth;
  const rightFarY = ty - effect.normalY * farWidth;

  const coneGradient = ctx.createLinearGradient(sx, sy, tx, ty);
  coneGradient.addColorStop(0, `rgba(230, 242, 255, ${alpha * 0.2})`);
  coneGradient.addColorStop(0.45, `rgba(182, 207, 236, ${alpha * 0.18})`);
  coneGradient.addColorStop(1, "rgba(120, 146, 170, 0)");

  ctx.save();
  ctx.fillStyle = coneGradient;
  ctx.beginPath();
  ctx.moveTo(leftNearX, leftNearY);
  ctx.lineTo(leftFarX, leftFarY);
  ctx.lineTo(rightFarX, rightFarY);
  ctx.lineTo(rightNearX, rightNearY);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = `rgba(210, 228, 248, ${alpha * 0.32})`;
  ctx.lineWidth = Math.max(0.8, (1 - progress) * 3.2);
  ctx.beginPath();
  ctx.moveTo(leftNearX, leftNearY);
  ctx.lineTo(leftFarX, leftFarY);
  ctx.moveTo(rightNearX, rightNearY);
  ctx.lineTo(rightFarX, rightFarY);
  ctx.stroke();
  ctx.restore();
}

function drawMetalSheen(effect, progress, alpha) {
  if (effect.type !== "metal") {
    return;
  }

  const rayCount = 8;
  const baseReach = effect.coreRadius * 1.7 + progress * canvasEl.width * 0.08;

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < rayCount; i += 1) {
    const spread = (i / (rayCount - 1)) * 1.2 - 0.6;
    const dirX = effect.dirX + effect.normalX * spread;
    const dirY = effect.dirY + effect.normalY * spread;
    const dir = normalizeVector(dirX, dirY);
    const reach = baseReach * (0.7 + (i % 3) * 0.13);
    const x2 = effect.x + dir.x * reach;
    const y2 = effect.y + dir.y * reach;

    ctx.strokeStyle = `rgba(255, 224, 146, ${alpha * 0.72})`;
    ctx.lineWidth = Math.max(0.7, (1 - progress) * 2.4);
    ctx.beginPath();
    ctx.moveTo(effect.x, effect.y);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawEmberWake(effect, progress, alpha, nowMs) {
  if (effect.type !== "embertrail") {
    return;
  }

  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  for (let i = 0; i < 10; i += 1) {
    const t = i / 9;
    const trail = effect.coreRadius * 1.4 + t * canvasEl.width * 0.08;
    const drift = Math.sin(nowMs * 0.009 + i * 0.9) * effect.coreRadius * 0.45;
    const x = effect.x - effect.dirX * trail + effect.normalX * drift;
    const y = effect.y - effect.dirY * trail + effect.normalY * drift;
    const radius = Math.max(0.8, effect.coreRadius * 0.14 * (1 - t));
    const emberAlpha = alpha * (1 - t) * 0.42;

    ctx.fillStyle = `rgba(255, ${130 + i * 8}, 65, ${emberAlpha})`;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawPunchParticles(effect, elapsedMs, alphaScale) {
  const additive = ["sparks", "plasma", "metal", "embertrail"].includes(effect.type);

  ctx.save();
  if (additive) {
    ctx.globalCompositeOperation = "lighter";
  }

  effect.particles.forEach((particle) => {
    const localElapsed = elapsedMs - particle.delayMs;
    if (localElapsed <= 0 || localElapsed >= particle.lifeMs) {
      return;
    }

    const t = localElapsed / 1000;
    const progress = localElapsed / particle.lifeMs;
    const dragFactor = Math.exp(-particle.drag * t);
    const x = effect.x + particle.vx * t * dragFactor;
    const y = effect.y + particle.vy * t * dragFactor + 0.5 * particle.gravity * t * t;
    const alpha = Math.pow(1 - progress, 1.45) * particle.opacity * alphaScale;

    if (alpha <= 0.01) {
      return;
    }

    const trailLength = particle.streak * (1 - progress) * 0.06;
    const tx = x - particle.vx * trailLength;
    const ty = y - particle.vy * trailLength;

    ctx.strokeStyle = `rgba(${particle.color}, ${alpha})`;
    ctx.lineWidth = Math.max(0.75, particle.size * (1 - progress * 0.65));
    ctx.lineCap = "round";
    ctx.shadowBlur = additive ? 10 * alpha + particle.size * 2 : 0;
    ctx.shadowColor = `rgba(${particle.color}, ${alpha * 0.95})`;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(x, y);
    ctx.stroke();

    ctx.fillStyle = `rgba(${particle.color}, ${alpha * 0.88})`;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(0.6, particle.size * 0.42), 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

function drawPunchEffects(nowMs) {
  for (let i = punchBursts.length - 1; i >= 0; i -= 1) {
    const effect = punchBursts[i];
    const elapsed = nowMs - effect.bornAt;
    const progress = elapsed / effect.lifeMs;

    if (progress >= 1) {
      punchBursts.splice(i, 1);
      continue;
    }

    const alpha = Math.pow(1 - progress, 1.05);
    drawImpactFlash(effect, progress, alpha);
    drawShockwaveShell(effect, progress, alpha);
    drawPlasmaFilaments(effect, progress, alpha);
    drawDebrisCloud(effect, progress, alpha);
    drawAirBlastCone(effect, progress, alpha);
    drawMetalSheen(effect, progress, alpha);
    drawEmberWake(effect, progress, alpha, nowMs);
    drawPunchParticles(effect, elapsed, alpha);
  }
}

function drawEyeBloom(eye, radius, innerTone, midTone, outerTone, alpha) {
  const gradient = ctx.createRadialGradient(
    eye.x,
    eye.y,
    radius * 0.04,
    eye.x,
    eye.y,
    radius
  );
  gradient.addColorStop(0, `rgba(${innerTone}, ${alpha})`);
  gradient.addColorStop(0.38, `rgba(${midTone}, ${alpha * 0.82})`);
  gradient.addColorStop(1, `rgba(${outerTone}, 0)`);
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(eye.x, eye.y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawArcPath(start, end, jitterAmount, segments, tone, alpha, phase) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const normal = normalizeVector(-dy, dx);

  ctx.beginPath();
  ctx.moveTo(start.x, start.y);

  for (let i = 1; i < segments; i += 1) {
    const t = i / segments;
    const wave =
      Math.sin(t * 16 + phase + i * 1.1) * jitterAmount * (1 - Math.abs(t - 0.5) * 0.65);
    const x = start.x + dx * t + normal.x * wave;
    const y = start.y + dy * t + normal.y * wave;
    ctx.lineTo(x, y);
  }

  ctx.lineTo(end.x, end.y);
  ctx.strokeStyle = `rgba(${tone}, ${alpha})`;
  ctx.stroke();
}

function drawHeatVisionEyes(eyes, eyeFrame, nowMs) {
  const pulse = 1 + Math.sin(nowMs * 0.01) * 0.08;

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";

  [eyes.left, eyes.right].forEach((eye, index) => {
    const side = index === 0 ? -1 : 1;
    const sweep = Math.sin(nowMs * 0.006 + index * 1.7) * 0.06;
    const direction = normalizeVector(
      eyeFrame.gaze.x + eyeFrame.axisX.x * (side * 0.22 + sweep),
      eyeFrame.gaze.y + eyeFrame.axisX.y * (side * 0.22 + sweep)
    );
    const beamLength = canvasEl.height * 1.15;
    const endX = eye.x + direction.x * beamLength;
    const endY = eye.y + direction.y * beamLength;

    const plume = ctx.createLinearGradient(eye.x, eye.y, endX, endY);
    plume.addColorStop(0, "rgba(255, 246, 225, 0.84)");
    plume.addColorStop(0.14, "rgba(255, 152, 103, 0.75)");
    plume.addColorStop(1, "rgba(255, 80, 50, 0)");
    ctx.strokeStyle = plume;
    ctx.lineWidth = eyeFrame.eyeGap * 0.23 * pulse;
    ctx.shadowBlur = 18;
    ctx.shadowColor = "rgba(255, 116, 82, 0.5)";
    ctx.beginPath();
    ctx.moveTo(eye.x, eye.y);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    ctx.strokeStyle = "rgba(255, 248, 238, 0.82)";
    ctx.lineWidth = eyeFrame.eyeGap * 0.07;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.moveTo(eye.x, eye.y);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    drawEyeBloom(
      eye,
      eyeFrame.eyeGap * 0.38 * pulse,
      "255, 252, 238",
      "255, 163, 106",
      "255, 76, 46",
      0.95
    );
  });

  ctx.restore();
}

function drawIonBeamEyes(eyes, eyeFrame, nowMs) {
  const pulse = 1 + Math.sin(nowMs * 0.012) * 0.07;

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";

  [eyes.left, eyes.right].forEach((eye, index) => {
    const side = index === 0 ? -1 : 1;
    const sway = Math.sin(nowMs * 0.008 + index * 1.4) * 0.05;
    const direction = normalizeVector(
      eyeFrame.gaze.x + eyeFrame.axisX.x * (side * 0.16 + sway),
      eyeFrame.gaze.y + eyeFrame.axisX.y * (side * 0.16 + sway)
    );
    const beamLength = canvasEl.height * 1.1;
    const endX = eye.x + direction.x * beamLength;
    const endY = eye.y + direction.y * beamLength;

    const sheath = ctx.createLinearGradient(eye.x, eye.y, endX, endY);
    sheath.addColorStop(0, "rgba(236, 255, 255, 0.78)");
    sheath.addColorStop(0.22, "rgba(121, 228, 255, 0.72)");
    sheath.addColorStop(0.72, "rgba(72, 157, 255, 0.36)");
    sheath.addColorStop(1, "rgba(72, 157, 255, 0)");
    ctx.strokeStyle = sheath;
    ctx.lineWidth = eyeFrame.eyeGap * 0.14 * pulse;
    ctx.shadowBlur = 16;
    ctx.shadowColor = "rgba(82, 177, 255, 0.45)";
    ctx.beginPath();
    ctx.moveTo(eye.x, eye.y);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    ctx.strokeStyle = "rgba(240, 255, 255, 0.85)";
    ctx.lineWidth = eyeFrame.eyeGap * 0.045;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.moveTo(eye.x, eye.y);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    drawEyeBloom(
      eye,
      eyeFrame.eyeGap * 0.34 * pulse,
      "244, 255, 255",
      "149, 237, 255",
      "61, 136, 228",
      0.9
    );
  });

  ctx.restore();
}

function drawElectricArcEyes(eyes, eyeFrame, nowMs) {
  const arcWidth = clamp(eyeFrame.eyeGap * 0.045, 1.4, 4);

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";
  ctx.lineWidth = arcWidth;

  [eyes.left, eyes.right].forEach((eye, index) => {
    const side = index === 0 ? -1 : 1;
    const target = {
      x: eye.x + eyeFrame.gaze.x * (eyeFrame.eyeGap * 2.9) + eyeFrame.axisX.x * side * eyeFrame.eyeGap,
      y: eye.y + eyeFrame.gaze.y * (eyeFrame.eyeGap * 2.9) + eyeFrame.axisX.y * side * eyeFrame.eyeGap
    };
    const phase = nowMs * 0.022 + index * 2.1;
    drawArcPath(eye, target, eyeFrame.eyeGap * 0.22, 11, "148, 243, 255", 0.72, phase);
    drawArcPath(eye, target, eyeFrame.eyeGap * 0.12, 9, "232, 255, 255", 0.9, phase + 0.8);

    drawEyeBloom(
      eye,
      eyeFrame.eyeGap * 0.33,
      "239, 255, 255",
      "132, 233, 255",
      "70, 128, 220",
      0.85
    );
  });

  drawArcPath(
    eyes.left,
    eyes.right,
    eyeFrame.eyeGap * 0.16,
    12,
    "130, 229, 255",
    0.55 + Math.sin(nowMs * 0.024) * 0.15,
    nowMs * 0.03
  );

  ctx.restore();
}

function drawEnergyGlowEyes(eyes, eyeFrame, nowMs) {
  const pulse = 1 + Math.sin(nowMs * 0.009) * 0.17;

  ctx.save();
  ctx.globalCompositeOperation = "screen";

  [eyes.left, eyes.right].forEach((eye, index) => {
    const radius = eyeFrame.eyeGap * (0.34 + index * 0.015) * pulse;
    drawEyeBloom(eye, radius, "255, 255, 255", "106, 231, 255", "61, 143, 226", 0.9);

    ctx.strokeStyle = "rgba(170, 240, 255, 0.6)";
    ctx.lineWidth = Math.max(1, eyeFrame.eyeGap * 0.04);
    ctx.beginPath();
    ctx.arc(eye.x, eye.y, radius * 0.46, 0, Math.PI * 2);
    ctx.stroke();
  });

  ctx.restore();
}

function drawMoltenIrisEyes(eyes, eyeFrame, nowMs) {
  const pulse = 1 + Math.sin(nowMs * 0.011) * 0.1;

  ctx.save();
  ctx.globalCompositeOperation = "screen";

  [eyes.left, eyes.right].forEach((eye, index) => {
    const radius = eyeFrame.eyeGap * 0.31 * pulse;
    drawEyeBloom(eye, radius, "255, 250, 232", "255, 170, 82", "210, 63, 28", 0.92);

    for (let i = 0; i < 7; i += 1) {
      const cycle = (nowMs * 0.045 + i * 18 + index * 23) % 125;
      const life = 1 - cycle / 125;
      const plumeX = eye.x + Math.sin(nowMs * 0.004 + i * 1.6) * radius * 0.46;
      const plumeY = eye.y - cycle * 1.4;
      const plumeSize = Math.max(0.5, radius * 0.24 * life);
      const hueOffset = i * 10;
      const alpha = life * 0.32;

      ctx.fillStyle = `rgba(255, ${150 + hueOffset}, 68, ${alpha})`;
      ctx.beginPath();
      ctx.arc(plumeX, plumeY, plumeSize, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  ctx.restore();
}

function drawFrostGazeEyes(eyes, eyeFrame, nowMs) {
  const pulse = 1 + Math.sin(nowMs * 0.01) * 0.08;

  ctx.save();
  ctx.globalCompositeOperation = "screen";

  [eyes.left, eyes.right].forEach((eye, index) => {
    const radius = eyeFrame.eyeGap * 0.33 * pulse;
    drawEyeBloom(
      eye,
      radius,
      "247, 255, 255",
      "178, 233, 255",
      "112, 165, 212",
      0.88
    );

    for (let i = 0; i < 9; i += 1) {
      const cycle = (nowMs * 0.04 + i * 16 + index * 27) % 140;
      const life = 1 - cycle / 140;
      const drift = Math.sin(nowMs * 0.004 + i * 1.35) * radius * 0.34;
      const x = eye.x + drift;
      const y = eye.y - cycle * 1.05;
      const flakeSize = Math.max(0.5, radius * 0.19 * life);
      ctx.fillStyle = `rgba(214, 245, 255, ${life * 0.28})`;
      ctx.beginPath();
      ctx.arc(x, y, flakeSize, 0, Math.PI * 2);
      ctx.fill();
    }

    const rayCount = 6;
    for (let r = 0; r < rayCount; r += 1) {
      const angle = (Math.PI * 2 * r) / rayCount + nowMs * 0.0008;
      const reach = radius * (0.75 + (r % 2) * 0.18);
      const x2 = eye.x + Math.cos(angle) * reach;
      const y2 = eye.y + Math.sin(angle) * reach;
      ctx.strokeStyle = "rgba(203, 238, 255, 0.28)";
      ctx.lineWidth = Math.max(0.7, eyeFrame.eyeGap * 0.02);
      ctx.beginPath();
      ctx.moveTo(eye.x, eye.y);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  });

  ctx.restore();
}

function drawXRayScanEyes(eyes, eyeFrame, faceBounds, nowMs) {
  const pulse = 1 + Math.sin(nowMs * 0.012) * 0.06;

  ctx.save();
  ctx.globalCompositeOperation = "screen";

  [eyes.left, eyes.right].forEach((eye, index) => {
    const side = index === 0 ? -1 : 1;
    const direction = normalizeVector(
      eyeFrame.gaze.x + eyeFrame.axisX.x * side * 0.1,
      eyeFrame.gaze.y + eyeFrame.axisX.y * side * 0.1
    );
    const beamLength = canvasEl.height * 0.75;
    const endX = eye.x + direction.x * beamLength;
    const endY = eye.y + direction.y * beamLength;

    const beam = ctx.createLinearGradient(eye.x, eye.y, endX, endY);
    beam.addColorStop(0, "rgba(233, 255, 214, 0.75)");
    beam.addColorStop(0.25, "rgba(157, 236, 118, 0.45)");
    beam.addColorStop(1, "rgba(110, 198, 86, 0)");
    ctx.strokeStyle = beam;
    ctx.lineWidth = eyeFrame.eyeGap * 0.08 * pulse;
    ctx.beginPath();
    ctx.moveTo(eye.x, eye.y);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    drawEyeBloom(
      eye,
      eyeFrame.eyeGap * 0.28 * pulse,
      "247, 255, 240",
      "169, 239, 132",
      "94, 171, 83",
      0.8
    );
  });

  if (faceBounds) {
    const scanY = faceBounds.y + ((nowMs * 0.15) % faceBounds.height);
    const scanHeight = Math.max(16, faceBounds.height * 0.09);

    const scanGradient = ctx.createLinearGradient(0, scanY - scanHeight, 0, scanY + scanHeight);
    scanGradient.addColorStop(0, "rgba(140, 222, 120, 0)");
    scanGradient.addColorStop(0.5, "rgba(160, 245, 136, 0.22)");
    scanGradient.addColorStop(1, "rgba(140, 222, 120, 0)");

    ctx.save();
    ctx.beginPath();
    ctx.rect(faceBounds.x, faceBounds.y, faceBounds.width, faceBounds.height);
    ctx.clip();
    ctx.fillStyle = scanGradient;
    ctx.fillRect(faceBounds.x, scanY - scanHeight, faceBounds.width, scanHeight * 2);

    const lineSpacing = Math.max(8, faceBounds.height * 0.04);
    for (let y = faceBounds.y; y <= faceBounds.y + faceBounds.height; y += lineSpacing) {
      ctx.strokeStyle = "rgba(138, 220, 114, 0.08)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(faceBounds.x, y + (nowMs * 0.02) % lineSpacing);
      ctx.lineTo(faceBounds.x + faceBounds.width, y + (nowMs * 0.02) % lineSpacing);
      ctx.stroke();
    }
    ctx.restore();

    ctx.strokeStyle = "rgba(146, 228, 120, 0.25)";
    ctx.lineWidth = Math.max(1, faceBounds.width * 0.005);
    ctx.strokeRect(faceBounds.x, faceBounds.y, faceBounds.width, faceBounds.height);
  }

  ctx.restore();
}

function drawEyeEffects(nowMs) {
  if (eyeSelect.value === "none") {
    return;
  }

  const faceLandmarks = latestFaceResult?.faceLandmarks?.[0];
  if (!faceLandmarks) {
    return;
  }

  const eyes = getEyeCenters(faceLandmarks);
  if (!eyes) {
    return;
  }

  const eyeFrame = buildEyeFrame(eyes, faceLandmarks);
  const faceBounds = getFaceBounds(faceLandmarks);
  const effect = eyeSelect.value;
  if (effect === "heat") {
    drawHeatVisionEyes(eyes, eyeFrame, nowMs);
  } else if (effect === "ion") {
    drawIonBeamEyes(eyes, eyeFrame, nowMs);
  } else if (effect === "arc") {
    drawElectricArcEyes(eyes, eyeFrame, nowMs);
  } else if (effect === "glow") {
    drawEnergyGlowEyes(eyes, eyeFrame, nowMs);
  } else if (effect === "ember") {
    drawMoltenIrisEyes(eyes, eyeFrame, nowMs);
  } else if (effect === "frost") {
    drawFrostGazeEyes(eyes, eyeFrame, nowMs);
  } else if (effect === "xray") {
    drawXRayScanEyes(eyes, eyeFrame, faceBounds, nowMs);
  }
}

function pickMimeType() {
  const options = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm"
  ];
  return options.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function resetDownloadLink() {
  if (activeRecordingUrl) {
    URL.revokeObjectURL(activeRecordingUrl);
    activeRecordingUrl = null;
  }
  downloadLink.hidden = true;
  downloadLink.removeAttribute("href");
}

function stopRecordingTracks() {
  if (!recordingStream) {
    return;
  }

  recordingStream.getTracks().forEach((track) => {
    track.stop();
  });
  recordingStream = null;
}

function finalizeRecording() {
  if (!recordedChunks.length) {
    setStatus("No recording data captured.", true);
    stopRecordingTracks();
    return;
  }

  const blobType = mediaRecorder?.mimeType || "video/webm";
  const blob = new Blob(recordedChunks, { type: blobType });
  resetDownloadLink();
  activeRecordingUrl = URL.createObjectURL(blob);
  downloadLink.href = activeRecordingUrl;
  downloadLink.hidden = false;

  setStatus("Recording ready. Download video.");
  stopRecordingTracks();
}

function startRecording() {
  if (!webcamStream) {
    setStatus("Camera is not ready yet.", true);
    return;
  }

  resetDownloadLink();
  recordedChunks = [];

  const stream = canvasEl.captureStream(30);
  webcamStream.getAudioTracks().forEach((track) => {
    stream.addTrack(track.clone());
  });
  recordingStream = stream;

  try {
    const mimeType = pickMimeType();
    mediaRecorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);
  } catch (error) {
    setStatus(`Recording is not supported: ${error.message}`, true);
    stopRecordingTracks();
    return;
  }

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  mediaRecorder.onstop = finalizeRecording;

  mediaRecorder.start(160);
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("Recording...");
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    return;
  }

  stopBtn.disabled = true;
  setStatus("Finalizing video...");
  mediaRecorder.stop();
  recordBtn.disabled = false;
}

function runDetection(nowMs) {
  if (!faceLandmarker || !handLandmarker) {
    return;
  }

  latestFaceResult = faceLandmarker.detectForVideo(videoEl, nowMs);
  latestHandResult = handLandmarker.detectForVideo(videoEl, nowMs);
  detectPunches(nowMs);
}

function renderFrame(nowMs) {
  resizeCanvasToDisplaySize();

  if (videoEl.readyState >= 2) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    drawMirroredVideo();

    if (nowMs - lastDetectMs >= detectIntervalMs) {
      lastDetectMs = nowMs;
      runDetection(nowMs);
    }

    drawEyeEffects(nowMs);
    drawPunchEffects(nowMs);
  }

  rafId = requestAnimationFrame(renderFrame);
}

async function initTrackers() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    },
    runningMode: "VIDEO",
    numFaces: 1
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
}

async function initCamera() {
  webcamStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user"
    },
    audio: {
      echoCancellation: true,
      noiseSuppression: true
    }
  });

  videoEl.srcObject = webcamStream;
  await videoEl.play();
}

async function bootstrap() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera API unavailable in this browser.", true);
    return;
  }

  setStatus("Loading models and camera...");

  try {
    await Promise.all([initTrackers(), initCamera()]);
    setStatus("Live. Throw punches and pick your effects.");
    recordBtn.disabled = false;
    rafId = requestAnimationFrame(renderFrame);
  } catch (error) {
    setStatus(`Startup failed: ${error.message}`, true);
  }
}

recordBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);

window.addEventListener("resize", resizeCanvasToDisplaySize);
window.addEventListener("beforeunload", () => {
  cancelAnimationFrame(rafId);
  stopRecordingTracks();
  resetDownloadLink();
  webcamStream?.getTracks().forEach((track) => track.stop());
});

bootstrap();
