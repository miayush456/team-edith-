/* ============================================================
   SYNAPSE 2.5D Race Renderer — app.js
   Full Implementation: Sunlit F1 Grand Prix Broadcast
   - Beautiful Daytime Lighting & Azure Sky (No pitch black!)
   - Clear Track Perimeter: All light poles & gantry pillars strictly outside barriers
   - Authentic F1 Circuit Materials: Grey tarmac, gravel traps, green turf, vibrant apex curbs
   - High-End F1 Car Models with Pearlescent Clearcoat & Pirelli tires
   - Dynamic Overlap Transparency & Holographic Wireframe Matrix
   - Dynamic 3rd-Person Chase Cameras (Battle, Car A, Car B, Cockpit, Top)
   - Real-Time Broadcast HUD, Incident Alerts, G-Force & Minimap Frustum
   ============================================================ */

import * as THREE from 'three';
import { connectLive, parseLiveMessage } from './live_source.js';

// ============================================================
// PHASE 1: Data Ingestion & Interpolation Engine
// ============================================================

/** Parse and validate a Schema 1.1 replay JSON. */
function parseReplay(data) {
  if (data.schema_version !== '1.1' || data.kind !== 'racing_event_replay') {
    throw new Error('Unsupported replay format: expected schema 1.1, racing_event_replay');
  }
  return {
    participantId: data.source?.participant_id || 'Unknown Racer',
    tickHz: data.tick_hz || 30,
    car: data.car || { length_m: 4.8, width_m: 2.0 },
    track: data.track,
    frames: data.frames,
    result: data.result || {},
  };
}

/** Shortest-angle linear interpolation for heading (prevents wrapping jumps). */
function lerpAngle(a, b, t) {
  let diff = b - a;
  diff = diff - Math.round(diff / (2 * Math.PI)) * 2 * Math.PI;
  return a + diff * t;
}

/** Sample a replay frame at an arbitrary simulation time (seconds). */
function sampleAtTime(frames, tickHz, timeS) {
  if (!frames || frames.length === 0) return null;
  const lastTick = frames.length - 1;
  const duration = lastTick / tickHz;
  const clampedTime = Math.max(0, Math.min(timeS, duration));

  let pos = clampedTime * tickHz;
  const nearest = Math.round(pos);
  if (Math.abs(pos - nearest) < 1e-9) pos = nearest;

  const tickA = Math.min(Math.floor(pos), lastTick);
  const tickB = Math.min(tickA + 1, lastTick);
  const frac = tickA === tickB ? 0 : pos - tickA;

  const a = frames[tickA];
  const b = frames[tickB];

  return {
    x: a.x + (b.x - a.x) * frac,
    y: a.y + (b.y - a.y) * frac,
    heading: lerpAngle(a.heading, b.heading, frac),
    speed: a.speed + (b.speed - a.speed) * frac,
    steer: a.steer + (b.steer - a.steer) * frac,
    throttle_brake: a.throttle_brake + (b.throttle_brake - a.throttle_brake) * frac,
    yaw_rate: a.yaw_rate + (b.yaw_rate - a.yaw_rate) * frac,
    lateral_velocity: a.lateral_velocity + (b.lateral_velocity - a.lateral_velocity) * frac,
    off_track: a.off_track,
    events: a.events || [],
    terminated: a.terminated || false,
    truncated: a.truncated || false,
    tick: tickA,
    finished: clampedTime >= duration,
  };
}

// ============================================================
// PHASE 2: 3D Scene, Track Mesh & Sunlit F1 Environment
// ============================================================

/** Build a closed ribbon mesh between two polylines. */
function buildRibbonMesh(polyA, polyB, material, y_offset = 0) {
  const n = polyA.length;
  const positions = [];
  const indices = [];

  for (let i = 0; i < n; i++) {
    positions.push(polyA[i][0], y_offset, -polyA[i][1]);
    positions.push(polyB[i][0], y_offset, -polyB[i][1]);
  }

  for (let i = 0; i < n; i++) {
    const next = (i + 1) % n;
    const a = i * 2;
    const b = i * 2 + 1;
    const c = next * 2;
    const d = next * 2 + 1;
    indices.push(a, c, b);
    indices.push(b, c, d);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const mesh = new THREE.Mesh(geometry, material);
  mesh.receiveShadow = true;
  return mesh;
}

/** Build vertical barrier walls from a polyline with colored sponsor banners. */
function buildBarrierWall(poly, height, baseColor = 0xf1f5f9) {
  const group = new THREE.Group();
  const n = poly.length;
  const positions = [];
  const indices = [];

  for (let i = 0; i < n; i++) {
    positions.push(poly[i][0], 0, -poly[i][1]);
    positions.push(poly[i][0], height, -poly[i][1]);
  }

  for (let i = 0; i < n; i++) {
    const next = (i + 1) % n;
    const bl = i * 2;
    const tl = i * 2 + 1;
    const br = next * 2;
    const tr = next * 2 + 1;
    indices.push(bl, br, tl);
    indices.push(tl, br, tr);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: baseColor,
    roughness: 0.35,
    metalness: 0.15,
    side: THREE.DoubleSide,
  });

  const wallMesh = new THREE.Mesh(geometry, mat);
  wallMesh.castShadow = true;
  wallMesh.receiveShadow = true;
  group.add(wallMesh);

  // Top blue safety rail
  const railMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.3, metalness: 0.4 });
  const railPoints = poly.map(p => new THREE.Vector3(p[0], height + 0.05, -p[1]));
  railPoints.push(railPoints[0].clone());
  const railGeo = new THREE.BufferGeometry().setFromPoints(railPoints);
  const railLine = new THREE.Line(railGeo, new THREE.LineBasicMaterial({ color: 0x0284c7, linewidth: 3 }));
  group.add(railLine);

  return group;
}

/** Build sharp white track edge lines. */
function buildEdgeLine(poly, color = 0xffffff, opacity = 0.95) {
  const points = poly.map(p => new THREE.Vector3(p[0], 0.035, -p[1]));
  points.push(points[0].clone());
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: color,
    opacity: opacity,
    transparent: true,
  });
  return new THREE.Line(geometry, material);
}

/** Build 3D curbs at corner apexes with vivid red & white teeth. */
function buildCurbs(track) {
  const group = new THREE.Group();
  const curvature = track.curvature;
  const left = track.left_boundary;
  const right = track.right_boundary;
  const n = left.length;
  const CURB_WIDTH = 0.85;
  const CURB_THRESHOLD = 0.012;

  const redMat = new THREE.MeshStandardMaterial({ color: 0xe10600, roughness: 0.4, metalness: 0.1 });
  const whiteMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4, metalness: 0.1 });

  for (let i = 0; i < n; i++) {
    if (Math.abs(curvature[i]) < CURB_THRESHOLD) continue;

    const next = (i + 1) % n;
    const isLeftCurb = curvature[i] < 0;
    const boundary = isLeftCurb ? left : right;

    const bCurr = boundary[i];
    const bNext = boundary[next];
    const dx = bNext[0] - bCurr[0];
    const dy = bNext[1] - bCurr[1];
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.001) continue;

    const nx = isLeftCurb ? dy / len : -dy / len;
    const ny = isLeftCurb ? -dx / len : dx / len;

    const isRed = Math.floor(i / 2) % 2 === 0;
    const curbMat = isRed ? redMat : whiteMat;

    const p0 = [bCurr[0], bCurr[1]];
    const p1 = [bNext[0], bNext[1]];
    const p2 = [bNext[0] + nx * CURB_WIDTH, bNext[1] + ny * CURB_WIDTH];
    const p3 = [bCurr[0] + nx * CURB_WIDTH, bCurr[1] + ny * CURB_WIDTH];

    const positions = [
      p0[0], 0.03, -p0[1],
      p1[0], 0.03, -p1[1],
      p2[0], 0.03, -p2[1],
      p3[0], 0.03, -p3[1],
    ];

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.computeVertexNormals();

    const mesh = new THREE.Mesh(geo, curbMat);
    mesh.receiveShadow = true;
    group.add(mesh);
  }
  return group;
}

/** Build start / finish checkered grid line and high overhead gantry (100% outside barriers). */
function buildStartFinishStructure(track, trackCenter) {
  const group = new THREE.Group();
  const idx = 0;
  const left = track.left_boundary[idx];
  const right = track.right_boundary[idx];
  const bLeft = track.barrier_left[idx];
  const bRight = track.barrier_right[idx];

  const dx = right[0] - left[0];
  const dy = right[1] - left[1];
  const roadWidth = Math.hypot(dx, dy);
  const segments = 12;
  const lineWidth = 2.0;

  const next = 1;
  const tdx = track.centerline[next][0] - track.centerline[idx][0];
  const tdy = track.centerline[next][1] - track.centerline[idx][1];
  const tLen = Math.hypot(tdx, tdy) || 1;
  const fwdX = tdx / tLen;
  const fwdY = tdy / tLen;

  // 1. Checkered Finish Line on Asphalt
  for (let s = 0; s < segments; s++) {
    const t0 = s / segments;
    const t1 = (s + 1) / segments;
    const isWhite = s % 2 === 0;

    const lx0 = left[0] + dx * t0;
    const ly0 = left[1] + dy * t0;
    const lx1 = left[0] + dx * t1;
    const ly1 = left[1] + dy * t1;

    const positions = [
      lx0 - fwdX * lineWidth / 2, 0.035, -(ly0 - fwdY * lineWidth / 2),
      lx1 - fwdX * lineWidth / 2, 0.035, -(ly1 - fwdY * lineWidth / 2),
      lx1 + fwdX * lineWidth / 2, 0.035, -(ly1 + fwdY * lineWidth / 2),
      lx0 + fwdX * lineWidth / 2, 0.035, -(ly0 + fwdY * lineWidth / 2),
    ];

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      color: isWhite ? 0xffffff : 0x222630,
      roughness: 0.4,
      metalness: 0.1,
    });
    group.add(new THREE.Mesh(geo, mat));
  }

  // 2. F1 Grid Slot Boxes on the Road (Slot 1 and Slot 2)
  const slotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const yellowSlotMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });

  for (const [slotNum, sideOffset] of [[1, -1.35], [2, 1.35]]) {
    const slotX = track.centerline[idx][0] - (slotNum === 2 ? fwdX * 8 : 0);
    const slotY = track.centerline[idx][0] - (slotNum === 2 ? fwdY * 8 : 0);
    const perpX = -fwdY * sideOffset;
    const perpY = fwdX * sideOffset;

    // Painted slot outline box
    const boxGeo = new THREE.PlaneGeometry(2.4, 4.6);
    const boxMesh = new THREE.Mesh(
      boxGeo,
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true })
    );
    boxMesh.rotation.x = -Math.PI / 2;
    boxMesh.rotation.z = Math.atan2(fwdX, fwdY);
    boxMesh.position.set(slotX + perpX, 0.032, -(slotY + perpY));
    group.add(boxMesh);
  }

  // 3. Overhead Gantry Structure
  // Pillars are placed 3.5m OUTSIDE the barrier walls — completely off the track & runoffs!
  const dirLx = bLeft[0] - trackCenter.x;
  const dirLy = bLeft[1] - trackCenter.y;
  const lenL = Math.hypot(dirLx, dirLy) || 1;
  const pillarLeftPos = new THREE.Vector3(bLeft[0] + (dirLx / lenL) * 3.5, 0, -(bLeft[1] + (dirLy / lenL) * 3.5));

  const dirRx = bRight[0] - trackCenter.x;
  const dirRy = bRight[1] - trackCenter.y;
  const lenR = Math.hypot(dirRx, dirRy) || 1;
  const pillarRightPos = new THREE.Vector3(bRight[0] + (dirRx / lenR) * 3.5, 0, -(bRight[1] + (dirRy / lenR) * 3.5));

  const gantryHeight = 9.5; // High clearance over the entire circuit
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8, roughness: 0.3 });
  const trussMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.7, roughness: 0.4 });

  // Left Gantry Tower
  const towerLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.7, gantryHeight, 8), pillarMat);
  towerLeft.position.set(pillarLeftPos.x, gantryHeight / 2, pillarLeftPos.z);
  towerLeft.castShadow = true;
  group.add(towerLeft);

  // Right Gantry Tower
  const towerRight = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.7, gantryHeight, 8), pillarMat);
  towerRight.position.set(pillarRightPos.x, gantryHeight / 2, pillarRightPos.z);
  towerRight.castShadow = true;
  group.add(towerRight);

  // Overhead Crossbeam High in the Air
  const span = pillarLeftPos.distanceTo(pillarRightPos);
  const midPos = pillarLeftPos.clone().add(pillarRightPos).multiplyScalar(0.5);
  midPos.y = gantryHeight - 0.5;

  const crossBeam = new THREE.Mesh(new THREE.BoxGeometry(span, 1.2, 1.8), trussMat);
  crossBeam.position.copy(midPos);
  const bridgeAngle = Math.atan2(pillarRightPos.z - pillarLeftPos.z, pillarRightPos.x - pillarLeftPos.x);
  crossBeam.rotation.y = -bridgeAngle;
  crossBeam.castShadow = true;
  group.add(crossBeam);

  // 5 Red Starting Lights Mounted on Crossbeam High Above
  const lightGeo = new THREE.SphereGeometry(0.28, 16, 16);
  const lightMat = new THREE.MeshBasicMaterial({ color: 0xff0022 });
  const lightHousingMat = new THREE.MeshStandardMaterial({ color: 0x0f172a });

  const housing = new THREE.Mesh(new THREE.BoxGeometry(6.5, 0.9, 0.6), lightHousingMat);
  housing.position.set(midPos.x, midPos.y - 0.7, midPos.z);
  housing.rotation.y = -bridgeAngle;
  group.add(housing);

  for (let l = -2; l <= 2; l++) {
    const light = new THREE.Mesh(lightGeo, lightMat);
    light.position.set(midPos.x + Math.cos(-bridgeAngle) * l * 1.2, midPos.y - 0.7, midPos.z - Math.sin(-bridgeAngle) * l * 1.2);
    group.add(light);
  }

  return group;
}

/** Build stadium perimeter light towers — STRICTLY ON THE OUTSIDE PERIMETER. */
function buildStadiumLights(track, trackCenter) {
  const group = new THREE.Group();
  const n = track.centerline.length;
  const step = Math.max(1, Math.floor(n / 8));

  const poleMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.8, roughness: 0.2 });
  const lampMat = new THREE.MeshBasicMaterial({ color: 0xfffbeb });

  for (let i = 0; i < n; i += step) {
    const bL = track.barrier_left[i];
    const bR = track.barrier_right[i];

    // Determine which barrier is the outer perimeter (further from track center)
    const distL = Math.hypot(bL[0] - trackCenter.x, bL[1] - trackCenter.y);
    const distR = Math.hypot(bR[0] - trackCenter.x, bR[1] - trackCenter.y);
    const outerBarrier = distL > distR ? bL : bR;

    // Vector pointing strictly OUTWARD away from the track
    const outDx = outerBarrier[0] - trackCenter.x;
    const outDy = outerBarrier[1] - trackCenter.y;
    const outLen = Math.hypot(outDx, outDy) || 1;
    const outNormX = outDx / outLen;
    const outNormY = outDy / outLen;

    // Place pole 18 METERS OUTSIDE the outer barrier on the grass!
    const poleX = outerBarrier[0] + outNormX * 18;
    const poleZ = -(outerBarrier[1] + outNormY * 18);
    const poleHeight = 22;

    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.6, poleHeight, 8), poleMat);
    pole.position.set(poleX, poleHeight / 2, poleZ);
    pole.castShadow = true;
    group.add(pole);

    const head = new THREE.Mesh(new THREE.BoxGeometry(4.2, 1.4, 2.0), poleMat);
    head.position.set(poleX, poleHeight, poleZ);
    group.add(head);

    const bulb = new THREE.Mesh(new THREE.BoxGeometry(3.6, 1.0, 0.4), lampMat);
    bulb.position.set(poleX, poleHeight, poleZ + 1.0);
    group.add(bulb);
  }
  return group;
}

/** Build spectator grandstands — strictly placed outside the outer barriers. */
function buildGrandstands(track, trackCenter) {
  const group = new THREE.Group();
  const n = track.centerline.length;
  const count = Math.min(20, Math.floor(n / 4));

  const standMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.5, metalness: 0.3 });
  const roofMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.3, metalness: 0.5 });
  const seatMatRed = new THREE.MeshStandardMaterial({ color: 0xe10600 });
  const seatMatBlue = new THREE.MeshStandardMaterial({ color: 0x2563eb });

  for (let i = 2; i < count; i += 6) {
    const bL = track.barrier_left[i];
    const bR = track.barrier_right[i];

    const distL = Math.hypot(bL[0] - trackCenter.x, bL[1] - trackCenter.y);
    const distR = Math.hypot(bR[0] - trackCenter.x, bR[1] - trackCenter.y);
    const outerP = distL > distR ? bL : bR;

    const outDx = outerP[0] - trackCenter.x;
    const outDy = outerP[1] - trackCenter.y;
    const outLen = Math.hypot(outDx, outDy) || 1;

    // Placed 14m outside the outer barrier
    const posX = outerP[0] + (outDx / outLen) * 14;
    const posZ = -(outerP[1] + (outDy / outLen) * 14);

    const tier = new THREE.Mesh(new THREE.BoxGeometry(14, 4.5, 7), standMat);
    tier.position.set(posX, 2.25, posZ);
    tier.castShadow = true;
    tier.receiveShadow = true;
    group.add(tier);

    const roof = new THREE.Mesh(new THREE.BoxGeometry(15, 0.4, 9), roofMat);
    roof.position.set(posX, 5.5, posZ);
    roof.castShadow = true;
    group.add(roof);

    // Multi-colored crowd seats
    for (let row = 0; row < 3; row++) {
      const seats = new THREE.Mesh(new THREE.BoxGeometry(13.5, 0.3, 1.2), (row % 2 === 0) ? seatMatRed : seatMatBlue);
      seats.position.set(posX, 1.5 + row * 0.9, posZ - 1.5 + row * 1.5);
      group.add(seats);
    }
  }
  return group;
}

/** Build the complete, sunlit 3D track environment. */
function buildTrackScene(track) {
  const group = new THREE.Group();

  // Compute track center
  let cx = 0, cy = 0;
  for (const p of track.centerline) { cx += p[0]; cy += p[1]; }
  cx /= track.centerline.length;
  cy /= track.centerline.length;
  const trackCenter = { x: cx, y: cy };

  // 1. Asphalt Road Surface (Realistic grey racing tarmac)
  const roadMat = new THREE.MeshStandardMaterial({
    color: 0x363b46,
    roughness: 0.72,
    metalness: 0.12,
    side: THREE.DoubleSide,
  });
  group.add(buildRibbonMesh(track.left_boundary, track.right_boundary, roadMat, 0.015));

  // 2. Runoff Areas:
  // Sandy Golden Gravel Traps & F1 Royal Blue Runoff
  const gravelMat = new THREE.MeshStandardMaterial({
    color: 0xd9b372,
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  const blueRunoffMat = new THREE.MeshStandardMaterial({
    color: 0x1e3a8a,
    roughness: 0.75,
    metalness: 0.1,
    side: THREE.DoubleSide,
  });

  // Assign gravel to outer runoff, blue strip to inner runoff
  group.add(buildRibbonMesh(track.barrier_left, track.left_boundary, blueRunoffMat, 0.008));
  group.add(buildRibbonMesh(track.right_boundary, track.barrier_right, gravelMat, 0.008));

  // 3. Crisp Track Boundary Edge Lines
  group.add(buildEdgeLine(track.left_boundary, 0xffffff, 0.95));
  group.add(buildEdgeLine(track.right_boundary, 0xffffff, 0.95));

  // 4. Dashed Centerline
  const centerPoints = track.centerline.map(p => new THREE.Vector3(p[0], 0.025, -p[1]));
  centerPoints.push(centerPoints[0].clone());
  const centerGeo = new THREE.BufferGeometry().setFromPoints(centerPoints);
  const centerLine = new THREE.Line(
    centerGeo,
    new THREE.LineDashedMaterial({ color: 0xffffff, opacity: 0.45, transparent: true, dashSize: 3.0, gapSize: 3.5 })
  );
  centerLine.computeLineDistances();
  group.add(centerLine);

  // 5. Clean White/Silver Safety Barrier Walls
  const barrierHeight = 1.35;
  group.add(buildBarrierWall(track.barrier_left, barrierHeight, 0xf8fafc));
  group.add(buildBarrierWall(track.barrier_right, barrierHeight, 0xf8fafc));

  // 6. Vibrant Apex Curbs (Red & White)
  group.add(buildCurbs(track));

  // 7. Start / Finish Gantry (100% outside barriers)
  group.add(buildStartFinishStructure(track, trackCenter));

  // 8. Stadium Perimeter Lights (100% on the outer perimeter)
  group.add(buildStadiumLights(track, trackCenter));

  // 9. Spectator Grandstands
  group.add(buildGrandstands(track, trackCenter));

  return group;
}

// ============================================================
// PHASE 3: High-Detail F1 Vehicle Models & Overlap Wireframe
// ============================================================

/** Build an F1 car model with clearcoat livery, Pirelli tires, and holographic overlap wireframe. */
function buildF1CarModel(liveryColor, accentColor, carNumber = '1') {
  const carGroup = new THREE.Group();
  const L = 4.8;
  const W = 2.0;

  const chassisGroup = new THREE.Group();
  carGroup.add(chassisGroup);

  const bodyMaterials = [];
  const edgeMaterials = [];
  const wireMaterials = [];

  function makeBodyMat(color, roughness = 0.2, metalness = 0.8) {
    const m = new THREE.MeshPhysicalMaterial({
      color,
      roughness,
      metalness,
      clearcoat: 0.9,
      clearcoatRoughness: 0.1,
      transparent: true,
      opacity: 1.0,
    });
    bodyMaterials.push(m);
    return m;
  }

  const liveryMat = makeBodyMat(liveryColor, 0.15, 0.85);
  const carbonMat = makeBodyMat(0x0f172a, 0.4, 0.6);
  const accentMat = makeBodyMat(accentColor, 0.2, 0.7);
  const haloMat = makeBodyMat(0x1e293b, 0.3, 0.9);
  const visorMat = makeBodyMat(0x05070a, 0.05, 0.95);

  const tireMat = new THREE.MeshStandardMaterial({ color: 0x171717, roughness: 0.85, metalness: 0.05 });
  const sidewallMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 }); // Pirelli Yellow Soft stripe
  const rimMat = new THREE.MeshStandardMaterial({ color: liveryColor, roughness: 0.2, metalness: 0.9 });

  function addChassisComponent(geo, mat, pos, rot = null) {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    if (rot) mesh.rotation.copy(rot);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    chassisGroup.add(mesh);

    // Glowing Edge Outlines for Overlap
    const edgeGeo = new THREE.EdgesGeometry(geo);
    const edgeMat = new THREE.LineBasicMaterial({
      color: liveryColor,
      transparent: true,
      opacity: 0.0,
      linewidth: 2,
    });
    edgeMaterials.push(edgeMat);
    const edgeLine = new THREE.LineSegments(edgeGeo, edgeMat);
    edgeLine.position.copy(pos);
    if (rot) edgeLine.rotation.copy(rot);
    chassisGroup.add(edgeLine);

    // Wireframe Matrix for X-Ray Mode
    const wireMat = new THREE.MeshBasicMaterial({
      color: liveryColor,
      wireframe: true,
      transparent: true,
      opacity: 0.0,
    });
    wireMaterials.push(wireMat);
    const wireMesh = new THREE.Mesh(geo, wireMat);
    wireMesh.position.copy(pos);
    if (rot) wireMesh.rotation.copy(rot);
    chassisGroup.add(wireMesh);

    return mesh;
  }

  // 1. Aerodynamic Nose Cone (Tapers forward to +Z)
  const noseShape = new THREE.Shape();
  noseShape.moveTo(-0.28, 0);
  noseShape.lineTo(0.28, 0);
  noseShape.lineTo(0.12, L * 0.44);
  noseShape.lineTo(-0.12, L * 0.44);
  noseShape.closePath();
  const noseExtrude = { depth: 0.32, bevelEnabled: true, bevelThickness: 0.04, bevelSize: 0.04 };
  const noseGeo = new THREE.ExtrudeGeometry(noseShape, noseExtrude);
  noseGeo.rotateX(-Math.PI / 2);
  addChassisComponent(noseGeo, liveryMat, new THREE.Vector3(0, 0.16, 0));

  // Nose tip camera / carbon nose cone
  const noseTip = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.15, 0.3), carbonMat);
  noseTip.position.set(0, 0.22, L * 0.44 + 0.1);
  chassisGroup.add(noseTip);

  // 2. Cockpit Tub & Monocoque
  const tubGeo = new THREE.BoxGeometry(0.78, 0.48, L * 0.44);
  addChassisComponent(tubGeo, carbonMat, new THREE.Vector3(0, 0.36, -L * 0.02));

  // 3. Sculpted Sidepods with dark intake vents
  const sidepodGeo = new THREE.BoxGeometry(0.42, 0.42, L * 0.42);
  addChassisComponent(sidepodGeo, liveryMat, new THREE.Vector3(-0.55, 0.34, -L * 0.06));
  addChassisComponent(sidepodGeo, liveryMat, new THREE.Vector3(0.55, 0.34, -L * 0.06));

  const intakeGeo = new THREE.PlaneGeometry(0.36, 0.34);
  const intakeMat = new THREE.MeshBasicMaterial({ color: 0x090d16 });
  const leftIntake = new THREE.Mesh(intakeGeo, intakeMat);
  leftIntake.position.set(-0.55, 0.34, -L * 0.06 + L * 0.21 + 0.01);
  chassisGroup.add(leftIntake);

  const rightIntake = new THREE.Mesh(intakeGeo, intakeMat);
  rightIntake.position.set(0.55, 0.34, -L * 0.06 + L * 0.21 + 0.01);
  chassisGroup.add(rightIntake);

  // 4. Engine Airbox & Shark Fin
  const airbox = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.35, 0.6), carbonMat);
  airbox.position.set(0, 0.72, -0.2);
  chassisGroup.add(airbox);

  const finShape = new THREE.Shape();
  finShape.moveTo(0, 0);
  finShape.lineTo(0, L * 0.34);
  finShape.lineTo(0.38, 0);
  finShape.closePath();
  const finGeo = new THREE.ExtrudeGeometry(finShape, { depth: 0.04, bevelEnabled: false });
  finGeo.rotateY(Math.PI / 2);
  addChassisComponent(finGeo, accentMat, new THREE.Vector3(0.02, 0.58, -L * 0.4));

  // 5. Cockpit Halo & Driver Helmet
  const haloGeo = new THREE.TorusGeometry(0.25, 0.04, 8, 16, Math.PI);
  const haloEuler = new THREE.Euler(Math.PI / 2, 0, 0);
  addChassisComponent(haloGeo, haloMat, new THREE.Vector3(0, 0.65, 0.15), haloEuler);

  const helmetGeo = new THREE.SphereGeometry(0.15, 16, 16);
  addChassisComponent(helmetGeo, accentMat, new THREE.Vector3(0, 0.56, 0.1));

  const visorGeo = new THREE.BoxGeometry(0.2, 0.07, 0.1);
  addChassisComponent(visorGeo, visorMat, new THREE.Vector3(0, 0.56, 0.2));

  // 6. Front Wing Assembly (Multi-tier with endplates at +Z)
  const fwMainGeo = new THREE.BoxGeometry(W * 0.94, 0.07, 0.42);
  addChassisComponent(fwMainGeo, carbonMat, new THREE.Vector3(0, 0.12, L * 0.46));

  const fwFlapGeo = new THREE.BoxGeometry(W * 0.9, 0.05, 0.22);
  addChassisComponent(fwFlapGeo, accentMat, new THREE.Vector3(0, 0.17, L * 0.44));

  for (const side of [-1, 1]) {
    const epGeo = new THREE.BoxGeometry(0.04, 0.26, 0.48);
    addChassisComponent(epGeo, liveryMat, new THREE.Vector3(side * (W * 0.47), 0.2, L * 0.46));
  }

  // 7. Rear Wing Assembly (DRS Flap and vertical endplates at -Z)
  const rwMainGeo = new THREE.BoxGeometry(W * 0.74, 0.06, 0.34);
  addChassisComponent(rwMainGeo, carbonMat, new THREE.Vector3(0, 0.78, -L * 0.44));

  const rwFlapGeo = new THREE.BoxGeometry(W * 0.72, 0.05, 0.2);
  addChassisComponent(rwFlapGeo, liveryMat, new THREE.Vector3(0, 0.85, -L * 0.46));

  for (const side of [-1, 1]) {
    const rwEpGeo = new THREE.BoxGeometry(0.04, 0.48, 0.44);
    addChassisComponent(rwEpGeo, carbonMat, new THREE.Vector3(side * (W * 0.37), 0.7, -L * 0.44));
  }

  // 8. Pulsing Red Rain Safety LED
  const ledGeo = new THREE.BoxGeometry(0.14, 0.12, 0.05);
  const ledMat = new THREE.MeshBasicMaterial({ color: 0xff0033 });
  const rainLed = new THREE.Mesh(ledGeo, ledMat);
  rainLed.position.set(0, 0.4, -L * 0.48);
  chassisGroup.add(rainLed);

  // 9. Wheels & Suspension with Pirelli Sidewall Stripe
  const wheelRadius = 0.34;
  const wheelWidth = 0.3;
  const wheelGeo = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelWidth, 24);
  wheelGeo.rotateZ(Math.PI / 2);

  const sidewallGeo = new THREE.RingGeometry(wheelRadius * 0.65, wheelRadius * 0.82, 24);
  const rimGeo = new THREE.CylinderGeometry(wheelRadius * 0.55, wheelRadius * 0.55, wheelWidth * 1.02, 16);
  rimGeo.rotateZ(Math.PI / 2);

  const wheelPositions = [
    { x: -W * 0.48, z: L * 0.32, isFront: true },
    { x: W * 0.48, z: L * 0.32, isFront: true },
    { x: -W * 0.48, z: -L * 0.32, isFront: false },
    { x: W * 0.48, z: -L * 0.32, isFront: false },
  ];

  carGroup.userData.wheels = [];
  carGroup.userData.frontWheels = [];
  carGroup.userData.chassis = chassisGroup;
  carGroup.userData.rainLed = rainLed;
  carGroup.userData.bodyMaterials = bodyMaterials;
  carGroup.userData.edgeMaterials = edgeMaterials;
  carGroup.userData.wireMaterials = wireMaterials;

  const wheelsParent = new THREE.Group();
  carGroup.add(wheelsParent);

  for (const wp of wheelPositions) {
    const wheelAnchor = new THREE.Group();
    wheelAnchor.position.set(wp.x, wheelRadius, wp.z);

    const wheelMesh = new THREE.Mesh(wheelGeo, tireMat);
    wheelMesh.castShadow = true;

    // Pirelli yellow/red sidewall rings on outer tire face
    const sideDir = Math.sign(wp.x);
    const sidewall = new THREE.Mesh(sidewallGeo, sidewallMat);
    sidewall.position.set(sideDir * (wheelWidth * 0.51), 0, 0);
    sidewall.rotation.y = sideDir * Math.PI / 2;
    wheelMesh.add(sidewall);

    // Rim
    const rimMesh = new THREE.Mesh(rimGeo, rimMat);
    wheelMesh.add(rimMesh);

    wheelAnchor.add(wheelMesh);
    wheelsParent.add(wheelAnchor);

    carGroup.userData.wheels.push(wheelMesh);
    if (wp.isFront) {
      carGroup.userData.frontWheels.push(wheelAnchor);
    }
  }

  // 10. Soft Elliptical Ground Drop Shadow
  const shadowGeo = new THREE.PlaneGeometry(W * 1.25, L * 1.15);
  const shadowMat = new THREE.MeshBasicMaterial({
    color: 0x000000,
    opacity: 0.35,
    transparent: true,
    depthWrite: false,
  });
  const shadow = new THREE.Mesh(shadowGeo, shadowMat);
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.025;
  carGroup.add(shadow);

  return carGroup;
}

// ============================================================
// MAIN APPLICATION STATE
// ============================================================

let scene, camera, renderer3d, clock;
let replayA = null, replayB = null;
let carModelA = null, carModelB = null;
let trackGroup = null;
let liveMode = false;
let liveSocket = null;
let liveRunId = null;
let liveLastSeq = -1;
let liveCars = [];
let liveEnded = false;
let liveLastPacketAt = 0;
let liveStaleShown = false;
let livePhysicsHz = 30;

// Settings & Modes
let playing = false;
let playbackSpeed = 1.0;
let currentTime = 0;
let maxTime = 0;
let cameraMode = 'battle'; // 'battle', 'carA', 'carB', 'cockpit', 'top'
let laneSeparation = true;

// Incident tracking
let lastLeader = null;
let lastOffTrackA = false;
let lastOffTrackB = false;

// Camera Smooth Dampening
const camTargetPos = new THREE.Vector3();
const camTargetLook = new THREE.Vector3();
const camCurrentPos = new THREE.Vector3();
const camCurrentLook = new THREE.Vector3();

// Track bounds
const trackBounds = { minX: 0, maxX: 0, minZ: 0, maxZ: 0 };

// DOM Elements
const loadingOverlay = document.getElementById('loading-overlay');
const loadStatus = document.getElementById('load-status');
const filePicker = document.getElementById('file-picker');
const fileA = document.getElementById('file-a');
const fileB = document.getElementById('file-b');
const btnLoad = document.getElementById('btn-load');
const btnQuickLoad = document.getElementById('btn-quick-load');
const liveUrl = document.getElementById('live-url');
const btnLiveConnect = document.getElementById('btn-live-connect');
const btnClosePicker = document.getElementById('btn-close-picker');
const btnOpenPicker = document.getElementById('btn-open-picker');
const pickerError = document.getElementById('picker-error');
const canvas = document.getElementById('race-canvas');

const btnPlay = document.getElementById('btn-play');
const btnRestart = document.getElementById('btn-restart');
const btnStepBack = document.getElementById('btn-step-back');
const btnStepFwd = document.getElementById('btn-step-fwd');
const btnLanes = document.getElementById('btn-lanes');
const scrubber = document.getElementById('scrubber');
const timeCurrent = document.getElementById('time-current');
const timeTotal = document.getElementById('time-total');
const progressBar = document.getElementById('race-progress-bar');
const broadcastAlert = document.getElementById('broadcast-alert');
const alertText = document.getElementById('alert-text');
const overlapIndicator = document.getElementById('overlap-indicator');
const towerSector = document.getElementById('tower-sector');
const towerSession = document.getElementById('tower-session');

// Winner Victory Screen Modal Elements
const winnerModal = document.getElementById('winner-modal');
const btnCloseWinner = document.getElementById('btn-close-winner');
const winnerName = document.getElementById('winner-name');
const winnerTimeVal = document.getElementById('winner-time-val');
const winnerP1Ind = document.getElementById('winner-p1-ind');
const podP1Name = document.getElementById('pod-p1-name');
const podP1Gap = document.getElementById('pod-p1-gap');
const winnerP2Ind = document.getElementById('winner-p2-ind');
const podP2Name = document.getElementById('pod-p2-name');
const podP2Gap = document.getElementById('pod-p2-gap');
const btnReplayWinner = document.getElementById('btn-replay-winner');
const btnSwitchTrackWinner = document.getElementById('btn-switch-track-winner');
const trackSwitcher = document.getElementById('track-switcher');

// Runtime state for tracks & screens
let currentTrackKey = 'oval';
let isAnimating = false;
let winnerShownForCurrentRace = false;

/** Show a broadcast banner message. */
function showAlert(text, type = 'info', durationMs = 2500) {
  if (!broadcastAlert) return;
  alertText.textContent = text;
  broadcastAlert.className = 'broadcast-alert ' + (type === 'warn' ? 'yellow' : type === 'danger' ? 'red' : '');
  broadcastAlert.classList.remove('hidden');
  clearTimeout(showAlert._timer);
  showAlert._timer = setTimeout(() => {
    broadcastAlert.classList.add('hidden');
  }, durationMs);
}

/** Format seconds as MM:SS.mmm. */
function formatTime(s) {
  const ms = Math.max(0, Math.round(s * 1000));
  const min = Math.floor(ms / 60000);
  const sec = Math.floor((ms % 60000) / 1000);
  const mil = ms % 1000;
  return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(mil).padStart(3, '0')}`;
}

function liveInitialFrame(track) {
  const point = track.centerline?.[0] || [0, 0];
  return {
    x: Number(point[0]) || 0,
    y: Number(point[1]) || 0,
    heading: Number(track.heading?.[0]) || 0,
    speed: 0,
    lateral_velocity: 0,
    yaw_rate: 0,
    steer: 0,
    steering_command: 0,
    throttle_brake: 0,
    off_track: false,
    events: [],
    tick: 0,
    terminated: false,
    truncated: false,
  };
}

function liveReplayShell(track, entrant) {
  const participantId = entrant?.participant_id || 'Unknown Racer';
  return {
    participantId,
    tickHz: livePhysicsHz,
    car: { length_m: 4.8, width_m: 2.0 },
    track,
    frames: [liveInitialFrame(track)],
    result: {},
  };
}

function setLiveControls(enabled) {
  for (const control of [btnPlay, btnRestart, btnStepBack, btnStepFwd, scrubber, trackSwitcher]) {
    if (control) control.disabled = enabled;
  }
  if (towerSession) towerSession.textContent = enabled ? 'LIVE' : 'RACE';
}

function handleLiveHello(message) {
  liveMode = true;
  liveEnded = false;
  liveRunId = message.run_id;
  liveLastSeq = -1;
  livePhysicsHz = message.physics_hz;
  liveCars = [];
  winnerShownForCurrentRace = false;
  currentTime = 0;
  maxTime = 0;
  lastLeader = null;
  lastOffTrackA = false;
  lastOffTrackB = false;
  replayA = liveReplayShell(message.track, message.entrants[0]);
  replayB = liveReplayShell(message.track, message.entrants[1] || { participant_id: 'Waiting' });
  replayA.tickHz = livePhysicsHz;
  replayB.tickHz = livePhysicsHz;
  setLiveControls(true);
  startRace();
  if (timeTotal) timeTotal.textContent = 'LIVE';
  showAlert('LIVE PYTHON RACE CONNECTED — GREEN FLAG', 'info', 3000);
}

function handleLiveState(message) {
  liveCars = message.cars.slice(0, 2).map((car) => ({
    ...car,
    finished: car.status === 'finished',
    tick: message.tick,
  }));
  currentTime = message.sim_time_s;
  maxTime = Math.max(maxTime, currentTime);
  liveLastPacketAt = performance.now();
  liveStaleShown = false;
  if (towerSession) towerSession.textContent = 'LIVE';
  for (const [index, car] of liveCars.entries()) {
    const target = index === 0 ? replayA : replayB;
    if (!target) continue;
    target.participantId = car.label || car.participant_id;
    target.frames = [{ ...car, tick: message.tick }];
    target.result = { status: car.status, steps: message.tick, elapsed_s: message.sim_time_s };
  }
}

function handleLiveEnd(message) {
  liveEnded = true;
  currentTime = message.sim_time_s;
  maxTime = Math.max(maxTime, currentTime);
  for (const [index, result] of message.results.entries()) {
    const target = index === 0 ? replayA : replayB;
    if (!target) continue;
    target.result = {
      status: result.status,
      steps: result.steps,
      elapsed_s: message.sim_time_s,
      lap_time_s: result.lap_time_s,
    };
    if (liveCars[index]) liveCars[index].status = result.status;
  }
  if (towerSession) towerSession.textContent = 'ENDED';
  showAlert('CHECKERED FLAG — LIVE PYTHON RACE FINISHED', 'info', 4000);
  if (!winnerShownForCurrentRace) {
    winnerShownForCurrentRace = true;
    setTimeout(() => showWinnerScreen(), 250);
  }
}

function startLiveConnection(url) {
  try {
    if (liveSocket) liveSocket.close();
    liveRunId = null;
    liveLastSeq = -1;
    liveLastPacketAt = performance.now();
    loadStatus.textContent = 'Connecting to live Python race…';
    loadingOverlay.style.display = 'flex';
    loadingOverlay.classList.remove('hidden');
    liveSocket = connectLive(url, {
      onOpen: () => showAlert('LIVE SOCKET OPEN — WAITING FOR PYTHON HELLO', 'info', 2500),
      onMessage: (raw) => {
        try {
          const parsed = parseLiveMessage(raw, { runId: liveRunId, seq: liveLastSeq });
          if (parsed.kind === 'synapse.live.hello') {
            handleLiveHello(parsed.message);
          } else if (parsed.kind === 'synapse.live.state') {
            liveLastSeq = parsed.seq;
            handleLiveState(parsed.message);
          } else if (parsed.kind === 'synapse.live.end') {
            liveLastSeq = parsed.seq;
            handleLiveEnd(parsed.message);
          }
        } catch (error) {
          console.error('Live packet rejected:', error);
          showAlert(`LIVE PACKET ERROR: ${error.message}`, 'danger', 4000);
        }
      },
      onError: () => showAlert('LIVE SOCKET ERROR', 'danger', 3000),
      onClose: () => {
        if (!liveEnded) showAlert('LIVE SOCKET CLOSED', 'warn', 3000);
      },
    });
  } catch (error) {
    showAlert(`LIVE CONNECTION FAILED: ${error.message}`, 'danger', 4000);
    loadingOverlay.classList.add('hidden');
    setTimeout(() => { loadingOverlay.style.display = 'none'; }, 400);
  }
}

/** Initialize Three.js WebGL Renderer, Camera, Lights for a vibrant sunlit F1 Grid. */
function initThreeJS() {
  scene = new THREE.Scene();

  // Vibrant daylight azure sky background and soft atmospheric haze
  scene.background = new THREE.Color(0x7ec0ee);
  scene.fog = new THREE.FogExp2(0x9bd0f5, 0.001);

  camera = new THREE.PerspectiveCamera(54, window.innerWidth / window.innerHeight, 0.4, 1200);
  camera.position.set(0, 60, 0);

  renderer3d = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, powerPreference: 'high-performance' });
  renderer3d.setSize(window.innerWidth, window.innerHeight);
  renderer3d.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer3d.shadowMap.enabled = true;
  renderer3d.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer3d.toneMapping = THREE.ACESFilmicToneMapping;
  renderer3d.toneMappingExposure = 1.15;

  // High-intensity warm sunlight
  const sun = new THREE.DirectionalLight(0xfff8ee, 3.2);
  sun.position.set(140, 220, 100);
  sun.castShadow = true;
  sun.shadow.mapSize.width = 2048;
  sun.shadow.mapSize.height = 2048;
  sun.shadow.camera.near = 10;
  sun.shadow.camera.far = 500;
  sun.shadow.camera.left = -220;
  sun.shadow.camera.right = 220;
  sun.shadow.camera.top = 220;
  sun.shadow.camera.bottom = -220;
  sun.shadow.bias = -0.0005;
  scene.add(sun);

  // Hemisphere Light (Sky blue above, warm green grass bounce below)
  const hemi = new THREE.HemisphereLight(0x7dd3fc, 0x55aa44, 1.4);
  scene.add(hemi);

  // Ambient daylight fill
  const ambient = new THREE.AmbientLight(0xdbeafe, 0.9);
  scene.add(ambient);

  // Vast green grass countryside ground plane
  const groundGeo = new THREE.PlaneGeometry(1800, 1800);
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0x3d8b37, // Rich vibrant green turf
    roughness: 0.9,
    metalness: 0.05,
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.06;
  ground.receiveShadow = true;
  scene.add(ground);

  clock = new THREE.Clock();
  window.addEventListener('resize', onResize);
}

function onResize() {
  if (!camera || !renderer3d) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer3d.setSize(window.innerWidth, window.innerHeight);
}

/** Load the track and both cars into the Three.js scene. */
function loadRaceIntoScene() {
  const track = replayA.track;

  // Build track
  trackGroup = buildTrackScene(track);
  scene.add(trackGroup);

  // Compute track bounds for top-down camera
  const allPoints = [...track.barrier_left, ...track.barrier_right];
  trackBounds.minX = Infinity; trackBounds.maxX = -Infinity;
  trackBounds.minZ = Infinity; trackBounds.maxZ = -Infinity;
  for (const p of allPoints) {
    trackBounds.minX = Math.min(trackBounds.minX, p[0]);
    trackBounds.maxX = Math.max(trackBounds.maxX, p[0]);
    trackBounds.minZ = Math.min(trackBounds.minZ, -p[1]);
    trackBounds.maxZ = Math.max(trackBounds.maxZ, -p[1]);
  }

  // Build F1 Cars:
  // Car A: Electric Cyan (#00f0ff) with pure white accents
  // Car B: Radiant Papaya Solar Orange (#ff5500) with solar gold accents (#ffd166)
  carModelA = buildF1CarModel(0x00f0ff, 0xffffff, '1');
  carModelB = buildF1CarModel(0xff5500, 0xffd166, '2');
  scene.add(carModelA);
  scene.add(carModelB);

  // Determine race duration
  const durA = replayA.frames.length > 1 ? (replayA.frames.length - 1) / replayA.tickHz : 0;
  const durB = replayB.frames.length > 1 ? (replayB.frames.length - 1) / replayB.tickHz : 0;
  if (!liveMode) maxTime = Math.max(durA, durB);

  // Update HUD text
  document.getElementById('p1-name').textContent = replayA.participantId.toUpperCase();
  document.getElementById('p2-name').textContent = replayB.participantId.toUpperCase();
  document.getElementById('telem-name-a').textContent = replayA.participantId.toUpperCase();
  document.getElementById('telem-name-b').textContent = replayB.participantId.toUpperCase();
  timeTotal.textContent = liveMode ? 'LIVE' : formatTime(maxTime);

  const trackId = replayA.track.meta?.id || replayA.track.meta?.track_id || 'CIRCUIT';
  const trackTag = document.getElementById('track-name-tag');
  if (trackTag) trackTag.textContent = trackId.replace('synthetic_', '').toUpperCase();

  if (trackId.includes('flowing')) currentTrackKey = 'flowing';
  else if (trackId.includes('tight')) currentTrackKey = 'tight';
  else if (trackId.includes('oval')) currentTrackKey = 'oval';
  if (trackSwitcher) trackSwitcher.value = currentTrackKey;

  // Initial Camera position behind start line
  const f0 = replayA.frames[0];
  const initAngle = Math.atan2(Math.cos(f0.heading), -Math.sin(f0.heading));
  const fwdX = Math.sin(initAngle);
  const fwdZ = Math.cos(initAngle);

  camCurrentPos.set(f0.x - fwdX * 18, 7.5, -f0.y - fwdZ * 18);
  camCurrentLook.set(f0.x, 1.2, -f0.y);
  camera.position.copy(camCurrentPos);
  camera.lookAt(camCurrentLook);

  showAlert('GREEN FLAG — RACE STARTED', 'info', 3000);
}

// ============================================================
// VEHICLE POSITIONING, ORIENTATION & TELEMETRY
// ============================================================

/** Update car positions, rotations, wheel animations, telemetry, and overlap wireframe. */
function updateCars(timeS, dt) {
  const fallbackA = sampleAtTime(replayA.frames, replayA.tickHz, timeS);
  const fallbackB = sampleAtTime(replayB.frames, replayB.tickHz, timeS);
  const sA = liveMode ? (liveCars[0] || fallbackA) : fallbackA;
  const sB = liveMode ? (liveCars[1] || fallbackB) : fallbackB;

  // --- 1. Compute Exact Heading Angles ---
  const rotYA = Math.atan2(Math.cos(sA.heading), -Math.sin(sA.heading));
  const rotYB = Math.atan2(Math.cos(sB.heading), -Math.sin(sB.heading));

  // --- 2. Calculate Lateral Lane Offsets ---
  let offAx = 0, offAz = 0;
  let offBx = 0, offBz = 0;

  if (laneSeparation) {
    const sepM = 1.35;
    offAx = -Math.cos(rotYA) * sepM;
    offAz = Math.sin(rotYA) * sepM;

    offBx = Math.cos(rotYB) * sepM;
    offBz = -Math.sin(rotYB) * sepM;
  }

  // Position root car groups
  carModelA.position.set(sA.x + offAx, 0, -sA.y + offAz);
  carModelA.rotation.y = rotYA;

  carModelB.position.set(sB.x + offBx, 0, -sB.y + offBz);
  carModelB.rotation.y = rotYB;

  // --- 3. Dynamic Overlap Transparency & Glowing Wireframe Matrix ---
  const carDist = carModelA.position.distanceTo(carModelB.position);
  const OVERLAP_THRESHOLD = 2.8;
  const overlapFactor = Math.max(0, Math.min(1, (OVERLAP_THRESHOLD - carDist) / 2.2));

  const targetBodyOpacity = 1.0 - overlapFactor * 0.55;
  const targetWireOpacity = overlapFactor * 0.95;

  // Car A transparency & wireframe
  for (const m of carModelA.userData.bodyMaterials) {
    m.opacity = targetBodyOpacity;
  }
  for (const m of carModelA.userData.edgeMaterials) {
    m.opacity = targetWireOpacity;
  }
  for (const m of carModelA.userData.wireMaterials) {
    m.opacity = targetWireOpacity * 0.35;
  }

  // Car B transparency & wireframe
  for (const m of carModelB.userData.bodyMaterials) {
    m.opacity = targetBodyOpacity;
  }
  for (const m of carModelB.userData.edgeMaterials) {
    m.opacity = targetWireOpacity;
  }
  for (const m of carModelB.userData.wireMaterials) {
    m.opacity = targetWireOpacity * 0.35;
  }

  // Overlap badge in timing tower
  if (overlapIndicator) {
    overlapIndicator.classList.toggle('hidden', overlapFactor <= 0.25);
  }

  // --- 4. Dynamic Visual Dynamics (Wheel steering, spin, chassis pitch/roll) ---
  const wheelRadius = 0.34;

  for (const w of carModelA.userData.frontWheels) {
    w.rotation.y = sA.steer * 0.7;
  }
  for (const w of carModelA.userData.wheels) {
    w.rotation.x += (sA.speed / wheelRadius) * dt;
  }

  for (const w of carModelB.userData.frontWheels) {
    w.rotation.y = sB.steer * 0.7;
  }
  for (const w of carModelB.userData.wheels) {
    w.rotation.x += (sB.speed / wheelRadius) * dt;
  }

  const chassisA = carModelA.userData.chassis;
  chassisA.rotation.z = -sA.yaw_rate * 0.035;
  chassisA.rotation.x = sA.throttle_brake < 0 ? sA.throttle_brake * 0.025 : 0;

  const chassisB = carModelB.userData.chassis;
  chassisB.rotation.z = -sB.yaw_rate * 0.035;
  chassisB.rotation.x = sB.throttle_brake < 0 ? sB.throttle_brake * 0.025 : 0;

  // Pulsing rear rain LED
  const pulseIntensity = (Math.sin(timeS * 14) + 1) * 0.5;
  carModelA.userData.rainLed.material.color.setRGB(0.5 + pulseIntensity * 0.5, 0, 0.05);
  carModelB.userData.rainLed.material.color.setRGB(0.5 + pulseIntensity * 0.5, 0, 0.05);

  // --- 5. Update Telemetry HUD & G-Force ---
  const speedKmhA = Math.round(sA.speed * 3.6);
  const speedKmhB = Math.round(sB.speed * 3.6);
  document.getElementById('speed-a').textContent = speedKmhA;
  document.getElementById('speed-b').textContent = speedKmhB;

  const gA = Math.abs((sA.speed * sA.yaw_rate) / 9.81);
  const gB = Math.abs((sB.speed * sB.yaw_rate) / 9.81);
  const gElemA = document.getElementById('gforce-a');
  const gElemB = document.getElementById('gforce-b');
  if (gElemA) gElemA.textContent = `${gA.toFixed(1)} G`;
  if (gElemB) gElemB.textContent = `${gB.toFixed(1)} G`;

  const degA = (sA.steer * (180 / Math.PI)).toFixed(1);
  const degB = (sB.steer * (180 / Math.PI)).toFixed(1);
  const steerValA = document.getElementById('steer-val-a');
  const steerValB = document.getElementById('steer-val-b');
  if (steerValA) steerValA.textContent = `${sA.steer >= 0 ? '+' : ''}${degA}°`;
  if (steerValB) steerValB.textContent = `${sB.steer >= 0 ? '+' : ''}${degB}°`;

  const gearA = Math.min(8, Math.max(1, Math.floor(speedKmhA / 38) + 1));
  const gearB = Math.min(8, Math.max(1, Math.floor(speedKmhB / 38) + 1));
  document.getElementById('gear-a').textContent = `G${gearA}`;
  document.getElementById('gear-b').textContent = `G${gearB}`;

  const thrA = Math.max(0, sA.throttle_brake);
  const brkA = Math.max(0, -sA.throttle_brake);
  document.getElementById('throttle-a').style.width = `${Math.min(100, thrA * 100)}%`;
  document.getElementById('brake-a').style.width = `${Math.min(100, brkA * 100)}%`;
  document.getElementById('steer-a').style.left = `${Math.max(5, Math.min(95, 50 + (sA.steer / 0.55) * 45))}%`;

  const thrB = Math.max(0, sB.throttle_brake);
  const brkB = Math.max(0, -sB.throttle_brake);
  document.getElementById('throttle-b').style.width = `${Math.min(100, thrB * 100)}%`;
  document.getElementById('brake-b').style.width = `${Math.min(100, brkB * 100)}%`;
  document.getElementById('steer-b').style.left = `${Math.max(5, Math.min(95, 50 + (sB.steer / 0.55) * 45))}%`;

  // --- 6. F1 Timing Tower, Sector & Incident Alerts ---
  const tickA = sA.tick;
  const tickB = sB.tick;

  let leaderIsA = true;
  let gapStr = 'LEADER';

  if (sA.finished && !sB.finished) {
    leaderIsA = true;
    gapStr = replayB.result.status?.replace('dnf_', 'DNF ').toUpperCase() || '+1 LAP';
  } else if (sB.finished && !sA.finished) {
    leaderIsA = false;
    gapStr = replayA.result.status?.replace('dnf_', 'DNF ').toUpperCase() || '+1 LAP';
  } else if (tickA >= tickB) {
    leaderIsA = true;
    const tickDiff = tickA - tickB;
    gapStr = tickDiff > 0 ? `+${(tickDiff / replayA.tickHz).toFixed(2)}s` : 'INTERVAL 0.00s';
  } else {
    leaderIsA = false;
    const tickDiff = tickB - tickA;
    gapStr = tickDiff > 0 ? `+${(tickDiff / replayB.tickHz).toFixed(2)}s` : 'INTERVAL 0.00s';
  }

  // Overtake detection alert
  const currentLeaderName = leaderIsA ? replayA.participantId.toUpperCase() : replayB.participantId.toUpperCase();
  if (lastLeader !== null && lastLeader !== currentLeaderName && timeS > 1.0) {
    showAlert(`⚡ OVERTAKE — ${currentLeaderName} TAKES P1!`, 'info', 3000);
  }
  lastLeader = currentLeaderName;

  // Off-track alerts
  if (sA.off_track && !lastOffTrackA) {
    showAlert(`⚠️ TRACK LIMITS — ${replayA.participantId.toUpperCase()} OFF TRACK`, 'warn', 2200);
  }
  lastOffTrackA = sA.off_track;

  if (sB.off_track && !lastOffTrackB) {
    showAlert(`⚠️ TRACK LIMITS — ${replayB.participantId.toUpperCase()} OFF TRACK`, 'warn', 2200);
  }
  lastOffTrackB = sB.off_track;

  // Sector calculation (1, 2, 3)
  const progressFraction = Math.max(0, Math.min(1, maxTime > 0 ? timeS / maxTime : 0));
  const sector = progressFraction < 0.33 ? 'SEC 1' : progressFraction < 0.66 ? 'SEC 2' : 'SEC 3';
  if (towerSector) towerSector.textContent = sector;

  const p1NameElem = document.getElementById('p1-name');
  const p2NameElem = document.getElementById('p2-name');
  const p1GapElem = document.getElementById('p1-gap');
  const p2GapElem = document.getElementById('p2-gap');
  const p1Ind = document.getElementById('p1-indicator');
  const p2Ind = document.getElementById('p2-indicator');

  if (leaderIsA) {
    p1NameElem.textContent = replayA.participantId.toUpperCase();
    p1GapElem.textContent = 'LEADER';
    p1Ind.className = 'driver-indicator cyan-bg';

    p2NameElem.textContent = replayB.participantId.toUpperCase();
    p2GapElem.textContent = gapStr;
    p2Ind.className = 'driver-indicator gold-bg';
  } else {
    p1NameElem.textContent = replayB.participantId.toUpperCase();
    p1GapElem.textContent = 'LEADER';
    p1Ind.className = 'driver-indicator gold-bg';

    p2NameElem.textContent = replayA.participantId.toUpperCase();
    p2GapElem.textContent = gapStr;
    p2Ind.className = 'driver-indicator cyan-bg';
  }

  if (progressBar) progressBar.style.width = `${progressFraction * 100}%`;

  return { sA, sB, rotYA, rotYB };
}

// ============================================================
// PHASE 4: Dynamic Third-Person Chase Camera System
// ============================================================

/** Update chase camera position with smooth exponential damping and lookahead. */
function updateCamera(sA, sB, rotYA, rotYB, dt) {
  const SMOOTH = 1 - Math.pow(0.0008, dt);

  const posA = carModelA.position;
  const posB = carModelB.position;

  if (cameraMode === 'top') {
    const cx = (trackBounds.minX + trackBounds.maxX) / 2;
    const cz = (trackBounds.minZ + trackBounds.maxZ) / 2;
    const span = Math.max(trackBounds.maxX - trackBounds.minX, trackBounds.maxZ - trackBounds.minZ);
    camTargetPos.set(cx, span * 0.88, cz);
    camTargetLook.set(cx, 0, cz);
  } else if (cameraMode === 'cockpit') {
    const fwdX = Math.sin(rotYA);
    const fwdZ = Math.cos(rotYA);
    camTargetPos.set(posA.x + fwdX * 0.8, posA.y + 0.85, posA.z + fwdZ * 0.8);
    camTargetLook.set(posA.x + fwdX * 30, posA.y + 0.7, posA.z + fwdZ * 30);
  } else if (cameraMode === 'carA') {
    const fwdX = Math.sin(rotYA);
    const fwdZ = Math.cos(rotYA);
    const dist = 13.5;
    const height = 5.6;

    camTargetPos.set(posA.x - fwdX * dist, posA.y + height, posA.z - fwdZ * dist);
    camTargetLook.set(posA.x + fwdX * 4, posA.y + 1.2, posA.z + fwdZ * 4);
  } else if (cameraMode === 'carB') {
    const fwdX = Math.sin(rotYB);
    const fwdZ = Math.cos(rotYB);
    const dist = 13.5;
    const height = 5.6;

    camTargetPos.set(posB.x - fwdX * dist, posB.y + height, posB.z - fwdZ * dist);
    camTargetLook.set(posB.x + fwdX * 4, posB.y + 1.2, posB.z + fwdZ * 4);
  } else {
    // Default Battle Chase Cam
    const midX = (posA.x + posB.x) * 0.5;
    const midZ = (posA.z + posB.z) * 0.5;
    const carDist = posA.distanceTo(posB);

    const avgRotY = lerpAngle(rotYA, rotYB, 0.5);
    const fwdX = Math.sin(avgRotY);
    const fwdZ = Math.cos(avgRotY);

    const dist = Math.min(32, 17 + carDist * 0.55);
    const height = Math.min(14, 6.5 + carDist * 0.25);

    camTargetPos.set(midX - fwdX * dist, height, midZ - fwdZ * dist);
    camTargetLook.set(midX + fwdX * 5, 1.2, midZ + fwdZ * 5);
  }

  camCurrentPos.lerp(camTargetPos, SMOOTH);
  camCurrentLook.lerp(camTargetLook, SMOOTH);

  camera.position.copy(camCurrentPos);
  camera.lookAt(camCurrentLook);
}

// ============================================================
// PHASE 5: Top-Right Radar Minimap with Live Camera Frustum Cone
// ============================================================

/** Draw the 2D circuit radar minimap on the overlay canvas. */
function drawMinimap(sA, sB, rotYA, rotYB) {
  const mc = document.getElementById('minimap-canvas');
  if (!mc) return;
  const ctx = mc.getContext('2d');
  const w = mc.width;
  const h = mc.height;

  ctx.clearRect(0, 0, w, h);

  const pad = 18;
  const spanX = Math.max(1, trackBounds.maxX - trackBounds.minX);
  const spanZ = Math.max(1, trackBounds.maxZ - trackBounds.minZ);
  const scale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanZ);
  const cx = (trackBounds.minX + trackBounds.maxX) / 2;
  const cz = (trackBounds.minZ + trackBounds.maxZ) / 2;

  function toMap(x3d, z3d) {
    return {
      x: w / 2 + (x3d - cx) * scale,
      y: h / 2 + (z3d - cz) * scale,
    };
  }

  // 1. Draw Track Surface Ribbon
  const left = replayA.track.left_boundary;
  const right = replayA.track.right_boundary;
  const n = left.length;

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = toMap(left[i][0], -left[i][1]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  for (let i = n - 1; i >= 0; i--) {
    const p = toMap(right[i][0], -right[i][1]);
    ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
  ctx.fill();

  // 2. Draw Track Boundaries
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = toMap(left[i][0], -left[i][1]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
  ctx.lineWidth = 1.4;
  ctx.stroke();

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = toMap(right[i][0], -right[i][1]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.stroke();

  // 3. Draw Camera View Frustum Cone
  const camPosMap = toMap(camera.position.x, camera.position.z);
  const lookDirX = camCurrentLook.x - camera.position.x;
  const lookDirZ = camCurrentLook.z - camera.position.z;
  const camAngle = Math.atan2(lookDirZ, lookDirX);

  const coneLen = 32;
  const coneFov = 0.45;
  ctx.beginPath();
  ctx.moveTo(camPosMap.x, camPosMap.y);
  ctx.lineTo(camPosMap.x + Math.cos(camAngle - coneFov) * coneLen, camPosMap.y + Math.sin(camAngle - coneFov) * coneLen);
  ctx.lineTo(camPosMap.x + Math.cos(camAngle + coneFov) * coneLen, camPosMap.y + Math.sin(camAngle + coneFov) * coneLen);
  ctx.closePath();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.fill();

  // 4. Car A Blip (Cyan)
  const posA = toMap(carModelA.position.x, carModelA.position.z);
  ctx.save();
  ctx.translate(posA.x, posA.y);
  ctx.rotate(rotYA);

  ctx.beginPath();
  ctx.arc(0, 0, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#00f0ff';
  ctx.shadowColor = '#00f0ff';
  ctx.shadowBlur = 10;
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.beginPath();
  ctx.moveTo(0, 6);
  ctx.lineTo(2.5, -2);
  ctx.lineTo(-2.5, -2);
  ctx.closePath();
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.restore();

  // 5. Car B Blip (Papaya)
  const posB = toMap(carModelB.position.x, carModelB.position.z);
  ctx.save();
  ctx.translate(posB.x, posB.y);
  ctx.rotate(rotYB);

  ctx.beginPath();
  ctx.arc(0, 0, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#ff7700';
  ctx.shadowColor = '#ff7700';
  ctx.shadowBlur = 10;
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.beginPath();
  ctx.moveTo(0, 6);
  ctx.lineTo(2.5, -2);
  ctx.lineTo(-2.5, -2);
  ctx.closePath();
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.restore();
}

// ============================================================
// ANIMATION LOOP
// ============================================================

function animate() {
  requestAnimationFrame(animate);

  const dt = clock.getDelta();

  if (!liveMode && playing && maxTime > 0) {
    currentTime += dt * playbackSpeed;
    if (currentTime >= maxTime) {
      currentTime = maxTime;
      playing = false;
      btnPlay.textContent = '▶';
      showAlert('CHECKERED FLAG — RACE FINISHED', 'info', 4000);
      if (!winnerShownForCurrentRace) {
        winnerShownForCurrentRace = true;
        setTimeout(() => {
          showWinnerScreen();
        }, 400);
      }
    }
  }

  if (liveMode && !liveEnded && liveLastPacketAt > 0) {
    const staleAfterMs = Math.max(250, (3 / livePhysicsHz) * 1000);
    if (performance.now() - liveLastPacketAt > staleAfterMs && !liveStaleShown) {
      liveStaleShown = true;
      if (towerSession) towerSession.textContent = 'STALE';
      showAlert('LIVE STREAM STALE — HOLDING LAST AUTHORITATIVE STATE', 'warn', 2500);
    }
  }

  if (!liveMode && scrubber && maxTime > 0) {
    scrubber.value = (currentTime / maxTime) * 1000;
  }
  if (timeCurrent) timeCurrent.textContent = formatTime(currentTime);

  const { sA, sB, rotYA, rotYB } = updateCars(currentTime, dt);
  updateCamera(sA, sB, rotYA, rotYB, dt);
  drawMinimap(sA, sB, rotYA, rotYB);

  renderer3d.render(scene, camera);
}

// ============================================================
// WINNER VICTORY PODIUM DISPLAY & CIRCUIT SWITCHER
// ============================================================

/** Populate and show the official winner podium modal. */
function showWinnerScreen() {
  if (!winnerModal || !replayA || !replayB) return;

  const aFinished = replayA.result?.status === 'finished';
  const bFinished = replayB.result?.status === 'finished';

  const durA = replayA.frames.length > 1 ? (replayA.frames.length - 1) / replayA.tickHz : 0;
  const durB = replayB.frames.length > 1 ? (replayB.frames.length - 1) / replayB.tickHz : 0;

  const timeA = replayA.result?.elapsed_s || durA;
  const timeB = replayB.result?.elapsed_s || durB;

  let aWins = true;
  if (aFinished && !bFinished) {
    aWins = true;
  } else if (!aFinished && bFinished) {
    aWins = false;
  } else if (aFinished && bFinished) {
    aWins = timeA <= timeB;
  } else {
    // Both DNF: compare steps/frames
    aWins = replayA.frames.length >= replayB.frames.length;
  }

  const p1Replay = aWins ? replayA : replayB;
  const p2Replay = aWins ? replayB : replayA;
  const isP1CarA = aWins;

  const p1Time = aWins ? timeA : timeB;
  const p2Time = aWins ? timeB : timeA;

  const p1Name = p1Replay.participantId.toUpperCase();
  const p2Name = p2Replay.participantId.toUpperCase();

  if (winnerName) winnerName.textContent = p1Name;
  if (winnerTimeVal) winnerTimeVal.textContent = formatTime(p1Time);

  if (podP1Name) podP1Name.textContent = p1Name;
  if (podP1Gap) podP1Gap.textContent = 'WINNER';
  if (winnerP1Ind) {
    winnerP1Ind.className = 'pod-indicator ' + (isP1CarA ? 'cyan-bg' : 'gold-bg');
  }

  if (podP2Name) podP2Name.textContent = p2Name;
  if (winnerP2Ind) {
    winnerP2Ind.className = 'pod-indicator ' + (isP1CarA ? 'gold-bg' : 'cyan-bg');
  }

  if (podP2Gap) {
    if (p2Replay.result?.status && p2Replay.result.status.startsWith('dnf')) {
      podP2Gap.textContent = p2Replay.result.status.replace('dnf_', 'DNF ').toUpperCase();
    } else {
      const diff = Math.abs(p2Time - p1Time);
      podP2Gap.textContent = `+${diff.toFixed(3)}s`;
    }
  }

  // Next track rotation suggestion
  const nextTrackInfo = {
    'oval': { key: 'flowing', label: 'Flowing' },
    'flowing': { key: 'tight', label: 'Tight Loop' },
    'tight': { key: 'oval', label: 'Oval' }
  };
  const next = nextTrackInfo[currentTrackKey] || nextTrackInfo['oval'];
  if (btnSwitchTrackWinner) {
    btnSwitchTrackWinner.textContent = `🏎️ Next Track: ${next.label}`;
  }

  winnerModal.classList.remove('hidden');
}

/** Switch between official circuits (Oval, Flowing, Tight Loop). */
async function loadTrack(trackKey) {
  currentTrackKey = trackKey;
  if (trackSwitcher) trackSwitcher.value = trackKey;
  if (winnerModal) winnerModal.classList.add('hidden');
  winnerShownForCurrentRace = false;

  const trackFiles = {
    oval: ['./racer_a.json', './racer_b.json'],
    flowing: ['./flowing_a.json', './flowing_b.json'],
    tight: ['./tight_a.json', './tight_b.json']
  };

  const files = trackFiles[trackKey] || trackFiles.oval;

  loadStatus.textContent = `Loading Circuit: ${trackKey.toUpperCase()}…`;
  loadingOverlay.style.display = 'flex';
  loadingOverlay.classList.remove('hidden');

  try {
    const [resA, resB] = await Promise.all([
      fetch(files[0]),
      fetch(files[1])
    ]);
    if (!resA.ok || !resB.ok) {
      throw new Error(`Failed to load replay files for circuit "${trackKey}"`);
    }
    const [jsonA, jsonB] = await Promise.all([resA.json(), resB.json()]);
    replayA = parseReplay(jsonA);
    replayB = parseReplay(jsonB);

    // Clean previous Three.js objects
    if (trackGroup && scene) {
      scene.remove(trackGroup);
      trackGroup = null;
    }
    if (carModelA && scene) {
      scene.remove(carModelA);
      carModelA = null;
    }
    if (carModelB && scene) {
      scene.remove(carModelB);
      carModelB = null;
    }

    currentTime = 0;
    playing = true;
    btnPlay.textContent = '⏸';
    lastLeader = null;
    lastOffTrackA = false;
    lastOffTrackB = false;

    loadRaceIntoScene();

    loadingOverlay.classList.add('hidden');
    setTimeout(() => { loadingOverlay.style.display = 'none'; }, 400);

    showAlert(`TRACK LOADED: ${trackKey.toUpperCase()} CIRCUIT`, 'info', 3000);
  } catch (err) {
    console.error('Error loading track:', err);
    showAlert(`Error loading track: ${err.message}`, 'danger', 4000);
    loadingOverlay.classList.add('hidden');
    setTimeout(() => { loadingOverlay.style.display = 'none'; }, 400);
  }
}

// ============================================================
// CONTROLS & EVENT LISTENERS
// ============================================================

// Play / Pause
btnPlay.addEventListener('click', () => {
  if (liveMode) return;
  if (currentTime >= maxTime) {
    currentTime = 0;
    winnerShownForCurrentRace = false;
    if (winnerModal) winnerModal.classList.add('hidden');
  }
  playing = !playing;
  btnPlay.textContent = playing ? '⏸' : '▶';
});

// Restart
btnRestart.addEventListener('click', () => {
  if (liveMode) return;
  currentTime = 0;
  playing = false;
  winnerShownForCurrentRace = false;
  if (winnerModal) winnerModal.classList.add('hidden');
  btnPlay.textContent = '▶';
  showAlert('RACE RESET TO START', 'info', 1500);
});

// Step Back / Forward 1 Second
if (btnStepBack) {
  btnStepBack.addEventListener('click', () => {
    if (liveMode) return;
    currentTime = Math.max(0, currentTime - 1.0);
    if (currentTime < maxTime - 0.1) {
      winnerShownForCurrentRace = false;
      if (winnerModal) winnerModal.classList.add('hidden');
    }
  });
}
if (btnStepFwd) {
  btnStepFwd.addEventListener('click', () => {
    if (liveMode) return;
    currentTime = Math.min(maxTime, currentTime + 1.0);
  });
}

// Scrubber seek
scrubber.addEventListener('input', () => {
  if (liveMode) return;
  currentTime = (scrubber.value / 1000) * maxTime;
  if (currentTime < maxTime - 0.1) {
    winnerShownForCurrentRace = false;
    if (winnerModal) winnerModal.classList.add('hidden');
  }
  playing = false;
  btnPlay.textContent = '▶';
});

// Playback Speed Presets
document.querySelectorAll('.speed-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    playbackSpeed = parseFloat(btn.dataset.speed);
    document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

// Camera Presets
document.querySelectorAll('.cam-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    cameraMode = btn.dataset.cam;
    document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

// Lane Separation Toggle
btnLanes.addEventListener('click', () => {
  laneSeparation = !laneSeparation;
  btnLanes.textContent = laneSeparation ? 'Lanes: Dual' : 'Lanes: Raw';
  btnLanes.classList.toggle('active', laneSeparation);
  showAlert(laneSeparation ? 'SIDE-BY-SIDE DUAL LANE ACTIVATED' : 'RAW TELEMETRY COORDINATES ACTIVATED', 'info', 2000);
});

// Track Switcher Dropdown
if (trackSwitcher) {
  trackSwitcher.addEventListener('change', (e) => {
    if (liveMode) return;
    loadTrack(e.target.value);
  });
}

// Winner Screen Actions
if (btnCloseWinner) {
  btnCloseWinner.addEventListener('click', () => {
    winnerModal.classList.add('hidden');
  });
}

if (btnReplayWinner) {
  btnReplayWinner.addEventListener('click', () => {
    winnerModal.classList.add('hidden');
    currentTime = 0;
    winnerShownForCurrentRace = false;
    playing = true;
    btnPlay.textContent = '⏸';
  });
}

if (btnSwitchTrackWinner) {
  btnSwitchTrackWinner.addEventListener('click', () => {
    winnerModal.classList.add('hidden');
    const nextTrackMap = {
      'oval': 'flowing',
      'flowing': 'tight',
      'tight': 'oval'
    };
    const nextKey = nextTrackMap[currentTrackKey] || 'flowing';
    loadTrack(nextKey);
  });
}

// Custom Replays Modal Opener
btnOpenPicker.addEventListener('click', () => {
  filePicker.style.display = 'flex';
  btnClosePicker.style.display = 'block';
});

btnClosePicker.addEventListener('click', () => {
  filePicker.style.display = 'none';
});

// Keyboard Shortcuts
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); btnPlay.click(); }
  if (e.code === 'KeyR') { btnRestart.click(); }
  if (e.code === 'KeyL') { btnLanes.click(); }
  if (e.code === 'BracketLeft') { btnStepBack?.click(); }
  if (e.code === 'BracketRight') { btnStepFwd?.click(); }
  if (e.code === 'Digit1') document.querySelector('[data-cam="battle"]')?.click();
  if (e.code === 'Digit2') document.querySelector('[data-cam="carA"]')?.click();
  if (e.code === 'Digit3') document.querySelector('[data-cam="carB"]')?.click();
  if (e.code === 'Digit4') document.querySelector('[data-cam="cockpit"]')?.click();
  if (e.code === 'Digit5') document.querySelector('[data-cam="top"]')?.click();
  if (e.code === 'ArrowRight') { currentTime = Math.min(currentTime + 1 / 30, maxTime); }
  if (e.code === 'ArrowLeft') { currentTime = Math.max(currentTime - 1 / 30, 0); }
  if (e.code === 'ArrowUp') {
    const speeds = [0.5, 1, 2, 4];
    const idx = speeds.indexOf(playbackSpeed);
    if (idx < speeds.length - 1) {
      document.querySelector(`[data-speed="${speeds[idx + 1]}"]`)?.click();
    }
  }
  if (e.code === 'ArrowDown') {
    const speeds = [0.5, 1, 2, 4];
    const idx = speeds.indexOf(playbackSpeed);
    if (idx > 0) {
      document.querySelector(`[data-speed="${speeds[idx - 1]}"]`)?.click();
    }
  }
});

// File picker inputs
function checkFilesReady() {
  btnLoad.disabled = !(fileA.files.length && fileB.files.length);
}
fileA.addEventListener('change', checkFilesReady);
fileB.addEventListener('change', checkFilesReady);

btnLoad.addEventListener('click', async () => {
  pickerError.textContent = '';
  try {
    const textA = await fileA.files[0].text();
    const textB = await fileB.files[0].text();
    replayA = parseReplay(JSON.parse(textA));
    replayB = parseReplay(JSON.parse(textB));

    if (replayA.track.meta.content_hash !== replayB.track.meta.content_hash) {
      throw new Error('Both replays must use the exact same track geometry.');
    }

    filePicker.style.display = 'none';
    startRace();
  } catch (err) {
    pickerError.textContent = err.message;
  }
});

// Quick Launch built-in sample replays
btnQuickLoad.addEventListener('click', async () => {
  pickerError.textContent = '';
  try {
    loadStatus.textContent = 'Fetching built-in replays…';
    const [resA, resB] = await Promise.all([
      fetch('./racer_a.json'),
      fetch('./racer_b.json')
    ]);
    if (!resA.ok || !resB.ok) {
      throw new Error('Failed to load sample replays from server.');
    }
    const [jsonA, jsonB] = await Promise.all([resA.json(), resB.json()]);
    replayA = parseReplay(jsonA);
    replayB = parseReplay(jsonB);

    filePicker.style.display = 'none';
    startRace();
  } catch (err) {
    pickerError.textContent = 'Could not load built-in files: ' + err.message;
  }
});

if (btnLiveConnect) {
  btnLiveConnect.addEventListener('click', () => {
    const url = liveUrl?.value?.trim();
    if (!url) {
      pickerError.textContent = 'Enter a live WebSocket URL.';
      return;
    }
    pickerError.textContent = '';
    filePicker.style.display = 'none';
    startLiveConnection(url);
  });
}

/** Start the race and transition loading overlay. */
function startRace() {
  loadStatus.textContent = 'Building 3D Circuit & F1 Models…';
  loadingOverlay.style.display = 'flex';
  loadingOverlay.classList.remove('hidden');

  requestAnimationFrame(() => {
    if (!scene) initThreeJS();
    loadRaceIntoScene();

    loadingOverlay.classList.add('hidden');
    setTimeout(() => { loadingOverlay.style.display = 'none'; }, 500);

    if (!isAnimating) {
      isAnimating = true;
      animate();
    }
  });
}

// Auto-Launch on Load
async function autoInit() {
  const liveQuery = new URLSearchParams(window.location.search).get('live');
  if (liveQuery) {
    startLiveConnection(liveQuery);
    return;
  }
  // 1. Check embedded data (for single-file bundled HTML distributions)
  const dataA = document.getElementById('replay-data-a');
  const dataB = document.getElementById('replay-data-b');
  if (dataA && dataB) {
    try {
      replayA = parseReplay(JSON.parse(dataA.textContent));
      replayB = parseReplay(JSON.parse(dataB.textContent));
      startRace();
      return;
    } catch (e) {
      console.warn('Embedded data parse failed:', e);
    }
  }

  // 2. Attempt fetching built-in sample files
  try {
    const [resA, resB] = await Promise.all([
      fetch('./racer_a.json'),
      fetch('./racer_b.json')
    ]);
    if (resA.ok && resB.ok) {
      const [jsonA, jsonB] = await Promise.all([resA.json(), resB.json()]);
      replayA = parseReplay(jsonA);
      replayB = parseReplay(jsonB);
      startRace();
      return;
    }
  } catch (e) {
    console.info('Auto-fetch bypassed, showing file picker:', e);
  }

  // 3. Fallback to file picker
  loadingOverlay.style.display = 'none';
  filePicker.style.display = 'flex';
}

autoInit();
