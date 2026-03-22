import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { WebGPURenderer } from 'three/webgpu'
import {
  Fn, instanceIndex, attributeArray, uniform,
  float, vec3, cos, sin, sqrt, fract,
} from 'three/tsl'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Pane } from 'tweakpane'

const HIERARCHY = [
  { name: 'century', turnsPerParent: 1,   spt: 32 },
  { name: 'decade',  turnsPerParent: 10,  spt: 32 },
  { name: 'year',    turnsPerParent: 10,  spt: 32 },
  { name: 'day',     turnsPerParent: 365, spt: 32 },
  { name: 'hour',    turnsPerParent: 24,  spt: 16 },
  { name: 'minute',  turnsPerParent: 60,  spt: 16 },
  { name: 'second',  turnsPerParent: 60,  spt: 32 },
]
const WINDOW = 120
const TAU = Math.PI * 2

// ── CPU evalCoil ─────────────────────────────────────────────────────────────
function evalCoil(
  t: number, maxLevel: number,
  totals: number[], offsets: number[],
  R: number, L: number, omega: number, tMag: number,
  outPos: { x: number; y: number; z: number },
  outN?: { x: number; y: number; z: number },
  outB?: { x: number; y: number; z: number },
) {
  const theta = t * omega
  const cT = Math.cos(theta), sT = Math.sin(theta)
  let px = R * cT, py = (t - 0.5) * L, pz = R * sT
  let nx = -cT, ny = 0, nz = -sT
  let bx = -L * sT / tMag, by = -R * omega / tMag, bz = L * cT / tMag
  let dtx = -R * omega * sT, dty = L, dtz = R * omega * cT

  for (let lvl = 1; lvl <= maxLevel; lvl++) {
    const alpha = t * totals[lvl] * TAU
    const cA = Math.cos(alpha), sA = Math.sin(alpha)
    const off = offsets[lvl]
    const dx = cA * nx + sA * bx, dy = cA * ny + sA * by, dz = cA * nz + sA * bz
    const ex = -sA * nx + cA * bx, ey = -sA * ny + cA * by, ez = -sA * nz + cA * bz
    px += off * dx; py += off * dy; pz += off * dz
    const w = off * totals[lvl] * TAU
    dtx += w * ex; dty += w * ey; dtz += w * ez
    const tLen = Math.sqrt(dtx * dtx + dty * dty + dtz * dtz)
    const tx = dtx / tLen, ty = dty / tLen, tz = dtz / tLen
    const dot = dx * tx + dy * ty + dz * tz
    let nnx = dx - dot * tx, nny = dy - dot * ty, nnz = dz - dot * tz
    const nLen = Math.sqrt(nnx * nnx + nny * nny + nnz * nnz)
    nnx /= nLen; nny /= nLen; nnz /= nLen
    nx = nnx; ny = nny; nz = nnz
    bx = ty * nnz - tz * nny; by = tz * nnx - tx * nnz; bz = tx * nny - ty * nnx
  }
  outPos.x = px; outPos.y = py; outPos.z = pz
  if (outN) { outN.x = nx; outN.y = ny; outN.z = nz }
  if (outB) { outB.x = bx; outB.y = by; outB.z = bz }
}

// ── GPU compute ──────────────────────────────────────────────────────────────
function createCoilCompute(
  maxLevel: number, totalTurns: number[], maxPts: number,
  offUniforms: ReturnType<typeof uniform>[],
  tBaseU: ReturnType<typeof uniform>, tStepU: ReturnType<typeof uniform>,
  baseFracUniforms: ReturnType<typeof uniform>[],
  radiusU: ReturnType<typeof uniform>, lengthU: ReturnType<typeof uniform>,
  omegaU: ReturnType<typeof uniform>, tmagU: ReturnType<typeof uniform>,
  posBuf: ReturnType<typeof attributeArray>,
) {
  return Fn(() => {
    const idx = instanceIndex
    const t = tBaseU.add(tStepU.mul(idx.toFloat()))
    const helixFrac = baseFracUniforms[0].add(tStepU.mul(float(totalTurns[0])).mul(idx.toFloat()))
    const theta = fract(helixFrac).mul(float(TAU))
    const cT = cos(theta), sT = sin(theta)
    const px = radiusU.mul(cT).toVar('px')
    const py = t.sub(0.5).mul(lengthU).toVar('py')
    const pz = radiusU.mul(sT).toVar('pz')
    const nx = cT.negate().toVar('nx'), ny = float(0).toVar('ny'), nz = sT.negate().toVar('nz')
    const bx = lengthU.negate().mul(sT).div(tmagU).toVar('bx')
    const by = radiusU.negate().mul(omegaU).div(tmagU).toVar('by')
    const bz = lengthU.mul(cT).div(tmagU).toVar('bz')
    for (let lvl = 1; lvl <= maxLevel; lvl++) {
      const lvlFrac = baseFracUniforms[lvl].add(tStepU.mul(float(totalTurns[lvl])).mul(idx.toFloat()))
      const alpha = fract(lvlFrac).mul(float(TAU))
      const cA = cos(alpha), sA = sin(alpha)
      const off = offUniforms[lvl]
      const nnx = cA.mul(nx).add(sA.mul(bx)).toVar(`nnx${lvl}`)
      const nny = cA.mul(ny).add(sA.mul(by)).toVar(`nny${lvl}`)
      const nnz = cA.mul(nz).add(sA.mul(bz)).toVar(`nnz${lvl}`)
      const nbx = sA.negate().mul(nx).add(cA.mul(bx)).toVar(`nbx${lvl}`)
      const nby = sA.negate().mul(ny).add(cA.mul(by)).toVar(`nby${lvl}`)
      const nbz = sA.negate().mul(nz).add(cA.mul(bz)).toVar(`nbz${lvl}`)
      px.addAssign(off.mul(nnx)); py.addAssign(off.mul(nny)); pz.addAssign(off.mul(nnz))
      nx.assign(nnx); ny.assign(nny); nz.assign(nnz)
      bx.assign(nbx); by.assign(nby); bz.assign(nbz)
    }
    posBuf.element(idx).assign(vec3(px, py, pz))
  })().compute(maxPts)
}

// ── Types ────────────────────────────────────────────────────────────────────
interface LevelGPU {
  posBuf: ReturnType<typeof attributeArray>; compute: ReturnType<typeof createCoilCompute>
  line: THREE.Line; maxPts: number
}
interface Runtime {
  renderer: WebGPURenderer; scene: THREE.Scene
  camera: THREE.PerspectiveCamera; controls: OrbitControls
  levels: LevelGPU[]; pane: Pane
}

export default function Scene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const runtimeRef = useRef<Runtime | null>(null)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    const overlay = overlayRef.current
    if (!canvas || !overlay) return
    let disposed = false
    let onSpaceKey: ((e: KeyboardEvent) => void) | null = null

    const START_YEAR = -10000, END_YEAR = 2100
    const TOTAL_YEARS = END_YEAR - START_YEAR
    const CENTURY_TURNS = TOTAL_YEARS / 100

    const params = {
      coilRadius: 1, turnSpacing: 10, fillFactor: 0.55,
      offsets: [0, 0, 0, 0, 0, 0, 0],
      panSpeed: 0.001, focusT: 0.8228,  // Julius Caesar assassinated, 44 BCE
    }

    function syncOffsets() {
      let off = params.coilRadius
      for (let i = 1; i < params.offsets.length; i++) {
        off = off * params.fillFactor / Math.sqrt(HIERARCHY[i].turnsPerParent)
        params.offsets[i] = off
      }
    }
    syncOffsets()

    const tBaseU = uniform(0), tStepU = uniform(0)
    const baseFracUniforms = HIERARCHY.map(() => uniform(0))
    const radiusU = uniform(params.coilRadius)
    const lengthU = uniform(CENTURY_TURNS * params.turnSpacing)
    const omegaU = uniform(CENTURY_TURNS * TAU), tmagU = uniform(1)
    const offUniforms = params.offsets.map(v => uniform(v))

    function syncHelixUniforms() {
      const R = params.coilRadius, L = CENTURY_TURNS * params.turnSpacing, omega = CENTURY_TURNS * TAU
      radiusU.value = R; lengthU.value = L; omegaU.value = omega
      tmagU.value = Math.sqrt(R * R * omega * omega + L * L)
    }

    function getTotalTurns(): number[] {
      const t = [CENTURY_TURNS]
      for (let i = 1; i < HIERARCHY.length; i++) t.push(t[i - 1] * HIERARCHY[i].turnsPerParent)
      return t
    }

    function getHelixConsts() {
      const R = params.coilRadius, L = CENTURY_TURNS * params.turnSpacing, omega = CENTURY_TURNS * TAU
      return { R, L, omega, tMag: Math.sqrt(R * R * omega * omega + L * L) }
    }

    function winBounds(totalTurns: number, spt: number) {
      const halfT = (WINDOW / 2) / totalTurns
      const tS = Math.max(0, params.focusT - halfT), tE = Math.min(1, params.focusT + halfT)
      return { tS, tR: tE - tS }
    }

    // ── Camera ──────────────────────────────────────────────────────────────
    const _pos = { x: 0, y: 0, z: 0 }, _nrm = { x: 0, y: 0, z: 0 }, _bin = { x: 0, y: 0, z: 0 }
    const _rN = new THREE.Vector3(), _rB = new THREE.Vector3(), _tan = new THREE.Vector3(), _off = new THREE.Vector3()
    const _cfP0 = { x: 0, y: 0, z: 0 }, _cfP1 = { x: 0, y: 0, z: 0 }
    const _cfNv = new THREE.Vector3(), _cfBv = new THREE.Vector3(), _cfTmp = new THREE.Vector3()
    let prevFocusT = params.focusT
    let _cachedTotals: number[] | null = null
    function getCachedTotals() { if (!_cachedTotals) _cachedTotals = getTotalTurns(); return _cachedTotals }

    // Camera level: which coil level the camera tracks (can be fractional for smooth transitions)
    // Rotation level: determines the rate at which the camera rotates to stay on the "correct side"
    // of the parent coil. Typically one level above the camera level.
    let cameraLevel = 3
    let rotationLevel = 2

    function cameraFrame(t: number) {
      const totals = getCachedTotals(), hc = getHelixConsts()
      const lvl = Math.round(cameraLevel)
      evalCoil(t, lvl, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _pos)

      // If fractional level, interpolate position between floor and ceil
      const frac = cameraLevel - Math.floor(cameraLevel)
      if (frac > 0.001 && frac < 0.999) {
        const lo = Math.floor(cameraLevel), hi = Math.ceil(cameraLevel)
        evalCoil(t, lo, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _cfP0)
        evalCoil(t, hi, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _cfP1)
        _pos.x = _cfP0.x + frac * (_cfP1.x - _cfP0.x)
        _pos.y = _cfP0.y + frac * (_cfP1.y - _cfP0.y)
        _pos.z = _cfP0.z + frac * (_cfP1.z - _cfP0.z)
      }

      // Simple Y-axis rotation frame: no tilt, no bob
      // Radial = outward from Y axis (always horizontal)
      const theta = t * hc.omega
      _rN.set(Math.cos(theta), 0, Math.sin(theta))
      // Up = world Y
      _rB.set(0, 1, 0)
      // Tangential = cross(up, radial) — horizontal, perpendicular to radial
      _tan.set(-Math.sin(theta), 0, Math.cos(theta))
    }

    const LEVEL_HUES = [0, 0.08, 0.16, 0.33, 0.55, 0.72, 0.88]

    function createLevels(scene: THREE.Scene): LevelGPU[] {
      syncHelixUniforms()
      const totals = getTotalTurns()
      return HIERARCHY.map((h, lvl) => {
        const maxPts = WINDOW * h.spt + 1
        const posBuf = attributeArray(new Float32Array(maxPts * 3), 'vec3')
        const compute = createCoilCompute(lvl, totals, maxPts, offUniforms, tBaseU, tStepU, baseFracUniforms, radiusU, lengthU, omegaU, tmagU, posBuf)
        const col = new THREE.Color().setHSL(LEVEL_HUES[lvl], 1, 0.5)
        const mat = new THREE.LineBasicMaterial({ color: col })
        const geom = new THREE.BufferGeometry(); geom.setAttribute('position', posBuf.value)
        const line = new THREE.Line(geom, mat); line.frustumCulled = false; scene.add(line)
        return { posBuf, compute, line, maxPts }
      })
    }

    function disposeLevels(rt: Runtime) { for (const lv of rt.levels) { rt.scene.remove(lv.line); lv.line.geometry.dispose() }; rt.levels = [] }

    // Origin offset: the focus point's position is subtracted from everything
    // so all geometry stays near (0,0,0) where Float32 is most precise.
    const origin = { x: 0, y: 0, z: 0 }

    function computeOrigin() {
      const hc = getHelixConsts()
      const theta = params.focusT * hc.omega
      origin.x = hc.R * Math.cos(theta)
      origin.y = (params.focusT - 0.5) * hc.L
      origin.z = hc.R * Math.sin(theta)
    }

    function dispatchCompute(rt: Runtime) {
      computeOrigin()
      const totals = getTotalTurns(), hc = getHelixConsts(), pos = { x: 0, y: 0, z: 0 }
      for (let i = 0; i < rt.levels.length; i++) {
        const w = winBounds(totals[i], HIERARCHY[i].spt), maxPts = rt.levels[i].maxPts
        const arr = rt.levels[i].posBuf.value.array as Float32Array
        for (let v = 0; v < maxPts; v++) {
          const t = w.tS + w.tR * v / (maxPts - 1)
          evalCoil(t, i, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, pos)
          arr[v * 3] = pos.x - origin.x; arr[v * 3 + 1] = pos.y - origin.y; arr[v * 3 + 2] = pos.z - origin.z
        }
        rt.levels[i].posBuf.value.needsUpdate = true
        ;(rt.levels[i].posBuf as any).version = (rt.levels[i].posBuf as any).version + 1 || 1
      }
    }

    const _ufP = new THREE.Vector3(), _ufP2 = new THREE.Vector3()

    // Ideal view distance for a given camera level
    function viewDistForLevel(lvl: number): number {
      // Interpolate between integer levels
      const lo = Math.floor(lvl), hi = Math.ceil(lvl), frac = lvl - lo
      const distLo = (lo === 0 ? params.coilRadius : params.offsets[lo]) * 8
      const distHi = (hi === 0 ? params.coilRadius : params.offsets[Math.min(hi, params.offsets.length - 1)]) * 8
      return distLo + frac * (distHi - distLo)
    }

    let cameraInitialized = false
    const _camDir = new THREE.Vector3()

    function positionCamera(rt: Runtime) {
      cameraFrame(params.focusT)
      _ufP2.set(_pos.x - origin.x, _pos.y - origin.y, _pos.z - origin.z)
      const dist = viewDistForLevel(cameraLevel)

      if (!cameraInitialized) {
        // First call: set ideal starting angle
        const tilt = 0.25
        rt.camera.position.copy(_ufP2)
          .addScaledVector(_rN, dist * Math.cos(tilt))
          .addScaledVector(_rB, -dist * Math.sin(tilt))
          .addScaledVector(_tan, -dist * 0.1)
        rt.controls.target.copy(_ufP2)
        cameraInitialized = true
      } else {
        // Preserve user's orbit angle, update target + distance
        _camDir.copy(rt.camera.position).sub(rt.controls.target)
        const curDist = _camDir.length()
        if (curDist > 0.0001) _camDir.multiplyScalar(dist / curDist)
        else _camDir.set(dist, 0, 0)
        rt.controls.target.copy(_ufP2)
        rt.camera.position.copy(_ufP2).add(_camDir)
      }
    }

    function updateFocus(rt: Runtime) {
      prevFocusT = params.focusT
      dispatchCompute(rt)
      positionCamera(rt)
    }

    // ── Date formatting ─────────────────────────────────────────────────────
    function focusTToDate(ft: number): string {
      const fracYear = START_YEAR + ft * TOTAL_YEARS
      const year = Math.floor(fracYear)
      const dayOfYear = (fracYear - year) * 365.25
      const day = Math.floor(dayOfYear)
      const fracDay = dayOfYear - day
      const hour = Math.floor(fracDay * 24)
      const minute = Math.floor((fracDay * 24 - hour) * 60)
      const second = Math.floor(((fracDay * 24 - hour) * 60 - minute) * 60)
      const month = Math.floor(day / 30.44) + 1
      const dayOfMonth = Math.floor(day - (month - 1) * 30.44) + 1
      const pad = (n: number) => String(n).padStart(2, '0')
      const yearStr = year < 0 ? `${-year} BCE` : `${year} CE`
      return `${yearStr} ${pad(month)}/${pad(dayOfMonth)} ${pad(hour)}:${pad(minute)}:${pad(second)}`
    }

    function yearLabel(year: number): string {
      return year < 0 ? `${-year} BCE` : year === 0 ? '1 BCE' : `${year} CE`
    }

    function dayLabel(fracYear: number): string {
      const dayOfYear = (fracYear - Math.floor(fracYear)) * 365.25
      const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
      const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
      let rem = Math.floor(dayOfYear), m = 0
      while (m < 11 && rem >= daysInMonth[m]) { rem -= daysInMonth[m]; m++ }
      return `${months[m]} ${rem + 1}`
    }

    function hourLabel(fracYear: number): string {
      const dayOfYear = (fracYear - Math.floor(fracYear)) * 365.25
      const fracDay = dayOfYear - Math.floor(dayOfYear)
      return `${String(Math.floor(fracDay * 24)).padStart(2, '0')}:00`
    }

    // ── Init ────────────────────────────────────────────────────────────────
    const init = async () => {
      if (!navigator.gpu) { document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU not supported</div>'; return }
      const w = window.innerWidth, h = window.innerHeight
      const renderer = new WebGPURenderer({ canvas: canvas!, antialias: true, logarithmicDepthBuffer: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.setSize(w, h, false)
      await renderer.init()
      if (!(renderer as any).backend?.isWebGPUBackend) { renderer.dispose(); return }
      if (disposed) { renderer.dispose(); return }

      const scene = new THREE.Scene(); scene.background = new THREE.Color(0x111111)
      const camera = new THREE.PerspectiveCamera(60, w / h, 0.0000001, 10000)
      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true; controls.dampingFactor = 0.08; controls.enablePan = false; controls.enableZoom = false

      scene.add(new THREE.DirectionalLight(0xffffff, 2).translateX(3).translateY(5).translateZ(4))
      scene.add(new THREE.AmbientLight(0xffffff, 0.3))

      const pane = new Pane({ title: 'Time Helix' })

      // ── Unified label system ─────────────────────────────────────────────
      const LABEL_TIERS = [
        { name: 'century', yearStep: 100, coilLevel: 0, minCamLvl: 0, windowYears: Infinity, lineLen: 0.3, fontPx: 24, skipEvery: 0 },
        { name: 'decade', yearStep: 10, coilLevel: 1, minCamLvl: 1, windowYears: 500, lineLen: 0.15, fontPx: 18, skipEvery: 100 },
        { name: 'year', yearStep: 1, coilLevel: 2, minCamLvl: 2, windowYears: 50, lineLen: 0.08, fontPx: 14, skipEvery: 10 },
        { name: 'day', yearStep: 1/365.25, coilLevel: 3, minCamLvl: 3, windowYears: 1, lineLen: 0.03, fontPx: 11, skipEvery: 0 },
        { name: 'hour', yearStep: 1/8766, coilLevel: 4, minCamLvl: 4, windowYears: 0.01, lineLen: 0.01, fontPx: 9, skipEvery: 0 },
      ]

      const MAX_MARKERS = 300
      const markerPosArr = new Float32Array(MAX_MARKERS * 2 * 3)
      const markerPosAttr = new THREE.BufferAttribute(markerPosArr, 3)
      const markerColArr = new Float32Array(MAX_MARKERS * 2 * 3)
      const markerColAttr = new THREE.BufferAttribute(markerColArr, 3)
      markerColArr.fill(1)
      const markerGeom = new THREE.BufferGeometry()
      markerGeom.setAttribute('position', markerPosAttr)
      markerGeom.setAttribute('color', markerColAttr)
      const markerSegs = new THREE.LineSegments(markerGeom, new THREE.LineBasicMaterial({ vertexColors: true }))
      markerSegs.frustumCulled = false; scene.add(markerSegs)

      // Focus indicator line (always red)
      const focusPosArr = new Float32Array(2 * 3)
      const focusPosAttr = new THREE.BufferAttribute(focusPosArr, 3)
      const focusGeom = new THREE.BufferGeometry(); focusGeom.setAttribute('position', focusPosAttr)
      const focusSegs = new THREE.LineSegments(focusGeom, new THREE.LineBasicMaterial({ color: 0xff4444 }))
      focusSegs.frustumCulled = false; scene.add(focusSegs)

      function eventWindowYears(level: number): number { return level === 0 ? Infinity : level === 1 ? 500 : 50 }

      // ── Events ────────────────────────────────────────────────────────────
      interface CoilEvent { name: string; t: number; level: number; color: string }

      function yearToT(year: number): number { return (year - START_YEAR) / TOTAL_YEARS }
      function dateToT(dateStr: string): number {
        // Parse manually to handle negative years (BCE) which JS Date mangles
        const m = dateStr.match(/^(-?\d+)-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}))?/)
        if (m) {
          const year = parseInt(m[1])
          const month = parseInt(m[2]), day = parseInt(m[3])
          const hour = m[4] ? parseInt(m[4]) : 0, min = m[5] ? parseInt(m[5]) : 0
          const daysToMonth = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
          const dayOfYear = daysToMonth[month - 1] + day - 1
          const fracYear = year + (dayOfYear + (hour + min / 60) / 24) / 365.25
          return (fracYear - START_YEAR) / TOTAL_YEARS
        }
        const ms = new Date(dateStr).getTime()
        return (1970 + ms / (365.25 * 24 * 3600 * 1000) - START_YEAR) / TOTAL_YEARS
      }

      const events: CoilEvent[] = [
        // Ancient civilizations
        { name: 'Founding of Rome', t: yearToT(-753), level: 0, color: '#ff6b6b' },
        { name: 'Fall of Western Rome', t: yearToT(476), level: 0, color: '#ff6b6b' },
        { name: 'First Olympic Games', t: yearToT(-776), level: 0, color: '#ffd93d' },
        { name: 'Battle of Marathon', t: yearToT(-490), level: 1, color: '#6bcb77' },
        { name: 'Battle of Thermopylae', t: yearToT(-480), level: 1, color: '#6bcb77' },
        { name: 'Death of Socrates', t: yearToT(-399), level: 1, color: '#4d96ff' },
        { name: 'Alexander the Great born', t: yearToT(-356), level: 1, color: '#9b59b6' },
        { name: 'Death of Alexander', t: yearToT(-323), level: 1, color: '#9b59b6' },
        { name: 'Julius Caesar assassinated', t: dateToT('-0044-03-15'), level: 1, color: '#ff6b6b' },
        { name: 'Augustus becomes Emperor', t: yearToT(-27), level: 1, color: '#ff6b6b' },
        { name: 'Eruption of Vesuvius', t: dateToT('0079-08-24'), level: 1, color: '#e74c3c' },
        { name: 'Construction of Parthenon begins', t: yearToT(-447), level: 1, color: '#ffd93d' },
        { name: 'Peloponnesian War begins', t: yearToT(-431), level: 1, color: '#6bcb77' },
        { name: 'Birth of Aristotle', t: yearToT(-384), level: 1, color: '#4d96ff' },
        { name: 'Cleopatra dies', t: yearToT(-30), level: 1, color: '#e74c3c' },
        // Ancient world
        { name: 'Great Pyramid built', t: yearToT(-2560), level: 0, color: '#ffd93d' },
        { name: 'Code of Hammurabi', t: yearToT(-1754), level: 0, color: '#c0a060' },
        { name: 'Trojan War (traditional)', t: yearToT(-1184), level: 0, color: '#ffd93d' },
        { name: 'Iron Age begins', t: yearToT(-1200), level: 0, color: '#aaa' },
        // Medieval & modern
        { name: 'Fall of Constantinople', t: yearToT(1453), level: 0, color: '#e74c3c' },
        { name: 'Columbus reaches Americas', t: dateToT('1492-10-12'), level: 1, color: '#4d96ff' },
        { name: 'Gutenberg printing press', t: yearToT(1440), level: 1, color: '#c0a060' },
        { name: 'Magna Carta', t: dateToT('1215-06-15'), level: 1, color: '#6bcb77' },
        { name: 'French Revolution', t: dateToT('1789-07-14'), level: 1, color: '#ff6b6b' },
        { name: 'American Independence', t: dateToT('1776-07-04'), level: 1, color: '#4d96ff' },
        // Modern
        { name: 'Moon Landing', t: dateToT('1969-07-20T20:17:00Z'), level: 2, color: '#00ffcc' },
        { name: 'World War I begins', t: dateToT('1914-07-28'), level: 1, color: '#e74c3c' },
        { name: 'World War II begins', t: dateToT('1939-09-01'), level: 1, color: '#e74c3c' },
        { name: 'World War II ends', t: dateToT('1945-09-02'), level: 1, color: '#6bcb77' },
        { name: 'Berlin Wall falls', t: dateToT('1989-11-09'), level: 2, color: '#ffd93d' },
        { name: 'Internet goes public', t: yearToT(1991), level: 2, color: '#4d96ff' },
      ]

      // Pre-parsed event colors
      const eventRGB = events.map(ev => { const c = new THREE.Color(ev.color); return [c.r, c.g, c.b] as const })

      // ── DOM label pool ──────────────────────────────────────────────────
      const POOL_SIZE = 120
      const labelPool: HTMLDivElement[] = []
      for (let i = 0; i < POOL_SIZE; i++) {
        const div = document.createElement('div')
        div.style.cssText = 'position:absolute;font-family:monospace;pointer-events:none;white-space:nowrap;display:none'
        overlay.appendChild(div)
        labelPool.push(div)
      }

      // Fixed focus label + connecting line on the right side
      const focusLabelDiv = document.createElement('div')
      focusLabelDiv.style.cssText = 'position:absolute;right:20px;font-family:monospace;color:#ff4444;font-size:14px;font-weight:bold;pointer-events:none;white-space:nowrap;transform:translateY(-50%)'
      overlay.appendChild(focusLabelDiv)
      const focusLineDiv = document.createElement('div')
      focusLineDiv.style.cssText = 'position:absolute;height:1px;background:#ff4444;pointer-events:none'
      overlay.appendChild(focusLineDiv)
      let poolIdx = 0
      const _projVec = new THREE.Vector3()

      const rt: Runtime = { renderer, scene, camera, controls, levels: [], pane }
      runtimeRef.current = rt

      rt.levels = createLevels(scene)
      dispatchCompute(rt)

      // Initial camera via updateFocus
      updateFocus(rt)

      // ── Controls ──────────────────────────────────────────────────────────
      let zoomTarget = cameraLevel
      canvas.addEventListener('wheel', (e) => {
        e.preventDefault()
        animating = false // cancel any ongoing animation
        const step = THREE.MathUtils.clamp(-e.deltaY * 0.003, -0.5, 0.5)
        zoomTarget = THREE.MathUtils.clamp(zoomTarget + step, 0, HIERARCHY.length - 1)
      }, { passive: false })

      let dragging = false, dragOriginX = 0, dragDisplacement = 0, focusDragging = false
      const focusBinding = pane.addBinding(params, 'focusT', { min: 0, max: 1, step: 0.0001, label: 'focus' })
      focusBinding.on('change', () => { if (!focusDragging) updateFocus(rt) })
      canvas.addEventListener('pointerdown', (e) => { if (e.button === 2) { dragging = true; focusDragging = true; dragOriginX = e.clientX; dragDisplacement = 0; e.preventDefault() } })
      canvas.addEventListener('pointermove', (e) => { if (dragging) dragDisplacement = (e.clientX - dragOriginX) * 0.01 })
      const stopDrag = () => { dragging = false; focusDragging = false }
      canvas.addEventListener('pointerup', stopDrag); canvas.addEventListener('pointercancel', stopDrag)
      canvas.addEventListener('contextmenu', (e) => e.preventDefault())

      // Click event labels: animate to event, zoom out if needed but never zoom in
      overlay.addEventListener('click', (e) => {
        const target = e.target as HTMLElement
        const t = target.dataset?.eventT
        const lvl = target.dataset?.eventLevel
        if (t) {
          const evLevel = lvl ? parseFloat(lvl) : cameraLevel
          const targetLevel = Math.min(evLevel, cameraLevel) // only zoom out, never in
          animateToT(parseFloat(t), targetLevel)
        }
      })

      pane.addBinding(params, 'coilRadius', { min: 0.1, max: 5, step: 0.1, label: 'coil radius' }).on('change', () => { syncHelixUniforms(); dispatchCompute(rt) })
      pane.addBinding(params, 'turnSpacing', { min: 0.1, max: 20, step: 0.1, label: 'turn spacing' }).on('change', () => { syncHelixUniforms(); dispatchCompute(rt) })
      pane.addBinding(params, 'fillFactor', { min: 0.1, max: 1.5, step: 0.01, label: 'fill factor' }).on('change', () => { syncOffsets(); dispatchCompute(rt) })

      // ── Camera level panel (draggable) ────────────────────────────────────
      const levelPanel = document.createElement('div')
      levelPanel.style.cssText = 'position:absolute;left:20px;bottom:20px;background:rgba(17,17,17,0.85);border:1px solid #444;border-radius:6px;padding:8px;font-family:monospace;font-size:12px;color:#ccc;cursor:move;user-select:none;z-index:10'
      const timeDisplay = document.createElement('div')
      timeDisplay.style.cssText = 'margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #333'
      const timeLabel = document.createElement('div')
      timeLabel.textContent = 'Current Time'
      timeLabel.style.cssText = 'font-weight:bold;font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'
      const timeValue = document.createElement('div')
      timeValue.style.cssText = 'font-size:13px;color:#ff4444;font-weight:bold'
      timeDisplay.appendChild(timeLabel)
      timeDisplay.appendChild(timeValue)
      levelPanel.appendChild(timeDisplay)

      // ── Playback controls ────────────────────────────────────────────────
      // Speed auto-derived from camera level (seconds of sim-time per real second)
      const LEVEL_SPEEDS = [
        { label: '100yr/s', value: 100 * 365.25 * 86400 },
        { label: '10yr/s', value: 10 * 365.25 * 86400 },
        { label: '1yr/s', value: 365.25 * 86400 },
        { label: '1day/s', value: 86400 },
        { label: '1hr/s', value: 3600 },
        { label: '1min/s', value: 60 },
        { label: '1s/s', value: 1 },
      ]
      let playing = false
      let speedMult = 1.0

      const playbackDiv = document.createElement('div')
      playbackDiv.style.cssText = 'margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #333'

      const playBtn = document.createElement('div')
      playBtn.style.cssText = 'text-align:center;padding:4px 8px;border-radius:3px;cursor:pointer;background:rgba(255,255,255,0.08);color:#ccc;font-size:12px;font-family:monospace;transition:background 0.15s;user-select:none'
      playBtn.addEventListener('click', (e) => { e.stopPropagation(); playing = !playing; updatePlayBtn() })
      playBtn.addEventListener('mouseenter', () => { playBtn.style.background = 'rgba(255,255,255,0.2)' })
      playBtn.addEventListener('mouseleave', () => { playBtn.style.background = 'rgba(255,255,255,0.08)' })

      const sliderRow = document.createElement('div')
      sliderRow.style.cssText = 'display:flex;align-items:center;gap:6px;margin-top:6px'
      const speedSlider = document.createElement('input')
      speedSlider.type = 'range'
      speedSlider.min = '0.1'; speedSlider.max = '1'; speedSlider.step = '0.05'; speedSlider.value = '1'
      speedSlider.style.cssText = 'flex:1;accent-color:#888;height:4px'
      speedSlider.addEventListener('input', () => { speedMult = parseFloat(speedSlider.value); updatePlayBtn() })
      const speedLabel = document.createElement('div')
      speedLabel.style.cssText = 'font-size:10px;color:#888;min-width:28px;text-align:right;font-family:monospace'
      sliderRow.appendChild(speedSlider)
      sliderRow.appendChild(speedLabel)

      function updatePlayBtn() {
        const lvl = Math.min(Math.round(cameraLevel), LEVEL_SPEEDS.length - 1)
        const base = LEVEL_SPEEDS[lvl].label
        const mult = speedMult < 1 ? `${speedMult.toFixed(1)}x` : '1x'
        playBtn.textContent = playing ? `⏸ ${base}` : `▶ ${base}`
        speedLabel.textContent = mult
      }

      playbackDiv.appendChild(playBtn)
      playbackDiv.appendChild(sliderRow)
      levelPanel.appendChild(playbackDiv)
      updatePlayBtn()

      onSpaceKey = (e: KeyboardEvent) => {
        if (e.code === 'Space' && !e.repeat) { e.preventDefault(); playing = !playing; updatePlayBtn() }
      }
      window.addEventListener('keydown', onSpaceKey)

      const panelTitle = document.createElement('div')
      panelTitle.textContent = 'Camera Level'
      panelTitle.style.cssText = 'font-weight:bold;margin-bottom:6px;font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px'
      levelPanel.appendChild(panelTitle)

      const levelNames = ['Century', 'Decade', 'Year', 'Day', 'Hour', 'Minute', 'Second']
      const levelBtns: HTMLDivElement[] = []
      levelNames.forEach((name, i) => {
        const btn = document.createElement('div')
        btn.textContent = name
        btn.style.cssText = 'padding:4px 10px;margin:2px 0;border-radius:3px;cursor:pointer;transition:background 0.15s'
        btn.addEventListener('mouseenter', () => { if (Math.round(cameraLevel) !== i) btn.style.background = 'rgba(255,255,255,0.1)' })
        btn.addEventListener('mouseleave', () => { if (Math.round(cameraLevel) !== i) btn.style.background = 'none' })
        btn.addEventListener('click', (e) => { e.stopPropagation(); animateToT(params.focusT, i) })
        levelPanel.appendChild(btn)
        levelBtns.push(btn)
      })
      canvas.parentElement!.appendChild(levelPanel)

      // Highlight active level
      function updateLevelHighlight() {
        const active = Math.round(cameraLevel)
        levelBtns.forEach((btn, i) => {
          btn.style.background = i === active ? 'rgba(255,255,255,0.15)' : 'none'
          btn.style.color = i === active ? '#fff' : '#aaa'
        })
      }
      updateLevelHighlight()

      // Draggable
      let panelDragging = false, panelDragX = 0, panelDragY = 0
      const dragTargets = new Set([levelPanel, panelTitle, timeDisplay, timeLabel, timeValue, playbackDiv])
      levelPanel.addEventListener('pointerdown', (e) => {
        if (!dragTargets.has(e.target as any)) return
        panelDragging = true
        panelDragX = e.clientX - levelPanel.offsetLeft
        panelDragY = e.clientY - levelPanel.offsetTop
        e.preventDefault()
      })
      window.addEventListener('pointermove', (e) => {
        if (!panelDragging) return
        levelPanel.style.left = `${e.clientX - panelDragX}px`
        levelPanel.style.top = `${e.clientY - panelDragY}px`
        levelPanel.style.bottom = 'auto'
      })
      window.addEventListener('pointerup', () => { panelDragging = false })

      // ── Animate ───────────────────────────────────────────────────────────
      const _camRight = new THREE.Vector3(), _camFwd = new THREE.Vector3()
      const _mp = { x: 0, y: 0, z: 0 }

      // ── Focus animation ────────────────────────────────────────────────
      let animating = false, animStartT = 0, animEndT = 0
      let animStartLevel = 0, animEndLevel = 0
      let animStartTime = 0, animDuration = 0
      let animTransitLevel = 0 // lowest (most zoomed-out) level during flight
      let animReturnLevel: number | null = null // queue return to default after arrival

      function animateToT(targetT: number, targetLevel?: number) {
        animating = true
        animStartT = params.focusT
        animEndT = targetT
        animStartLevel = cameraLevel
        animEndLevel = targetLevel ?? cameraLevel
        animStartTime = performance.now()
        animReturnLevel = null
        zoomTarget = animEndLevel

        const temporalDistYears = Math.abs(animEndT - animStartT) * TOTAL_YEARS
        const levelDist = Math.abs(animEndLevel - animStartLevel)

        // Duration: scale with log-distance + level change
        if (temporalDistYears < 0.01) {
          // Level-only change
          animDuration = 300 + levelDist * 200
        } else {
          const logDist = Math.log10(Math.max(1, temporalDistYears))
          animDuration = 400 + logDist * 300 + levelDist * 150
        }
        animDuration = Math.max(300, Math.min(2500, animDuration))

        // Transit level: zoom out for long temporal jumps so the user sees the journey
        const minLvl = Math.min(animStartLevel, animEndLevel)
        if (temporalDistYears < 1) animTransitLevel = minLvl
        else if (temporalDistYears < 50) animTransitLevel = Math.min(minLvl, 2)
        else if (temporalDistYears < 500) animTransitLevel = Math.min(minLvl, 1)
        else animTransitLevel = Math.min(minLvl, 0)
      }

      const animate = () => {
        if (disposed) return
        rafRef.current = requestAnimationFrame(animate)

        // Smooth animation to target
        if (animating) {
          const elapsed = performance.now() - animStartTime
          const progress = Math.min(elapsed / animDuration, 1)
          // Cubic ease-in-out
          const ease = progress < 0.5
            ? 4 * progress * progress * progress
            : 1 - Math.pow(-2 * progress + 2, 3) / 2

          params.focusT = animStartT + (animEndT - animStartT) * ease

          // Level: interpolate start→end with a zoom-out hump for long journeys
          const baseLvl = animStartLevel + (animEndLevel - animStartLevel) * ease
          const zoomOut = Math.min(animStartLevel, animEndLevel) - animTransitLevel
          cameraLevel = baseLvl - zoomOut * Math.sin(progress * Math.PI)

          updateFocus(rt)
          updateLevelHighlight()
          updatePlayBtn()
          if (progress >= 1) {
            animating = false
            zoomTarget = cameraLevel
            if (animReturnLevel !== null) {
              const retLvl = animReturnLevel
              animReturnLevel = null
              animateToT(params.focusT, retLvl)
            }
          }
        }

        if (dragging && dragDisplacement !== 0) {
          animating = false; playing = false; updatePlayBtn()
          const totals = getCachedTotals()
          const panLvl = Math.min(Math.round(cameraLevel), totals.length - 1)
          params.focusT = THREE.MathUtils.clamp(params.focusT - dragDisplacement * params.panSpeed / totals[panLvl], 0, 1)
          updateFocus(rt)
        }

        // Smooth scroll zoom: lerp cameraLevel toward zoomTarget
        if (!animating) {
          const diff = zoomTarget - cameraLevel
          if (Math.abs(diff) > 0.001) {
            cameraLevel += diff * 0.12
            positionCamera(rt)
            updateLevelHighlight()
            updatePlayBtn()
          }
        }

        // Playback: advance focusT by speed * dt
        if (playing && !animating && !dragging) {
          const totalSeconds = TOTAL_YEARS * 365.25 * 24 * 3600
          const lvl = Math.min(Math.round(cameraLevel), LEVEL_SPEEDS.length - 1)
          const dt = LEVEL_SPEEDS[lvl].value * speedMult / 60 / totalSeconds // per frame at ~60fps
          params.focusT = THREE.MathUtils.clamp(params.focusT + dt, 0, 1)
          updateFocus(rt)
          updatePlayBtn()
        }

        rt.controls.update()

        // Update current time display
        timeValue.textContent = focusTToDate(params.focusT)

        rt.camera.getWorldDirection(_camFwd)
        _camRight.crossVectors(_camFwd, rt.camera.up).normalize()

        const totals = getCachedTotals(), hc = getHelixConsts()
        const sw = window.innerWidth, sh = window.innerHeight
        // Reset label pool
        poolIdx = 0
        for (const d of labelPool) { d.style.display = 'none'; d.style.transform = 'translateY(-100%)'; d.style.textAlign = 'left'; d.style.fontWeight = 'normal'; d.style.pointerEvents = 'none'; d.style.cursor = 'default'; d.dataset.eventT = ''; d.dataset.eventLevel = '' }

        const focusYear = START_YEAR + params.focusT * TOTAL_YEARS
        const camLvl = Math.round(cameraLevel)
        let markerIdx = 0

        // ── Events (always write line segments; labels only when on-screen) ─
        for (let i = 0; i < events.length; i++) {
          const ev = events[i]
          const evYear = START_YEAR + ev.t * TOTAL_YEARS
          const evWindow = eventWindowYears(ev.level)
          const temporalDist = Math.abs(evYear - focusYear)
          if (evWindow !== Infinity && temporalDist > evWindow) continue
          if (markerIdx >= MAX_MARKERS) break

          evalCoil(ev.t, ev.level, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _mp)
          const sx = _mp.x - origin.x, sy = _mp.y - origin.y, sz = _mp.z - origin.z
          const j = markerIdx * 6
          markerPosArr[j] = sx; markerPosArr[j+1] = sy; markerPosArr[j+2] = sz
          markerPosArr[j+3] = sx - 0.2 * _camRight.x
          markerPosArr[j+4] = sy - 0.2 * _camRight.y
          markerPosArr[j+5] = sz - 0.2 * _camRight.z
          const rgb = eventRGB[i]
          markerColArr[j] = markerColArr[j+3] = rgb[0]
          markerColArr[j+1] = markerColArr[j+4] = rgb[1]
          markerColArr[j+2] = markerColArr[j+5] = rgb[2]
          markerIdx++

          // Label: project and assign from pool (skip if off-screen)
          const fadeStart = evWindow === Infinity ? Infinity : evWindow * 0.7
          const opacity = evWindow === Infinity ? 1.0
            : temporalDist < fadeStart ? 1.0
            : 1.0 - (temporalDist - fadeStart) / (evWindow - fadeStart)

          const ex = sx - 0.2 * _camRight.x, ey = sy - 0.2 * _camRight.y, ez = sz - 0.2 * _camRight.z
          _projVec.set(ex, ey, ez).project(rt.camera)
          if (_projVec.z > 1) continue
          const scrX = (_projVec.x * 0.5 + 0.5) * sw
          const scrY = (-_projVec.y * 0.5 + 0.5) * sh
          if (scrX < -200 || scrX > sw + 200 || scrY < -50 || scrY > sh + 50) continue

          if (poolIdx >= POOL_SIZE) continue
          const div = labelPool[poolIdx++]
          div.textContent = ev.name
          div.style.display = 'block'
          div.style.left = `${scrX}px`
          div.style.top = `${scrY}px`
          div.style.fontSize = '14px'
          div.style.opacity = String(Math.max(0.3, opacity))
          div.style.color = ev.color
          div.style.fontWeight = 'bold'
          div.style.textAlign = 'right'
          div.style.transform = 'translate(-100%, -100%)'
          div.style.pointerEvents = 'auto'
          div.style.cursor = 'pointer'
          div.dataset.eventT = String(ev.t)
          div.dataset.eventLevel = String(ev.level)
        }

        // ── Label tiers (century → hour) ───────────────────────────────────
        for (const tier of LABEL_TIERS) {
          if (camLvl < tier.minCamLvl) continue
          const wndStart = tier.windowYears === Infinity ? START_YEAR : Math.max(START_YEAR, focusYear - tier.windowYears)
          const wndEnd = tier.windowYears === Infinity ? END_YEAR : Math.min(END_YEAR, focusYear + tier.windowYears)
          const halfSteps = Math.min(80, tier.windowYears === Infinity ? 200 : Math.ceil(tier.windowYears / tier.yearStep))
          const iterStart = Math.max(wndStart, focusYear - halfSteps * tier.yearStep)
          const iterEnd = Math.min(wndEnd, focusYear + halfSteps * tier.yearStep)
          const firstYear = Math.ceil(iterStart / tier.yearStep) * tier.yearStep

          for (let year = firstYear; year <= iterEnd; year += tier.yearStep) {
            if (tier.skipEvery > 0 && Math.abs(year - Math.round(year / tier.skipEvery) * tier.skipEvery) < tier.yearStep * 0.01) continue
            if (markerIdx >= MAX_MARKERS) break

            const t = (year - START_YEAR) / TOTAL_YEARS
            evalCoil(t, tier.coilLevel, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _mp)
            const sx = _mp.x - origin.x, sy = _mp.y - origin.y, sz = _mp.z - origin.z
            const j = markerIdx * 6
            markerPosArr[j] = sx; markerPosArr[j+1] = sy; markerPosArr[j+2] = sz
            markerPosArr[j+3] = sx + tier.lineLen * _camRight.x
            markerPosArr[j+4] = sy + tier.lineLen * _camRight.y
            markerPosArr[j+5] = sz + tier.lineLen * _camRight.z
            markerColArr[j] = markerColArr[j+3] = 1
            markerColArr[j+1] = markerColArr[j+4] = 1
            markerColArr[j+2] = markerColArr[j+5] = 1
            markerIdx++

            const temporalDist = Math.abs(year - focusYear)
            const fadeStart = tier.windowYears === Infinity ? Infinity : tier.windowYears * 0.7
            const opacity = tier.windowYears === Infinity ? 1.0
              : temporalDist < fadeStart ? 1.0
              : 1.0 - (temporalDist - fadeStart) / (tier.windowYears - fadeStart)

            _projVec.set(sx, sy, sz).project(rt.camera)
            if (_projVec.z > 1) continue
            const scrX = (_projVec.x * 0.5 + 0.5) * sw
            const scrY = (-_projVec.y * 0.5 + 0.5) * sh
            if (scrX < -100 || scrX > sw + 100 || scrY < -50 || scrY > sh + 50) continue

            if (poolIdx >= POOL_SIZE) continue
            const div = labelPool[poolIdx++]
            let text: string
            if (tier.name === 'day') text = dayLabel(year)
            else if (tier.name === 'hour') text = hourLabel(year)
            else text = yearLabel(Math.round(year))

            div.textContent = text
            div.style.display = 'block'
            div.style.left = `${scrX}px`
            div.style.top = `${scrY}px`
            div.style.fontSize = `${tier.fontPx}px`
            div.style.opacity = String(Math.max(0.2, opacity))
            div.style.color = 'white'
          }
        }

        // ── Focus indicator ────────────────────────────────────────────────
        {
          evalCoil(params.focusT, camLvl, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _mp)
          const fx = _mp.x - origin.x, fy = _mp.y - origin.y, fz = _mp.z - origin.z
          focusPosArr[0] = fx; focusPosArr[1] = fy; focusPosArr[2] = fz
          focusPosArr[3] = fx + 0.1 * _camRight.x
          focusPosArr[4] = fy + 0.1 * _camRight.y
          focusPosArr[5] = fz + 0.1 * _camRight.z
          focusPosAttr.needsUpdate = true

          _projVec.set(fx, fy, fz).project(rt.camera)
          const fScrX = (_projVec.x * 0.5 + 0.5) * sw
          const fScrY = (-_projVec.y * 0.5 + 0.5) * sh
          const labelY = Math.max(20, Math.min(sh - 20, fScrY))
          focusLabelDiv.textContent = focusTToDate(params.focusT)
          focusLabelDiv.style.top = `${labelY}px`
          const labelX = sw - 20 - focusLabelDiv.offsetWidth
          const lineLeft = Math.min(fScrX, labelX)
          const lineWidth = Math.abs(labelX - fScrX)
          focusLineDiv.style.left = `${lineLeft}px`
          focusLineDiv.style.top = `${labelY}px`
          focusLineDiv.style.width = `${lineWidth}px`
        }

        markerGeom.setDrawRange(0, markerIdx * 2)
        markerPosAttr.needsUpdate = true
        markerColAttr.needsUpdate = true

        rt.renderer.render(rt.scene, rt.camera)
      }
      rafRef.current = requestAnimationFrame(animate)
    }

    init()
    const handleResize = () => {
      const rt = runtimeRef.current; if (!rt) return
      rt.renderer.setSize(window.innerWidth, window.innerHeight, false)
      rt.camera.aspect = window.innerWidth / window.innerHeight
      rt.camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', handleResize)
    return () => {
      disposed = true; window.removeEventListener('resize', handleResize); if (onSpaceKey) window.removeEventListener('keydown', onSpaceKey)
      cancelAnimationFrame(rafRef.current)
      const rt = runtimeRef.current
      if (rt) { rt.pane.dispose(); rt.controls.dispose(); disposeLevels(rt); rt.renderer.dispose(); runtimeRef.current = null }
      // Clean up DOM labels
      while (overlay.firstChild) overlay.removeChild(overlay.firstChild)
    }
  }, [])

  return (
    <>
      <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
      <div ref={overlayRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'hidden' }} />
    </>
  )
}
