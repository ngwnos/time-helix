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
      panSpeed: 0.001, focusT: 0.7854,  // Ionian Revolt, ~497 BCE
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

    let prevOriginT = -1
    let originBlend = 1.0 // 1 = follow helix exactly, 0 = follow Y axis (no wobble)
    function computeOrigin() {
      const hc = getHelixConsts()
      const theta = params.focusT * hc.omega
      const helixX = hc.R * Math.cos(theta)
      const helixZ = hc.R * Math.sin(theta)
      origin.y = (params.focusT - 0.5) * hc.L

      if (prevOriginT < 0) {
        origin.x = helixX; origin.z = helixZ
        prevOriginT = params.focusT
        return
      }

      const dt = Math.abs(params.focusT - prevOriginT)
      const turnsPerFrame = dt * CENTURY_TURNS
      prevOriginT = params.focusT

      // Blend target: slow → 1 (helix), fast → 0 (Y axis)
      const target = turnsPerFrame < 0.001 ? 1.0 : Math.max(0, 1.0 - turnsPerFrame * 20)
      originBlend += (target - originBlend) * 0.08

      origin.x = helixX * originBlend
      origin.z = helixZ * originBlend
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
      const hc = getHelixConsts()
      const theta = params.focusT * hc.omega
      // Camera target = sub-coil offset from the century helix.
      // Strips out the helical X/Z oscillation — at level 0 this is (0,0,0).
      _ufP2.x = _pos.x - hc.R * Math.cos(theta)
      _ufP2.y = _pos.y - origin.y
      _ufP2.z = _pos.z - hc.R * Math.sin(theta)

      const dist = viewDistForLevel(cameraLevel)

      if (!cameraInitialized) {
        const tilt = 0.25
        rt.camera.position.copy(_ufP2)
          .addScaledVector(_rN, dist * Math.cos(tilt))
          .addScaledVector(_rB, -dist * Math.sin(tilt))
          .addScaledVector(_tan, -dist * 0.1)
        rt.controls.target.copy(_ufP2)
        cameraInitialized = true
      } else {
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

    function eventDateStr(t: number): string {
      const fracYear = START_YEAR + t * TOTAL_YEARS
      const year = Math.floor(fracYear)
      const frac = fracYear - year
      // No sub-year precision — just show the year
      if (frac < 0.003) return yearLabel(year) // within ~1 day of Jan 1
      const dayOfYear = frac * 365.25
      const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
      const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
      let rem = Math.floor(dayOfYear), m = 0
      while (m < 11 && rem >= daysInMonth[m]) { rem -= daysInMonth[m]; m++ }
      const fracDay = dayOfYear - Math.floor(dayOfYear)
      // Has hour precision?
      if (fracDay > 0.02) {
        const hour = Math.floor(fracDay * 24)
        return `${yearLabel(year)}, ${months[m]} ${rem + 1} ${hour}:00`
      }
      return `${yearLabel(year)}, ${months[m]} ${rem + 1}`
    }

    function relativeDateStr(t: number, focusT: number): string {
      const dy = (t - focusT) * TOTAL_YEARS
      const abs = Math.abs(dy)
      const sign = dy < 0 ? '-' : '+'
      if (abs < 1 / 8766) return 'now'
      if (abs < 1 / 365.25) return `${sign}${(abs * 8766).toFixed(0)}h`
      if (abs < 1) return `${sign}${(abs * 365.25).toFixed(0)}d`
      if (abs < 100) return `${sign}${abs.toFixed(1)}y`
      if (abs < 1000) return `${sign}${Math.round(abs)}y`
      return `${sign}${(abs / 1000).toFixed(1)}ky`
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


      function eventWindowYears(level: number): number {
        return level === 0 ? 3000 : level === 1 ? 500 : level === 2 ? 50 : level === 3 ? 5 : 0.5
      }

      // ── Events ────────────────────────────────────────────────────────────
      interface CoilEvent { name: string; t: number; level: number; color: string; tMin?: number; tMax?: number }

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

      let events: CoilEvent[] = [
        // Prehistoric & early civilization
        { name: 'First cave paintings (Lascaux)', t: yearToT(-8000), level: 0, color: '#c0a060' },
        { name: 'Agriculture begins (Fertile Crescent)', t: yearToT(-9500), level: 0, color: '#6bcb77' },
        { name: 'Çatalhöyük settled', t: yearToT(-7500), level: 0, color: '#c0a060' },
        { name: 'First writing (Sumerian cuneiform)', t: yearToT(-3400), level: 0, color: '#c0a060' },
        { name: 'Unification of Egypt', t: yearToT(-3100), level: 0, color: '#ffd93d' },
        { name: 'Great Pyramid built', t: yearToT(-2560), level: 0, color: '#ffd93d' },
        { name: 'Indus Valley Civilization peak', t: yearToT(-2500), level: 0, color: '#9b59b6' },
        { name: 'Stonehenge completed', t: yearToT(-2500), level: 0, color: '#aaa' },
        { name: 'Bronze Age begins', t: yearToT(-3300), level: 0, color: '#c0a060' },
        { name: 'Code of Hammurabi', t: yearToT(-1754), level: 0, color: '#c0a060' },
        { name: 'Shang Dynasty begins', t: yearToT(-1600), level: 0, color: '#ffd93d' },
        { name: 'Trojan War (traditional)', t: yearToT(-1184), level: 0, color: '#ffd93d' },
        { name: 'Iron Age begins', t: yearToT(-1200), level: 0, color: '#aaa' },
        { name: 'Phoenician alphabet spreads', t: yearToT(-1050), level: 0, color: '#c0a060' },
        // Classical antiquity
        { name: 'First Olympic Games', t: yearToT(-776), level: 0, color: '#ffd93d' },
        { name: 'Founding of Rome', t: yearToT(-753), level: 0, color: '#ff6b6b' },
        { name: 'Buddha born', t: yearToT(-563), level: 0, color: '#9b59b6' },
        { name: 'Confucius born', t: yearToT(-551), level: 0, color: '#9b59b6' },
        { name: 'Athenian democracy established', t: yearToT(-508), level: 1, color: '#4d96ff' },
        // Ionian Revolt (499–493 BCE) — level 2 summary, level 3 details
        { name: 'Ionian Revolt begins', t: yearToT(-499), level: 2, color: '#6bcb77' },
        { name: 'Aristagoras appeals to Athens', t: dateToT('-0499-01-15'), level: 3, color: '#6bcb77', tMin: dateToT('-0499-03-01'), tMax: dateToT('-0498-11-01') },
        { name: 'Greeks burn Sardis', t: dateToT('-0498-06-01'), level: 3, color: '#e74c3c', tMin: dateToT('-0498-04-01'), tMax: dateToT('-0498-08-01') },
        { name: 'Battle of Ephesus', t: dateToT('-0498-09-01'), level: 3, color: '#e74c3c', tMin: dateToT('-0498-06-01'), tMax: dateToT('-0498-12-01') },
        { name: 'Cyprus joins revolt', t: dateToT('-0497-03-01'), level: 3, color: '#6bcb77', tMin: dateToT('-0498-01-01'), tMax: dateToT('-0497-06-01') },
        { name: 'Battle of Salamis (Cyprus)', t: dateToT('-0497-07-01'), level: 3, color: '#e74c3c', tMin: dateToT('-0497-05-01'), tMax: dateToT('-0497-09-01') },
        { name: 'Carians defeated at Marsyas', t: dateToT('-0497-10-01'), level: 3, color: '#e74c3c', tMin: dateToT('-0497-06-01'), tMax: dateToT('-0496-02-01') },
        { name: 'Aristagoras killed in Thrace', t: yearToT(-497), level: 3, color: '#e74c3c', tMin: yearToT(-497), tMax: yearToT(-496) },
        { name: 'Persian siege of Miletus begins', t: dateToT('-0494-06-01'), level: 3, color: '#e74c3c', tMin: dateToT('-0494-04-01'), tMax: dateToT('-0494-08-01') },
        { name: 'Battle of Lade', t: dateToT('-0494-08-01'), level: 3, color: '#e74c3c', tMin: dateToT('-0494-07-01'), tMax: dateToT('-0494-09-01') },
        { name: 'Fall of Miletus', t: dateToT('-0494-10-01'), level: 3, color: '#e74c3c', tMin: dateToT('-0494-09-01'), tMax: dateToT('-0494-12-01') },
        { name: 'Ionian Revolt ends', t: yearToT(-493), level: 2, color: '#e74c3c' },
        { name: 'Battle of Marathon', t: yearToT(-490), level: 1, color: '#6bcb77' },
        { name: 'Battle of Thermopylae', t: yearToT(-480), level: 1, color: '#6bcb77' },
        { name: 'Construction of Parthenon begins', t: yearToT(-447), level: 1, color: '#ffd93d' },
        { name: 'Peloponnesian War begins', t: yearToT(-431), level: 1, color: '#6bcb77' },
        { name: 'Death of Socrates', t: yearToT(-399), level: 1, color: '#4d96ff' },
        { name: 'Birth of Aristotle', t: yearToT(-384), level: 1, color: '#4d96ff' },
        { name: 'Alexander the Great born', t: yearToT(-356), level: 1, color: '#9b59b6' },
        { name: 'Death of Alexander', t: yearToT(-323), level: 1, color: '#9b59b6' },
        { name: 'Ashoka unifies India', t: yearToT(-261), level: 1, color: '#9b59b6' },
        { name: 'Great Wall construction begins', t: yearToT(-221), level: 0, color: '#ffd93d' },
        { name: 'Julius Caesar assassinated', t: dateToT('-0044-03-15'), level: 1, color: '#ff6b6b' },
        { name: 'Augustus becomes Emperor', t: yearToT(-27), level: 1, color: '#ff6b6b' },
        { name: 'Cleopatra dies', t: yearToT(-30), level: 1, color: '#e74c3c' },
        { name: 'Jesus of Nazareth born', t: yearToT(-4), level: 0, color: '#9b59b6' },
        { name: 'Eruption of Vesuvius', t: dateToT('0079-10-24'), level: 1, color: '#e74c3c', tMin: dateToT('0079-08-24'), tMax: dateToT('0079-10-24') },
        { name: 'Vesuvius: Plinian column erupts (~1 PM)', t: dateToT('0079-10-24T13:00'), level: 3, color: '#e74c3c', tMin: dateToT('0079-08-24T13:00'), tMax: dateToT('0079-10-24T13:00') },
        { name: 'Vesuvius: first pyroclastic surge (~1 AM)', t: dateToT('0079-10-25T01:00'), level: 3, color: '#e74c3c', tMin: dateToT('0079-08-25T01:00'), tMax: dateToT('0079-10-25T01:00') },
        { name: 'Pompeii buried', t: dateToT('0079-10-25T07:00'), level: 3, color: '#e74c3c', tMin: dateToT('0079-08-25T07:00'), tMax: dateToT('0079-10-25T07:00') },
        { name: 'Colosseum completed', t: yearToT(80), level: 1, color: '#ffd93d' },
        // Late antiquity & early medieval
        { name: 'Constantine converts to Christianity', t: yearToT(312), level: 1, color: '#9b59b6' },
        { name: 'Fall of Western Rome', t: yearToT(476), level: 0, color: '#ff6b6b' },
        { name: 'Muhammad born', t: yearToT(570), level: 0, color: '#9b59b6' },
        { name: 'Islamic Golden Age begins', t: yearToT(750), level: 0, color: '#ffd93d' },
        { name: 'Charlemagne crowned Emperor', t: dateToT('0800-12-25'), level: 1, color: '#ff6b6b' },
        { name: 'Viking Age begins', t: yearToT(793), level: 1, color: '#6bcb77' },
        // High medieval
        { name: 'Norman Conquest of England', t: yearToT(1066), level: 1, color: '#6bcb77' },
        { name: 'First Crusade', t: yearToT(1096), level: 1, color: '#e74c3c' },
        { name: 'Genghis Khan unites Mongols', t: yearToT(1206), level: 0, color: '#e74c3c' },
        { name: 'Magna Carta', t: dateToT('1215-06-15'), level: 1, color: '#6bcb77' },
        { name: 'Black Death reaches Europe', t: yearToT(1347), level: 0, color: '#e74c3c' },
        { name: 'Ming Dynasty begins', t: yearToT(1368), level: 1, color: '#ffd93d' },
        // Renaissance & early modern
        { name: 'Gutenberg printing press', t: yearToT(1440), level: 1, color: '#c0a060' },
        { name: 'Fall of Constantinople', t: yearToT(1453), level: 0, color: '#e74c3c' },
        { name: 'Columbus reaches Americas', t: dateToT('1492-10-12'), level: 1, color: '#4d96ff' },
        { name: 'Vasco da Gama reaches India', t: yearToT(1498), level: 1, color: '#4d96ff' },
        { name: 'Luther\'s 95 Theses', t: yearToT(1517), level: 1, color: '#9b59b6' },
        { name: 'Magellan circumnavigates globe', t: yearToT(1522), level: 1, color: '#4d96ff' },
        { name: 'Copernicus publishes heliocentric model', t: yearToT(1543), level: 1, color: '#4d96ff' },
        { name: 'Shakespeare born', t: yearToT(1564), level: 1, color: '#ffd93d' },
        { name: 'Galileo observes Jupiter\'s moons', t: yearToT(1610), level: 1, color: '#4d96ff' },
        { name: 'Newton publishes Principia', t: yearToT(1687), level: 1, color: '#4d96ff' },
        // Revolutions & enlightenment
        { name: 'American Independence', t: dateToT('1776-07-04'), level: 1, color: '#4d96ff' },
        { name: 'French Revolution', t: dateToT('1789-07-14'), level: 1, color: '#ff6b6b' },
        { name: 'Haitian Revolution', t: yearToT(1791), level: 1, color: '#ff6b6b' },
        { name: 'Napoleon crowned Emperor', t: yearToT(1804), level: 1, color: '#ff6b6b' },
        { name: 'Battle of Waterloo', t: dateToT('1815-06-18'), level: 1, color: '#6bcb77' },
        // Industrial age
        { name: 'Steam locomotive (Stephenson)', t: yearToT(1825), level: 1, color: '#c0a060' },
        { name: 'Darwin publishes Origin of Species', t: yearToT(1859), level: 1, color: '#4d96ff' },
        { name: 'US Civil War begins', t: yearToT(1861), level: 1, color: '#e74c3c' },
        { name: 'Abolition of US slavery', t: yearToT(1865), level: 1, color: '#6bcb77' },
        { name: 'Meiji Restoration (Japan)', t: yearToT(1868), level: 1, color: '#ffd93d' },
        { name: 'Telephone invented', t: yearToT(1876), level: 2, color: '#c0a060' },
        { name: 'Light bulb invented', t: yearToT(1879), level: 2, color: '#ffd93d' },
        // 20th century
        { name: 'First powered flight', t: dateToT('1903-12-17'), level: 1, color: '#00ffcc' },
        { name: 'Einstein publishes relativity', t: yearToT(1905), level: 1, color: '#4d96ff' },
        { name: 'World War I begins', t: dateToT('1914-07-28'), level: 1, color: '#e74c3c' },
        { name: 'Russian Revolution', t: yearToT(1917), level: 1, color: '#ff6b6b' },
        { name: 'World War I ends', t: dateToT('1918-11-11'), level: 1, color: '#6bcb77' },
        { name: 'Penicillin discovered', t: yearToT(1928), level: 2, color: '#4d96ff' },
        { name: 'World War II begins', t: dateToT('1939-09-01'), level: 1, color: '#e74c3c' },
        { name: 'D-Day', t: dateToT('1944-06-06'), level: 2, color: '#6bcb77' },
        { name: 'Atomic bomb dropped on Hiroshima', t: dateToT('1945-08-06'), level: 1, color: '#e74c3c' },
        { name: 'World War II ends', t: dateToT('1945-09-02'), level: 1, color: '#6bcb77' },
        { name: 'United Nations founded', t: yearToT(1945), level: 1, color: '#4d96ff' },
        { name: 'India gains independence', t: yearToT(1947), level: 1, color: '#6bcb77' },
        { name: 'Israel established', t: yearToT(1948), level: 1, color: '#ff6b6b' },
        { name: 'DNA structure discovered', t: yearToT(1953), level: 2, color: '#4d96ff' },
        { name: 'Sputnik launched', t: dateToT('1957-10-04'), level: 2, color: '#00ffcc' },
        { name: 'Cuban Missile Crisis', t: yearToT(1962), level: 2, color: '#e74c3c' },
        { name: 'Moon Landing', t: dateToT('1969-07-20T20:17:00'), level: 1, color: '#00ffcc' },
        { name: 'Fall of Saigon', t: yearToT(1975), level: 2, color: '#e74c3c' },
        { name: 'First personal computer', t: yearToT(1977), level: 2, color: '#c0a060' },
        { name: 'Chernobyl disaster', t: dateToT('1986-04-26'), level: 2, color: '#e74c3c' },
        { name: 'Berlin Wall falls', t: dateToT('1989-11-09'), level: 1, color: '#ffd93d' },
        { name: 'World Wide Web launched', t: yearToT(1991), level: 1, color: '#4d96ff' },
        { name: 'Soviet Union dissolves', t: yearToT(1991), level: 1, color: '#ff6b6b' },
        { name: 'Mandela becomes president', t: yearToT(1994), level: 2, color: '#6bcb77' },
        // 21st century
        { name: 'September 11 attacks', t: dateToT('2001-09-11'), level: 1, color: '#e74c3c' },
        { name: 'Human genome sequenced', t: yearToT(2003), level: 2, color: '#4d96ff' },
        { name: 'iPhone released', t: yearToT(2007), level: 2, color: '#c0a060' },
        { name: 'Arab Spring begins', t: yearToT(2011), level: 2, color: '#ff6b6b' },
        { name: 'Higgs boson discovered', t: yearToT(2012), level: 2, color: '#4d96ff' },
        { name: 'COVID-19 pandemic begins', t: dateToT('2020-03-11'), level: 1, color: '#e74c3c' },
        { name: 'ChatGPT released', t: dateToT('2022-11-30'), level: 2, color: '#00ffcc' },
      ]

      // Pre-parsed event colors
      let eventRGB = events.map(ev => { const c = new THREE.Color(ev.color); return [c.r, c.g, c.b] as const })

      const EVENT_SCHEMA = `{
  "events": [
    {
      "name": "Event Name",
      "year": 1969,
      "date": "1969-07-20",
      "level": 0,
      "color": "#ff6b6b"
    }
  ]
}

Fields:
- name (required): Display name
- year OR date (required): Use "year" for year-only precision (e.g. -753), "date" for specific dates (e.g. "1969-07-20" or "-0044-03-15T13:00")
- level (required): 0=century, 1=decade, 2=year, 3=day scale. Controls visibility range and coil placement.
- color (required): Hex color string
- tMin, tMax (optional): Uncertainty range as year or date, same format as year/date`

      function parseEventJson(json: string): CoilEvent[] | string {
        try {
          const data = JSON.parse(json)
          const arr = data.events || data
          if (!Array.isArray(arr)) return 'Expected { "events": [...] } or an array'
          const parsed: CoilEvent[] = []
          for (const e of arr) {
            if (!e.name || e.level == null || !e.color) return `Missing name/level/color on: ${JSON.stringify(e).slice(0, 60)}`
            const t = e.date ? dateToT(String(e.date)) : e.year != null ? yearToT(e.year) : null
            if (t == null) return `Missing year or date on: ${e.name}`
            const ev: CoilEvent = { name: e.name, t, level: e.level, color: e.color }
            if (e.tMin != null) ev.tMin = typeof e.tMin === 'string' ? dateToT(e.tMin) : yearToT(e.tMin)
            if (e.tMax != null) ev.tMax = typeof e.tMax === 'string' ? dateToT(e.tMax) : yearToT(e.tMax)
            parsed.push(ev)
          }
          return parsed
        } catch (err) { return String(err) }
      }

      function setEvents(newEvents: CoilEvent[]) {
        events = newEvents
        eventRGB = events.map(ev => { const c = new THREE.Color(ev.color); return [c.r, c.g, c.b] as const })
      }

      // ── DOM label pool ──────────────────────────────────────────────────
      const POOL_SIZE = 120
      const labelPool: HTMLDivElement[] = []
      for (let i = 0; i < POOL_SIZE; i++) {
        const div = document.createElement('div')
        div.style.cssText = 'position:absolute;font-family:monospace;pointer-events:none;white-space:nowrap;display:none'
        overlay.appendChild(div)
        labelPool.push(div)
      }

      // Event connector canvas (dots on coil + lines to labels)
      const connCanvas = document.createElement('canvas')
      connCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none'
      overlay.appendChild(connCanvas)

      // Fixed focus label + connecting line on the right side
      const focusChevron = document.createElement('div')
      focusChevron.innerHTML = '<svg width="28" height="20" viewBox="0 0 28 20"><path d="M28,0 L6,10 L28,20 L22,10 Z" fill="#ff4444"/></svg>'
      focusChevron.style.cssText = 'position:absolute;pointer-events:none;transform:translate(-100%,-50%);line-height:0'
      let chevronX = -1, chevronY = -1
      overlay.appendChild(focusChevron)
      const focusLabelDiv = document.createElement('div')
      focusLabelDiv.style.cssText = 'position:absolute;right:20px;font-family:monospace;color:#ff4444;font-size:14px;font-weight:bold;pointer-events:none;white-space:nowrap;transform:translateY(-50%)'
      overlay.appendChild(focusLabelDiv)
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

      let dragging = false, dragOriginX = 0, dragOriginY = 0, dragCurX = 0, dragCurY = 0, dragDisplacement = 0, focusDragging = false
      let panInertia = 0
      const focusBinding = pane.addBinding(params, 'focusT', { min: 0, max: 1, step: 0.0001, label: 'focus' })
      focusBinding.on('change', () => { if (!focusDragging) updateFocus(rt) })

      // ── Pan indicator widget ─────────────────────────────────────────────
      const panWidget = document.createElement('div')
      panWidget.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;display:none;z-index:5'
      const panSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
      panSvg.setAttribute('width', '100%'); panSvg.setAttribute('height', '100%')
      panSvg.style.cssText = 'position:absolute;top:0;left:0'
      panWidget.appendChild(panSvg)
      const panLabel = document.createElement('div')
      panLabel.style.cssText = 'position:absolute;font-family:monospace;font-size:11px;color:#ccc;background:rgba(17,17,17,0.85);border:1px solid #444;border-radius:4px;padding:2px 6px;white-space:nowrap;transform:translate(-50%,-100%);margin-top:-12px'
      panWidget.appendChild(panLabel)
      canvas.parentElement!.appendChild(panWidget)

      function formatPanRate(yearsPerSec: number): string {
        const abs = Math.abs(yearsPerSec)
        if (abs < 1e-6) return '0'
        if (abs < 1 / 8766) return `${(abs * 8766 * 60).toFixed(1)} min/s`
        if (abs < 1 / 365.25) return `${(abs * 8766).toFixed(1)} hr/s`
        if (abs < 1) return `${(abs * 365.25).toFixed(1)} day/s`
        if (abs < 100) return `${abs.toFixed(1)} yr/s`
        if (abs < 10000) return `${(abs / 100).toFixed(1)} c/s`
        return `${(abs / 1000).toFixed(0)}k yr/s`
      }

      function updatePanWidget() {
        if (!dragging) { panWidget.style.display = 'none'; return }
        panWidget.style.display = 'block'
        const ox = dragOriginX, oy = dragOriginY, cx = dragCurX, cy = dragCurY
        const ddx = cx - ox, ddy = cy - oy
        const dist = Math.sqrt(ddx * ddx + ddy * ddy)

        // Compute actual pan rate in years/second (~60fps)
        const totals = getCachedTotals()
        const panLvl = Math.min(Math.round(cameraLevel), totals.length - 1)
        const dtPerFrame = dragDisplacement * params.panSpeed / totals[panLvl] * TOTAL_YEARS
        const yearsPerSec = dtPerFrame * 60
        const dir = yearsPerSec > 0.0001 ? ' ◀' : yearsPerSec < -0.0001 ? ' ▶' : ''
        panLabel.textContent = formatPanRate(yearsPerSec) + dir
        panLabel.style.left = `${ox}px`
        panLabel.style.top = `${oy}px`

        // Build SVG: origin dot, line, spaced dots
        let svg = `<circle cx="${ox}" cy="${oy}" r="4" fill="#888" stroke="#444" stroke-width="1"/>`
        if (dist > 5) {
          svg += `<line x1="${ox}" y1="${oy}" x2="${cx}" y2="${cy}" stroke="#555" stroke-width="1" stroke-dasharray="4,3"/>`
          const nx = ddx / dist, ny = ddy / dist
          let pos = 8
          let gap = 8
          while (pos < dist - 4) {
            const dotX = ox + nx * pos, dotY = oy + ny * pos
            svg += `<circle cx="${dotX}" cy="${dotY}" r="2" fill="rgba(200,200,200,0.5)"/>`
            pos += gap
            gap += 2
          }
          svg += `<circle cx="${cx}" cy="${cy}" r="3" fill="#aaa" stroke="#555" stroke-width="1"/>`
        }
        panSvg.innerHTML = svg
      }

      canvas.addEventListener('pointerdown', (e) => {
        if (e.button !== 2) return
        dragging = true; focusDragging = true
        dragOriginX = e.clientX; dragOriginY = e.clientY
        dragCurX = e.clientX; dragCurY = e.clientY; dragDisplacement = 0
        canvas.requestPointerLock()
        e.preventDefault()
      })
      canvas.addEventListener('pointermove', (e) => {
        if (!dragging) return
        dragCurX += e.movementX; dragCurY += e.movementY
        const dy = (dragCurY - dragOriginY) * 0.01   // up = forward
        const dx = -(dragCurX - dragOriginX) * 0.005  // right = forward at half speed
        dragDisplacement = dy + dx
        updatePanWidget()
      })
      const stopDrag = () => {
        if (dragging) {
          panInertia = dragDisplacement
          if (document.pointerLockElement === canvas) document.exitPointerLock()
        }
        dragging = false; focusDragging = false; updatePanWidget()
      }
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
      levelPanel.style.cssText = 'position:absolute;left:20px;top:20px;background:rgba(17,17,17,0.85);border:1px solid #444;border-radius:6px;padding:8px;font-family:monospace;font-size:12px;color:#ccc;cursor:move;user-select:none;z-index:10'
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

      // ── Date mode toggle ─────────────────────────────────────────────────
      let dateMode: 'absolute' | 'relative' = 'absolute'
      const dateModeDiv = document.createElement('div')
      dateModeDiv.style.cssText = 'margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #333'
      const dateModeBtn = document.createElement('div')
      dateModeBtn.style.cssText = 'text-align:center;padding:4px 8px;border-radius:3px;cursor:pointer;background:rgba(255,255,255,0.08);color:#ccc;font-size:11px;font-family:monospace;transition:background 0.15s;user-select:none'
      dateModeBtn.addEventListener('click', (e) => { e.stopPropagation(); dateMode = dateMode === 'absolute' ? 'relative' : 'absolute'; dateModeBtn.textContent = dateMode === 'absolute' ? 'Dates: absolute' : 'Dates: relative' })
      dateModeBtn.addEventListener('mouseenter', () => { dateModeBtn.style.background = 'rgba(255,255,255,0.2)' })
      dateModeBtn.addEventListener('mouseleave', () => { dateModeBtn.style.background = 'rgba(255,255,255,0.08)' })
      dateModeBtn.textContent = 'Dates: absolute'
      dateModeDiv.appendChild(dateModeBtn)
      levelPanel.appendChild(dateModeDiv)

      // ── Custom events ────────────────────────────────────────────────────
      const customDiv = document.createElement('div')
      customDiv.style.cssText = 'margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #333'
      const customBtnRow = document.createElement('div')
      customBtnRow.style.cssText = 'display:flex;gap:4px'
      const customToggleBtn = document.createElement('div')
      customToggleBtn.textContent = 'Custom events'
      customToggleBtn.style.cssText = 'flex:1;text-align:center;padding:4px 8px;border-radius:3px;cursor:pointer;background:rgba(255,255,255,0.08);color:#ccc;font-size:11px;font-family:monospace;transition:background 0.15s;user-select:none'
      customToggleBtn.addEventListener('mouseenter', () => { customToggleBtn.style.background = 'rgba(255,255,255,0.2)' })
      customToggleBtn.addEventListener('mouseleave', () => { customToggleBtn.style.background = 'rgba(255,255,255,0.08)' })
      const copySchemaBtn = document.createElement('div')
      copySchemaBtn.textContent = 'Schema'
      copySchemaBtn.style.cssText = 'padding:4px 8px;border-radius:3px;cursor:pointer;background:rgba(255,255,255,0.08);color:#888;font-size:11px;font-family:monospace;transition:background 0.15s;user-select:none;min-width:52px;text-align:center'
      copySchemaBtn.addEventListener('mouseenter', () => { copySchemaBtn.style.background = 'rgba(255,255,255,0.2)' })
      copySchemaBtn.addEventListener('mouseleave', () => { copySchemaBtn.style.background = 'rgba(255,255,255,0.08)' })
      // Set fixed width from initial content before any text changes
      requestAnimationFrame(() => { copySchemaBtn.style.width = `${copySchemaBtn.offsetWidth}px` })
      copySchemaBtn.addEventListener('click', (e) => {
        e.stopPropagation()
        navigator.clipboard.writeText(EVENT_SCHEMA).then(() => {
          copySchemaBtn.textContent = 'Copied!'
          setTimeout(() => { copySchemaBtn.textContent = 'Schema' }, 1500)
        })
      })
      customBtnRow.appendChild(customToggleBtn)
      customBtnRow.appendChild(copySchemaBtn)
      customDiv.appendChild(customBtnRow)

      const customPane = document.createElement('div')
      customPane.style.cssText = 'display:none;margin-top:6px'
      const customTextarea = document.createElement('textarea')
      customTextarea.style.cssText = 'width:100%;height:120px;background:rgba(0,0,0,0.4);border:1px solid #555;border-radius:3px;color:#ccc;font-family:monospace;font-size:10px;padding:4px;resize:vertical;box-sizing:border-box'
      customTextarea.placeholder = 'Paste event JSON here...'
      const customSubmitRow = document.createElement('div')
      customSubmitRow.style.cssText = 'display:flex;gap:4px;margin-top:4px'
      const submitBtnStyle = 'flex:1;text-align:center;padding:4px 8px;border-radius:3px;cursor:pointer;font-size:11px;font-family:monospace;transition:background 0.15s;user-select:none'
      const addEventsBtn = document.createElement('div')
      addEventsBtn.textContent = 'Add'
      addEventsBtn.style.cssText = submitBtnStyle + ';background:rgba(100,200,100,0.15);color:#6bcb77'
      addEventsBtn.addEventListener('mouseenter', () => { addEventsBtn.style.background = 'rgba(100,200,100,0.3)' })
      addEventsBtn.addEventListener('mouseleave', () => { addEventsBtn.style.background = 'rgba(100,200,100,0.15)' })
      const replaceEventsBtn = document.createElement('div')
      replaceEventsBtn.textContent = 'Replace'
      replaceEventsBtn.style.cssText = submitBtnStyle + ';background:rgba(200,100,100,0.15);color:#e74c3c'
      replaceEventsBtn.addEventListener('mouseenter', () => { replaceEventsBtn.style.background = 'rgba(200,100,100,0.3)' })
      replaceEventsBtn.addEventListener('mouseleave', () => { replaceEventsBtn.style.background = 'rgba(200,100,100,0.15)' })
      const customStatus = document.createElement('div')
      customStatus.style.cssText = 'font-size:10px;color:#e74c3c;margin-top:4px;display:none'
      function handleCustomSubmit(replace: boolean) {
        const result = parseEventJson(customTextarea.value)
        if (typeof result === 'string') { customStatus.textContent = result; customStatus.style.display = 'block'; return }
        setEvents(replace ? result : [...events, ...result])
        customStatus.style.display = 'none'; customPane.style.display = 'none'
        customToggleBtn.textContent = `Custom events (${events.length})`
      }
      addEventsBtn.addEventListener('click', (e) => { e.stopPropagation(); handleCustomSubmit(false) })
      replaceEventsBtn.addEventListener('click', (e) => { e.stopPropagation(); handleCustomSubmit(true) })
      customSubmitRow.appendChild(addEventsBtn)
      customSubmitRow.appendChild(replaceEventsBtn)
      customPane.appendChild(customTextarea)
      customPane.appendChild(customSubmitRow)
      customPane.appendChild(customStatus)
      customDiv.appendChild(customPane)
      levelPanel.appendChild(customDiv)

      customToggleBtn.addEventListener('click', (e) => {
        e.stopPropagation()
        customPane.style.display = customPane.style.display === 'none' ? 'block' : 'none'
      })

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

      // ── Minimap timeline ─────────────────────────────────────────────────
      const minimap = document.createElement('div')
      minimap.style.cssText = 'position:absolute;bottom:8px;left:50%;transform:translateX(-50%);width:90%;height:28px;background:rgba(17,17,17,0.85);border:1px solid #444;border-radius:4px;z-index:10;overflow:visible'
      const miniCanvas = document.createElement('canvas')
      miniCanvas.style.cssText = 'width:100%;height:100%;cursor:default'
      minimap.appendChild(miniCanvas)
      const miniTooltip = document.createElement('div')
      miniTooltip.style.cssText = 'position:absolute;bottom:100%;left:0;margin-bottom:4px;font-family:monospace;font-size:11px;padding:2px 6px;background:rgba(17,17,17,0.95);border:1px solid #444;border-radius:4px;white-space:nowrap;pointer-events:none;opacity:0;transition:opacity 0.1s;transform:translateX(-50%)'
      minimap.appendChild(miniTooltip)
      // Hit-test data: updated each frame
      const miniDots: { px: number; py: number; ev: CoilEvent }[] = []
      let miniHovered: CoilEvent | null = null
      minimap.addEventListener('mousemove', (e) => {
        const rect = minimap.getBoundingClientRect()
        const mx = (e.clientX - rect.left) * devicePixelRatio
        const my = (e.clientY - rect.top) * devicePixelRatio
        let best: CoilEvent | null = null, bestDist = 12 * devicePixelRatio
        for (const d of miniDots) {
          const dx = mx - d.px, dy = my - d.py
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < bestDist) { bestDist = dist; best = d.ev }
        }
        miniHovered = best
        miniCanvas.style.cursor = best ? 'pointer' : 'default'
        if (best) {
          miniTooltip.textContent = best.name
          miniTooltip.style.color = best.color
          miniTooltip.style.left = `${(e.clientX - rect.left)}px`
          miniTooltip.style.opacity = '1'
        } else {
          miniTooltip.style.opacity = '0'
        }
      })
      minimap.addEventListener('mouseleave', () => { miniHovered = null; miniTooltip.style.opacity = '0'; miniCanvas.style.cursor = 'default' })
      minimap.addEventListener('click', () => {
        if (miniHovered) {
          const evLevel = miniHovered.level
          const targetLevel = Math.min(evLevel, cameraLevel)
          animateToT(miniHovered.t, targetLevel)
        }
      })
      canvas.parentElement!.appendChild(minimap)

      const K_WARP = 3 // exponential compression factor

      function yearToMiniPx(year: number, focusYr: number, halfW: number, halfRange: number): number {
        const dy = year - focusYr
        return halfW + halfW * Math.asinh((dy / halfRange) * Math.sinh(K_WARP)) / K_WARP
      }

      function updateMinimap() {
        const rect = minimap.getBoundingClientRect()
        const w = Math.floor(rect.width * devicePixelRatio)
        const h = Math.floor(rect.height * devicePixelRatio)
        if (miniCanvas.width !== w || miniCanvas.height !== h) {
          miniCanvas.width = w; miniCanvas.height = h
        }
        const ctx = miniCanvas.getContext('2d')!
        ctx.clearRect(0, 0, w, h)
        const halfW = w / 2
        const focusYr = START_YEAR + params.focusT * TOTAL_YEARS
        const camLvl = Math.round(cameraLevel)
        // Symmetric sliding window: halfRange shrinks with zoom level
        const levelRanges = [TOTAL_YEARS / 2, 2000, 200, 5, 0.5, 0.01, 0.0002]
        const lo = Math.floor(cameraLevel), hi = Math.min(lo + 1, levelRanges.length - 1)
        const frac = cameraLevel - lo
        const halfRange = levelRanges[lo] + frac * (levelRanges[hi] - levelRanges[lo])
        const edgeYrL = focusYr - halfRange, edgeYrR = focusYr + halfRange

        // Tick marks: centuries always, decades if room
        ctx.strokeStyle = 'rgba(255,255,255,0.12)'
        ctx.lineWidth = 1
        const firstCentury = Math.ceil(Math.max(START_YEAR, edgeYrL) / 100) * 100
        const lastCentury = Math.min(END_YEAR, edgeYrR)
        for (let yr = firstCentury; yr <= lastCentury; yr += 100) {
          const px = yearToMiniPx(yr, focusYr, halfW, halfRange)
          if (px < 0 || px > w) continue
          ctx.beginPath(); ctx.moveTo(px, h * 0.3); ctx.lineTo(px, h); ctx.stroke()
        }
        if (camLvl >= 1) {
          ctx.strokeStyle = 'rgba(255,255,255,0.06)'
          const firstDecade = Math.ceil(Math.max(START_YEAR, edgeYrL) / 10) * 10
          const lastDecade = Math.min(END_YEAR, edgeYrR)
          for (let yr = firstDecade; yr <= lastDecade; yr += 10) {
            if (yr % 100 === 0) continue
            const px = yearToMiniPx(yr, focusYr, halfW, halfRange)
            if (px < 0 || px > w) continue
            const prevPx = yearToMiniPx(yr - 10, focusYr, halfW, halfRange)
            if (Math.abs(px - prevPx) < 3) continue
            ctx.beginPath(); ctx.moveTo(px, h * 0.55); ctx.lineTo(px, h); ctx.stroke()
          }
        }

        // Events
        miniDots.length = 0
        for (const ev of events) {
          const evYr = START_YEAR + ev.t * TOTAL_YEARS
          const px = yearToMiniPx(evYr, focusYr, halfW, halfRange)
          if (px < -2 || px > w + 2) continue
          const r = ev.level === 0 ? 3 : ev.level === 1 ? 2 : 1.5
          ctx.fillStyle = ev.color
          ctx.globalAlpha = 0.9
          const bandCenter = 0.25 + ev.level * (0.5 / 3)
          const bandHalf = 0.06
          const hash = Math.sin(ev.t * 127.1 + ev.level * 311.7) * 43758.5453 % 1
          const dotY = h * (bandCenter + (hash - 0.5) * bandHalf * 2)
          ctx.beginPath(); ctx.arc(px, dotY, r * devicePixelRatio, 0, Math.PI * 2); ctx.fill()
          miniDots.push({ px, py: dotY, ev })
        }
        ctx.globalAlpha = 1

        // Helix boundary indicators
        const startPx = yearToMiniPx(START_YEAR, focusYr, halfW, halfRange)
        const endPx = yearToMiniPx(END_YEAR, focusYr, halfW, halfRange)
        ctx.strokeStyle = 'rgba(255,255,255,0.4)'
        ctx.lineWidth = 1.5 * devicePixelRatio
        if (startPx > 0 && startPx < w) {
          ctx.beginPath(); ctx.moveTo(startPx, 0); ctx.lineTo(startPx, h); ctx.stroke()
        }
        if (endPx > 0 && endPx < w) {
          ctx.beginPath(); ctx.moveTo(endPx, 0); ctx.lineTo(endPx, h); ctx.stroke()
        }

        // Focus indicator (center line)
        ctx.strokeStyle = '#ff4444'
        ctx.lineWidth = 2 * devicePixelRatio
        ctx.beginPath(); ctx.moveTo(halfW, 0); ctx.lineTo(halfW, h); ctx.stroke()

        // Edge year labels (show actual window edges, not clamped)
        ctx.fillStyle = 'rgba(255,255,255,0.3)'
        ctx.font = `${9 * devicePixelRatio}px monospace`
        ctx.textBaseline = 'middle'
        ctx.textAlign = 'left'
        ctx.fillText(yearLabel(Math.round(edgeYrL)), 4 * devicePixelRatio, h * 0.45)
        ctx.textAlign = 'right'
        ctx.fillText(yearLabel(Math.round(edgeYrR)), w - 4 * devicePixelRatio, h * 0.45)
      }

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
          animDuration = 400 + levelDist * 250
        } else {
          const logDist = Math.log10(Math.max(1, temporalDistYears))
          animDuration = 600 + logDist * 500 + levelDist * 200
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
        } else if (!dragging && Math.abs(panInertia) > 0.0001) {
          const totals = getCachedTotals()
          const panLvl = Math.min(Math.round(cameraLevel), totals.length - 1)
          params.focusT = THREE.MathUtils.clamp(params.focusT - panInertia * params.panSpeed / totals[panLvl], 0, 1)
          panInertia *= 0.92
          updateFocus(rt)
        }

        // Smooth scroll zoom: lerp cameraLevel toward zoomTarget
        if (!animating) {
          const diff = zoomTarget - cameraLevel
          if (Math.abs(diff) > 0.001) {
            cameraLevel += diff * 0.12
            updateFocus(rt)
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

        // ── Events (always write line segments; collect labels for de-overlap) ─
        const eventLabels: { scrX: number; scrY: number; coilX: number; coilY: number; opacity: number; ev: CoilEvent; anchorY: number; right: boolean }[] = []
        for (let i = 0; i < events.length; i++) {
          const ev = events[i]
          const evYear = START_YEAR + ev.t * TOTAL_YEARS
          const evWindow = eventWindowYears(ev.level)
          const temporalDist = Math.abs(evYear - focusYear)
          if (evWindow !== Infinity && temporalDist > evWindow) continue
          if (markerIdx >= MAX_MARKERS) break

          evalCoil(ev.t, ev.level, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _mp)
          const sx = _mp.x - origin.x, sy = _mp.y - origin.y, sz = _mp.z - origin.z

          const fadeStart = evWindow === Infinity ? Infinity : evWindow * 0.7
          const opacity = evWindow === Infinity ? 1.0
            : temporalDist < fadeStart ? 1.0
            : 1.0 - (temporalDist - fadeStart) / (evWindow - fadeStart)

          // Project coil point
          _projVec.set(sx, sy, sz).project(rt.camera)
          if (_projVec.z > 1) continue
          const coilX = (_projVec.x * 0.5 + 0.5) * sw
          const coilY = (-_projVec.y * 0.5 + 0.5) * sh
          if (coilX < -200 || coilX > sw + 200 || coilY < -50 || coilY > sh + 50) continue

          // Project label anchor (offset from coil)
          const labelOff = viewDistForLevel(cameraLevel) * 0.08
          const side = ev.level >= 3 ? 1 : -1
          const ex = sx + side * labelOff * _camRight.x, ey = sy + side * labelOff * _camRight.y, ez = sz + side * labelOff * _camRight.z
          _projVec.set(ex, ey, ez).project(rt.camera)
          const scrX = (_projVec.x * 0.5 + 0.5) * sw
          const scrY = (-_projVec.y * 0.5 + 0.5) * sh

          eventLabels.push({ scrX, scrY, coilX, coilY, opacity, ev, anchorY: scrY, right: ev.level >= 3 })
        }

        // De-overlap: sort by screen Y, push overlapping labels downward
        eventLabels.sort((a, b) => a.scrY - b.scrY)
        const MIN_GAP = 16
        for (let i = 1; i < eventLabels.length; i++) {
          const prev = eventLabels[i - 1], cur = eventLabels[i]
          if (Math.abs(cur.scrX - prev.scrX) > 200) continue
          if (cur.scrY - prev.scrY < MIN_GAP) {
            cur.scrY = prev.scrY + MIN_GAP
          }
        }

        // Assign to pool
        for (const lbl of eventLabels) {
          if (poolIdx >= POOL_SIZE) break
          const div = labelPool[poolIdx++]
          const dateStr = dateMode === 'relative' ? relativeDateStr(lbl.ev.t, params.focusT) : eventDateStr(lbl.ev.t)
          const c = new THREE.Color(lbl.ev.color).lerp(new THREE.Color(1, 1, 1), 0.4)
          div.innerHTML = `${lbl.ev.name} <span style="color:${c.getStyle()}">${dateStr}</span>`
          div.style.display = 'block'
          div.style.left = `${lbl.scrX}px`
          div.style.top = `${lbl.scrY}px`
          div.style.fontSize = '14px'
          div.style.opacity = String(Math.max(0.3, lbl.opacity))
          div.style.color = lbl.ev.color
          div.style.fontWeight = 'bold'
          div.style.textAlign = lbl.right ? 'left' : 'right'
          div.style.transform = lbl.right ? 'translateY(-100%)' : 'translate(-100%, -100%)'
          div.style.pointerEvents = 'auto'
          div.style.cursor = 'pointer'
          div.dataset.eventT = String(lbl.ev.t)
          div.dataset.eventLevel = String(lbl.ev.level)
        }

        // ── Event connectors (dots on coil + lines to labels) ─────────────
        {
          const dpr = devicePixelRatio
          const cw = Math.floor(sw * dpr), ch = Math.floor(sh * dpr)
          if (connCanvas.width !== cw || connCanvas.height !== ch) {
            connCanvas.width = cw; connCanvas.height = ch
          }
          const ctx = connCanvas.getContext('2d')!
          ctx.clearRect(0, 0, cw, ch)
          for (const lbl of eventLabels) {
            if (lbl.opacity < 0.05) continue
            const cx = lbl.coilX * dpr, cy = lbl.coilY * dpr
            const lx = lbl.scrX * dpr, ly = lbl.scrY * dpr
            ctx.globalAlpha = Math.max(0.3, lbl.opacity) * 0.5
            // Connector line
            ctx.strokeStyle = lbl.ev.color
            ctx.lineWidth = 1 * dpr
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(lx, ly); ctx.stroke()
            // Dot on coil
            ctx.fillStyle = lbl.ev.color
            ctx.globalAlpha = Math.max(0.3, lbl.opacity)
            ctx.beginPath(); ctx.arc(cx, cy, 3 * dpr, 0, Math.PI * 2); ctx.fill()
          }
          ctx.globalAlpha = 1
        }

        // ── Focus indicator ────────────────────────────────────────────────
        {
          // Use _pos from cameraFrame (already interpolated for fractional levels)
          cameraFrame(params.focusT)
          const fx = _pos.x - origin.x, fy = _pos.y - origin.y, fz = _pos.z - origin.z
          // Project coil point, then offset right in screen space
          _projVec.set(fx, fy, fz).project(rt.camera)
          const fScrX = (_projVec.x * 0.5 + 0.5) * sw + 20
          const fScrY = (-_projVec.y * 0.5 + 0.5) * sh
          if (chevronX < 0) { chevronX = fScrX; chevronY = fScrY }
          chevronX += (fScrX - chevronX) * 0.15
          chevronY += (fScrY - chevronY) * 0.15
          focusChevron.style.left = `${chevronX}px`
          focusChevron.style.top = `${chevronY}px`
          const labelY = Math.max(20, Math.min(sh - 20, fScrY))
          focusLabelDiv.textContent = focusTToDate(params.focusT)
          focusLabelDiv.style.top = `${labelY}px`
        }

        markerGeom.setDrawRange(0, markerIdx * 2)
        markerPosAttr.needsUpdate = true
        markerColAttr.needsUpdate = true

        updateMinimap()
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
