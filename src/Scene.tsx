import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { WebGPURenderer, MeshStandardNodeMaterial } from 'three/webgpu'
import { uv, uniform, uniformArray, float, floor, fract, uint, Fn } from 'three/tsl'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Pane } from 'tweakpane'

const SEGMENTS_PER_TURN = 12
const DAYS_PER_YEAR = 365
const DAY_SEGS_PER_TURN = 12
const HOURS_PER_DAY = 24
const HOUR_SEGS_PER_TURN = 8
const MINS_PER_HOUR = 60
const MIN_SEGS_PER_TURN = 8
const SECS_PER_MIN = 60
const SEC_SEGS_PER_TURN = 32
const SEC_WINDOW = 120 // ±60 seconds shown

const paletteColors = [
  new THREE.Color('#e63946'),
  new THREE.Color('#f4a261'),
  new THREE.Color('#e9c46a'),
  new THREE.Color('#2a9d8f'),
  new THREE.Color('#457b9d'),
  new THREE.Color('#1d3557'),
  new THREE.Color('#a8dadc'),
  new THREE.Color('#6a4c93'),
  new THREE.Color('#f72585'),
  new THREE.Color('#4cc9f0'),
  new THREE.Color('#80b918'),
  new THREE.Color('#ffffff'),
]

const colorPalette = uniformArray(paletteColors)
const numTurnsUniform = uniform(5)

const coilColorNode = Fn(() => {
  const turnFraction = fract(uv().x.mul(numTurnsUniform))
  const segmentIndex = uint(floor(turnFraction.mul(float(SEGMENTS_PER_TURN))))
  return colorPalette.element(segmentIndex)
})()

class HelixCurve extends THREE.Curve<THREE.Vector3> {
  turns: number
  radius: number
  length: number

  constructor(turns: number, radius: number, length: number) {
    super()
    this.turns = turns
    this.radius = radius
    this.length = length
  }

  getPoint(t: number, target = new THREE.Vector3()) {
    const angle = t * this.turns * Math.PI * 2
    return target.set(
      Math.cos(angle) * this.radius,
      (t - 0.5) * this.length,
      Math.sin(angle) * this.radius,
    )
  }
}

// Writes a day-coil position into arr[j..j+2] using the analytical helix Frenet frame.
function writeDayPos(
  t: number, omega: number, R: number, L: number, tMag: number,
  totalDayTurns: number, dayOff: number,
  arr: Float32Array, j: number,
) {
  const theta = t * omega
  const cT = Math.cos(theta), sT = Math.sin(theta)
  const px = cT * R, py = (t - 0.5) * L, pz = sT * R
  // Frenet normal (toward axis), ny=0
  const nx = -cT, nz = -sT
  // Normalized tangent
  const tx = -sT * R * omega / tMag, ty = L / tMag, tz = cT * R * omega / tMag
  // Binormal = T x N  (ny=0 simplifies)
  const bx = ty * nz, by = tz * nx - tx * nz, bz = -ty * nx
  // Day-coil offset
  const dA = t * totalDayTurns * Math.PI * 2
  const cD = Math.cos(dA), sD = Math.sin(dA)
  arr[j]     = px + dayOff * (cD * nx + sD * bx)
  arr[j + 1] = py + dayOff * sD * by
  arr[j + 2] = pz + dayOff * (cD * nz + sD * bz)
}

interface Runtime {
  renderer: WebGPURenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  coilMesh: THREE.Mesh | null
  dayCoilLine: THREE.Line | null
  hourCoilLine: THREE.Line | null
  minCoilLine: THREE.Line | null
  secCoilLine: THREE.Line | null
  minPosData: Float32Array | null
  totalMinSegs: number
  material: MeshStandardNodeMaterial
  lineMaterial: THREE.LineBasicMaterial
  pane: Pane
}

export default function Scene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const runtimeRef = useRef<Runtime | null>(null)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let disposed = false

    const params = {
      turns: 2,
      tubeRadius: 0.08,
      coilRadius: 1,
      turnSpacing: 5,
      outerOffset: 0.05,
      hourOffset: 0.01,
      minOffset: 0.003,
      secOffset: 0.0003,
      panSpeed: 10,
      focusT: 0.5,
    }

    // Compute helix position + Frenet frame at parameter t
    const _helixPos = new THREE.Vector3()
    const _helixN = new THREE.Vector3()
    const _helixB = new THREE.Vector3()
    const _helixT = new THREE.Vector3()

    function helixFrame(t: number) {
      const numTurns = Math.max(1, params.turns)
      const R = params.coilRadius
      const L = numTurns * params.turnSpacing
      const omega = numTurns * Math.PI * 2
      const theta = t * omega
      const cT = Math.cos(theta), sT = Math.sin(theta)

      _helixPos.set(cT * R, (t - 0.5) * L, sT * R)
      _helixN.set(-cT, 0, -sT) // toward axis
      const tMag = Math.sqrt(R * R * omega * omega + L * L)
      _helixT.set(-sT * R * omega / tMag, L / tMag, cT * R * omega / tMag)
      _helixB.crossVectors(_helixT, _helixN)
    }

    let prevFocusT = params.focusT
    const _offset = new THREE.Vector3()

    // Rotated frame: N and B rotated by the day-coil angle so the camera
    // follows the day coil's winding around the tube.
    const _rN = new THREE.Vector3()
    const _rB = new THREE.Vector3()

    function rotatedFrame(t: number) {
      helixFrame(t)
      const numTurns = Math.max(1, params.turns)
      const totalDays = numTurns * DAYS_PER_YEAR
      const dayAngle = t * totalDays * Math.PI * 2
      const c = Math.cos(dayAngle), s = Math.sin(dayAngle)
      // Rotate N,B around T by dayAngle
      _rN.copy(_helixN).multiplyScalar(c).addScaledVector(_helixB, s)
      _rB.copy(_helixN).multiplyScalar(-s).addScaledVector(_helixB, c)
    }

    function updateFocus(rt: Runtime) {
      const oldT = prevFocusT
      const newT = params.focusT
      prevFocusT = newT

      // Old frame: decompose camera offset into local coords
      rotatedFrame(oldT)
      _offset.copy(rt.camera.position).sub(_helixPos)
      const lx = _offset.dot(_rN)
      const ly = _offset.dot(_rB)
      const lz = _offset.dot(_helixT)

      // New frame: reconstruct camera position
      rotatedFrame(newT)
      rt.camera.position.copy(_helixPos)
        .addScaledVector(_rN, lx)
        .addScaledVector(_rB, ly)
        .addScaledVector(_helixT, lz)

      rt.controls.target.copy(_helixPos)

      rebuildSecondCoil(rt)
    }

    function rebuild(rt: Runtime) {
      if (rt.coilMesh) { rt.scene.remove(rt.coilMesh); rt.coilMesh.geometry.dispose() }
      if (rt.dayCoilLine) { rt.scene.remove(rt.dayCoilLine); rt.dayCoilLine.geometry.dispose() }
      if (rt.hourCoilLine) { rt.scene.remove(rt.hourCoilLine); rt.hourCoilLine.geometry.dispose() }
      if (rt.minCoilLine) { rt.scene.remove(rt.minCoilLine); rt.minCoilLine.geometry.dispose() }

      const numTurns = Math.max(1, params.turns)
      const length = numTurns * params.turnSpacing
      numTurnsUniform.value = numTurns
      const R = params.coilRadius
      const L = length
      const omega = numTurns * Math.PI * 2
      const tMag = Math.sqrt(R * R * omega * omega + L * L)
      const totalDayTurns = numTurns * DAYS_PER_YEAR
      const dayOff = params.tubeRadius + params.outerOffset
      const tmpColor = new THREE.Color()

      // ====== BIG COIL (tube) ======
      const curve = new HelixCurve(numTurns, R, L)
      const tubSegs = Math.ceil(numTurns * SEGMENTS_PER_TURN) * 8
      rt.coilMesh = new THREE.Mesh(
        new THREE.TubeGeometry(curve, tubSegs, params.tubeRadius, 12, false),
        rt.material,
      )
      rt.scene.add(rt.coilMesh)

      // ====== DAY COIL (line) ======
      const totalDaySegs = totalDayTurns * DAY_SEGS_PER_TURN
      const totalDayPts = totalDaySegs + 1
      const dayPos = new Float32Array(totalDayPts * 3)
      const dayCol = new Float32Array(totalDayPts * 3)

      for (let i = 0; i < totalDayPts; i++) {
        const t = i / totalDaySegs
        const j = i * 3
        writeDayPos(t, omega, R, L, tMag, totalDayTurns, dayOff, dayPos, j)
        const hue = (t * numTurns) % 1
        tmpColor.setHSL(hue, 1, 0.5)
        dayCol[j] = tmpColor.r; dayCol[j + 1] = tmpColor.g; dayCol[j + 2] = tmpColor.b
      }

      const dayGeom = new THREE.BufferGeometry()
      dayGeom.setAttribute('position', new THREE.BufferAttribute(dayPos, 3))
      dayGeom.setAttribute('color', new THREE.BufferAttribute(dayCol, 3))
      rt.dayCoilLine = new THREE.Line(dayGeom, rt.lineMaterial)
      rt.scene.add(rt.dayCoilLine)

      // ====== HOUR COIL (line) ======
      const totalHourTurns = totalDayTurns * HOURS_PER_DAY
      const totalHourSegs = totalHourTurns * HOUR_SEGS_PER_TURN
      const totalHourPts = totalHourSegs + 1

      // Pass 1: day-coil positions at hour resolution
      const dayHR = new Float32Array(totalHourPts * 3)
      for (let i = 0; i < totalHourPts; i++) {
        writeDayPos(i / totalHourSegs, omega, R, L, tMag, totalDayTurns, dayOff, dayHR, i * 3)
      }

      // Pass 2: parallel-transport frame along day-coil path + hour offsets
      const hourPos = new Float32Array(totalHourPts * 3)
      const hourCol = new Float32Array(totalHourPts * 3)

      const prevTan = new THREE.Vector3()
      const curTan = new THREE.Vector3()
      const dN = new THREE.Vector3()
      const dB = new THREE.Vector3()
      const crossV = new THREE.Vector3()
      const rotMat = new THREE.Matrix4()
      const hOff = params.hourOffset

      // Seed tangent from first two day-coil points
      curTan.set(dayHR[3] - dayHR[0], dayHR[4] - dayHR[1], dayHR[5] - dayHR[2]).normalize()

      // Seed normal perpendicular to tangent
      const ref = Math.abs(curTan.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)
      dN.crossVectors(ref, curTan).normalize()
      dB.crossVectors(curTan, dN)

      // Point 0: hourAngle = 0 → cos=1, sin=0
      hourPos[0] = dayHR[0] + hOff * dN.x
      hourPos[1] = dayHR[1] + hOff * dN.y
      hourPos[2] = dayHR[2] + hOff * dN.z
      tmpColor.setHSL(0, 1, 0.5)
      hourCol[0] = tmpColor.r; hourCol[1] = tmpColor.g; hourCol[2] = tmpColor.b

      prevTan.copy(curTan)

      for (let i = 1; i < totalHourPts; i++) {
        const t = i / totalHourSegs

        // Tangent via central finite diff
        const pi = Math.max(0, i - 1) * 3
        const ni = Math.min(totalHourPts - 1, i + 1) * 3
        curTan.set(dayHR[ni] - dayHR[pi], dayHR[ni + 1] - dayHR[pi + 1], dayHR[ni + 2] - dayHR[pi + 2]).normalize()

        // Parallel transport: rotate normal to match new tangent
        crossV.crossVectors(prevTan, curTan)
        const cl = crossV.length()
        if (cl > 1e-10) {
          crossV.divideScalar(cl)
          const ang = Math.acos(THREE.MathUtils.clamp(prevTan.dot(curTan), -1, 1))
          dN.applyMatrix4(rotMat.makeRotationAxis(crossV, ang))
        }
        dB.crossVectors(curTan, dN)

        // Hour-coil offset
        const hA = t * totalHourTurns * Math.PI * 2
        const cH = Math.cos(hA), sH = Math.sin(hA)
        const j = i * 3
        hourPos[j]     = dayHR[j]     + hOff * (cH * dN.x + sH * dB.x)
        hourPos[j + 1] = dayHR[j + 1] + hOff * (cH * dN.y + sH * dB.y)
        hourPos[j + 2] = dayHR[j + 2] + hOff * (cH * dN.z + sH * dB.z)

        // Hue cycles once per day
        const dayFrac = (t * totalDayTurns) % 1
        tmpColor.setHSL(dayFrac, 1, 0.5)
        hourCol[j] = tmpColor.r; hourCol[j + 1] = tmpColor.g; hourCol[j + 2] = tmpColor.b

        prevTan.copy(curTan)
      }

      const hourGeom = new THREE.BufferGeometry()
      hourGeom.setAttribute('position', new THREE.BufferAttribute(hourPos, 3))
      hourGeom.setAttribute('color', new THREE.BufferAttribute(hourCol, 3))
      rt.hourCoilLine = new THREE.Line(hourGeom, rt.lineMaterial)
      rt.scene.add(rt.hourCoilLine)

      // ====== MINUTE COIL (line) ======
      const totalMinTurns = totalHourTurns * MINS_PER_HOUR
      const totalMinSegs = totalMinTurns * MIN_SEGS_PER_TURN
      const totalMinPts = totalMinSegs + 1

      // Pass 1: hour-coil positions at minute resolution
      // We need to recompute the full chain (helix → day → hour) at minute resolution.
      // Day positions at minute resolution:
      const dayMR = new Float32Array(totalMinPts * 3)
      for (let i = 0; i < totalMinPts; i++) {
        writeDayPos(i / totalMinSegs, omega, R, L, tMag, totalDayTurns, dayOff, dayMR, i * 3)
      }

      // Hour positions at minute resolution (parallel transport on day path):
      const hourMR = new Float32Array(totalMinPts * 3)

      curTan.set(dayMR[3] - dayMR[0], dayMR[4] - dayMR[1], dayMR[5] - dayMR[2]).normalize()
      const ref2 = Math.abs(curTan.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)
      dN.crossVectors(ref2, curTan).normalize()
      dB.crossVectors(curTan, dN)
      prevTan.copy(curTan)

      hourMR[0] = dayMR[0] + hOff * dN.x
      hourMR[1] = dayMR[1] + hOff * dN.y
      hourMR[2] = dayMR[2] + hOff * dN.z

      for (let i = 1; i < totalMinPts; i++) {
        const t = i / totalMinSegs
        const pi = Math.max(0, i - 1) * 3
        const ni = Math.min(totalMinPts - 1, i + 1) * 3
        curTan.set(dayMR[ni] - dayMR[pi], dayMR[ni + 1] - dayMR[pi + 1], dayMR[ni + 2] - dayMR[pi + 2]).normalize()

        crossV.crossVectors(prevTan, curTan)
        const cl2 = crossV.length()
        if (cl2 > 1e-10) {
          crossV.divideScalar(cl2)
          const ang2 = Math.acos(THREE.MathUtils.clamp(prevTan.dot(curTan), -1, 1))
          dN.applyMatrix4(rotMat.makeRotationAxis(crossV, ang2))
        }
        dB.crossVectors(curTan, dN)
        prevTan.copy(curTan)

        const hA2 = t * totalHourTurns * Math.PI * 2
        const cH2 = Math.cos(hA2), sH2 = Math.sin(hA2)
        const j = i * 3
        hourMR[j]     = dayMR[j]     + hOff * (cH2 * dN.x + sH2 * dB.x)
        hourMR[j + 1] = dayMR[j + 1] + hOff * (cH2 * dN.y + sH2 * dB.y)
        hourMR[j + 2] = dayMR[j + 2] + hOff * (cH2 * dN.z + sH2 * dB.z)
      }

      // Pass 2: parallel-transport frame along hour-coil path + minute offsets
      const minPos = new Float32Array(totalMinPts * 3)
      const minCol = new Float32Array(totalMinPts * 3)
      const mOff = params.minOffset

      curTan.set(hourMR[3] - hourMR[0], hourMR[4] - hourMR[1], hourMR[5] - hourMR[2]).normalize()
      const ref3 = Math.abs(curTan.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)
      dN.crossVectors(ref3, curTan).normalize()
      dB.crossVectors(curTan, dN)
      prevTan.copy(curTan)

      minPos[0] = hourMR[0] + mOff * dN.x
      minPos[1] = hourMR[1] + mOff * dN.y
      minPos[2] = hourMR[2] + mOff * dN.z
      tmpColor.setHSL(0, 1, 0.5)
      minCol[0] = tmpColor.r; minCol[1] = tmpColor.g; minCol[2] = tmpColor.b

      for (let i = 1; i < totalMinPts; i++) {
        const t = i / totalMinSegs
        const pi = Math.max(0, i - 1) * 3
        const ni = Math.min(totalMinPts - 1, i + 1) * 3
        curTan.set(hourMR[ni] - hourMR[pi], hourMR[ni + 1] - hourMR[pi + 1], hourMR[ni + 2] - hourMR[pi + 2]).normalize()

        crossV.crossVectors(prevTan, curTan)
        const cl3 = crossV.length()
        if (cl3 > 1e-10) {
          crossV.divideScalar(cl3)
          const ang3 = Math.acos(THREE.MathUtils.clamp(prevTan.dot(curTan), -1, 1))
          dN.applyMatrix4(rotMat.makeRotationAxis(crossV, ang3))
        }
        dB.crossVectors(curTan, dN)
        prevTan.copy(curTan)

        const mA = t * totalMinTurns * Math.PI * 2
        const cM = Math.cos(mA), sM = Math.sin(mA)
        const j = i * 3
        minPos[j]     = hourMR[j]     + mOff * (cM * dN.x + sM * dB.x)
        minPos[j + 1] = hourMR[j + 1] + mOff * (cM * dN.y + sM * dB.y)
        minPos[j + 2] = hourMR[j + 2] + mOff * (cM * dN.z + sM * dB.z)

        // Hue cycles once per hour
        const hourFrac = (t * totalHourTurns) % 1
        tmpColor.setHSL(hourFrac, 1, 0.5)
        minCol[j] = tmpColor.r; minCol[j + 1] = tmpColor.g; minCol[j + 2] = tmpColor.b
      }

      const minGeom = new THREE.BufferGeometry()
      minGeom.setAttribute('position', new THREE.BufferAttribute(minPos, 3))
      minGeom.setAttribute('color', new THREE.BufferAttribute(minCol, 3))
      rt.minCoilLine = new THREE.Line(minGeom, rt.lineMaterial)
      rt.scene.add(rt.minCoilLine)

      // Store for second coil to sample from
      rt.minPosData = minPos
      rt.totalMinSegs = totalMinSegs

      rebuildSecondCoil(rt)
    }

    function rebuildSecondCoil(rt: Runtime) {
      if (rt.secCoilLine) { rt.scene.remove(rt.secCoilLine); rt.secCoilLine.geometry.dispose(); rt.secCoilLine = null }
      if (!rt.minPosData || rt.totalMinSegs < 2) return

      const numTurns = Math.max(1, params.turns)
      const totalDayTurns = numTurns * DAYS_PER_YEAR
      const totalHourTurns = totalDayTurns * HOURS_PER_DAY
      const totalMinTurns = totalHourTurns * MINS_PER_HOUR
      const totalSecTurns = totalMinTurns * SECS_PER_MIN
      const sOff = params.secOffset

      // Window: ±60 seconds centered on focusT
      const halfT = (SEC_WINDOW / 2) / totalSecTurns
      const tS = Math.max(0, params.focusT - halfT)
      const tE = Math.min(1, params.focusT + halfT)
      const tR = tE - tS
      const secTurnsInWindow = tR * totalSecTurns
      const totalSecSegs = Math.ceil(secTurnsInWindow * SEC_SEGS_PER_TURN)
      const totalSecPts = totalSecSegs + 1
      if (totalSecPts < 2) return

      const tmpColor = new THREE.Color()
      const prevTan = new THREE.Vector3()
      const curTan = new THREE.Vector3()
      const dN = new THREE.Vector3()
      const dB = new THREE.Vector3()
      const crossV = new THREE.Vector3()
      const rotMat = new THREE.Matrix4()

      // Sample the ACTUAL stored minute coil positions at second resolution
      // by linearly interpolating from rt.minPosData
      const minSR = new Float32Array(totalSecPts * 3)
      const mp = rt.minPosData
      const mSegs = rt.totalMinSegs

      for (let i = 0; i < totalSecPts; i++) {
        const t = tS + tR * i / (totalSecPts - 1)
        const fi = t * mSegs // floating index into minPos
        const i0 = Math.min(Math.floor(fi), mSegs)
        const i1 = Math.min(i0 + 1, mSegs)
        const frac = fi - i0
        const a = i0 * 3, b = i1 * 3, j = i * 3
        minSR[j]     = mp[a]     + frac * (mp[b]     - mp[a])
        minSR[j + 1] = mp[a + 1] + frac * (mp[b + 1] - mp[a + 1])
        minSR[j + 2] = mp[a + 2] + frac * (mp[b + 2] - mp[a + 2])
      }

      // Parallel transport on the sampled minute positions → wrap seconds
      const secPos = new Float32Array(totalSecPts * 3)
      const secCol = new Float32Array(totalSecPts * 3)

      curTan.set(minSR[3] - minSR[0], minSR[4] - minSR[1], minSR[5] - minSR[2]).normalize()
      const ref = Math.abs(curTan.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)
      dN.crossVectors(ref, curTan).normalize()
      dB.crossVectors(curTan, dN)
      prevTan.copy(curTan)

      const sA0 = tS * totalSecTurns * Math.PI * 2
      secPos[0] = minSR[0] + sOff * (Math.cos(sA0) * dN.x + Math.sin(sA0) * dB.x)
      secPos[1] = minSR[1] + sOff * (Math.cos(sA0) * dN.y + Math.sin(sA0) * dB.y)
      secPos[2] = minSR[2] + sOff * (Math.cos(sA0) * dN.z + Math.sin(sA0) * dB.z)
      const minFrac0 = (tS * totalMinTurns) % 1
      tmpColor.setHSL(minFrac0, 1, 0.5)
      secCol[0] = tmpColor.r; secCol[1] = tmpColor.g; secCol[2] = tmpColor.b

      for (let i = 1; i < totalSecPts; i++) {
        const t = tS + tR * i / (totalSecPts - 1)
        const pi = Math.max(0, i - 1) * 3
        const ni = Math.min(totalSecPts - 1, i + 1) * 3
        curTan.set(minSR[ni] - minSR[pi], minSR[ni + 1] - minSR[pi + 1], minSR[ni + 2] - minSR[pi + 2]).normalize()
        crossV.crossVectors(prevTan, curTan)
        const cl = crossV.length()
        if (cl > 1e-10) { crossV.divideScalar(cl); dN.applyMatrix4(rotMat.makeRotationAxis(crossV, Math.acos(THREE.MathUtils.clamp(prevTan.dot(curTan), -1, 1)))) }
        dB.crossVectors(curTan, dN)
        prevTan.copy(curTan)

        const sA = t * totalSecTurns * Math.PI * 2
        const cS = Math.cos(sA), sS = Math.sin(sA)
        const j = i * 3
        secPos[j]     = minSR[j]     + sOff * (cS * dN.x + sS * dB.x)
        secPos[j + 1] = minSR[j + 1] + sOff * (cS * dN.y + sS * dB.y)
        secPos[j + 2] = minSR[j + 2] + sOff * (cS * dN.z + sS * dB.z)

        const minFrac = (t * totalMinTurns) % 1
        tmpColor.setHSL(minFrac, 1, 0.5)
        secCol[j] = tmpColor.r; secCol[j + 1] = tmpColor.g; secCol[j + 2] = tmpColor.b
      }

      const secGeom = new THREE.BufferGeometry()
      secGeom.setAttribute('position', new THREE.BufferAttribute(secPos, 3))
      secGeom.setAttribute('color', new THREE.BufferAttribute(secCol, 3))
      rt.secCoilLine = new THREE.Line(secGeom, rt.lineMaterial)
      rt.scene.add(rt.secCoilLine)
    }

    const init = async () => {
      if (!navigator.gpu) {
        document.body.innerHTML =
          '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU is not supported in this browser.</div>'
        return
      }

      const width = window.innerWidth
      const height = window.innerHeight

      const renderer = new WebGPURenderer({ canvas, antialias: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      renderer.setSize(width, height, false)
      await renderer.init()

      const isWebGPU = (renderer as any).backend?.isWebGPUBackend
      if (!isWebGPU) {
        document.body.innerHTML =
          '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU backend failed to initialize.</div>'
        renderer.dispose()
        return
      }

      if (disposed) {
        renderer.dispose()
        return
      }

      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x111111)

      const camera = new THREE.PerspectiveCamera(60, width / height, 0.001, 200)

      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true
      controls.dampingFactor = 0.08
      controls.enablePan = false
      controls.minDistance = params.tubeRadius

      // Position camera relative to initial focus on helix
      rotatedFrame(params.focusT)
      controls.target.copy(_helixPos)
      // Offset: outward from rotated normal + slight binormal
      camera.position.copy(_helixPos)
        .addScaledVector(_rN, -2)
        .addScaledVector(_rB, 0.5)

      const dirLight = new THREE.DirectionalLight(0xffffff, 2)
      dirLight.position.set(3, 5, 4)
      scene.add(dirLight)

      const ambLight = new THREE.AmbientLight(0xffffff, 0.3)
      scene.add(ambLight)

      const material = new MeshStandardNodeMaterial({ roughness: 0.4, metalness: 0.1 })
      material.colorNode = coilColorNode

      const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true })

      const pane = new Pane({ title: 'Coil' })

      const rt: Runtime = {
        renderer, scene, camera, controls,
        coilMesh: null, dayCoilLine: null, hourCoilLine: null, minCoilLine: null, secCoilLine: null,
        minPosData: null, totalMinSegs: 0,
        material, lineMaterial, pane,
      }
      runtimeRef.current = rt

      rebuild(rt)
      updateFocus(rt)

      // Right-click drag → slide focus along the helix
      let dragging = false
      let dragStartX = 0
      let dragStartT = 0
      let focusDragging = false
      const focusBinding = pane.addBinding(params, 'focusT', { min: 0, max: 1, step: 0.0001, label: 'focus' })
      focusBinding.on('change', () => { if (!focusDragging) updateFocus(rt) })

      canvas.addEventListener('pointerdown', (e) => {
        if (e.button === 2) {
          dragging = true
          focusDragging = true
          dragStartX = e.clientX
          dragStartT = params.focusT
          e.preventDefault()
        }
      })
      canvas.addEventListener('pointermove', (e) => {
        if (!dragging) return
        const dx = e.clientX - dragStartX
        const numTurns = Math.max(1, params.turns)
        const totalSecs = numTurns * DAYS_PER_YEAR * HOURS_PER_DAY * MINS_PER_HOUR * SECS_PER_MIN
        const sensitivity = params.panSpeed / totalSecs
        params.focusT = THREE.MathUtils.clamp(dragStartT - dx * sensitivity, 0, 1)
        focusBinding.refresh()
        updateFocus(rt)
      })
      const stopDrag = () => { dragging = false; focusDragging = false }
      canvas.addEventListener('pointerup', stopDrag)
      canvas.addEventListener('pointercancel', stopDrag)
      canvas.addEventListener('contextmenu', (e) => e.preventDefault())

      pane.addBinding(params, 'turns', { min: 1, max: 20, step: 1 })
        .on('change', () => { rebuild(rt); updateFocus(rt) })
      pane.addBinding(params, 'tubeRadius', { min: 0.01, max: 0.5, step: 0.01, label: 'tube radius' })
        .on('change', () => { rebuild(rt); updateFocus(rt) })
      pane.addBinding(params, 'coilRadius', { min: 0.1, max: 5, step: 0.1, label: 'coil radius' })
        .on('change', () => { rebuild(rt); updateFocus(rt) })
      pane.addBinding(params, 'turnSpacing', { min: 0.1, max: 5, step: 0.1, label: 'turn spacing' })
        .on('change', () => { rebuild(rt); updateFocus(rt) })
      pane.addBinding(params, 'outerOffset', { min: 0.05, max: 1, step: 0.01, label: 'day offset' })
        .on('change', () => { rebuild(rt); updateFocus(rt) })
      pane.addBinding(params, 'hourOffset', { min: 0.001, max: 0.1, step: 0.001, label: 'hour offset' })
        .on('change', () => { rebuild(rt); updateFocus(rt) })
      pane.addBinding(params, 'minOffset', { min: 0.0005, max: 0.05, step: 0.0005, label: 'min offset' })
        .on('change', () => { rebuild(rt); updateFocus(rt) })
      pane.addBinding(params, 'secOffset', { min: 0.0001, max: 0.01, step: 0.0001, label: 'sec offset' })
        .on('change', () => rebuildSecondCoil(rt))
      pane.addBinding(params, 'panSpeed', { min: 1, max: 3600, step: 1, label: 'sec/pixel' })

      const animate = () => {
        if (disposed) return
        rafRef.current = requestAnimationFrame(animate)
        const r = runtimeRef.current!
        r.controls.update()
        r.renderer.render(r.scene, r.camera)
      }
      rafRef.current = requestAnimationFrame(animate)
    }

    init()

    const handleResize = () => {
      const rt = runtimeRef.current
      if (!rt) return
      const w = window.innerWidth
      const h = window.innerHeight
      rt.renderer.setSize(w, h, false)
      rt.camera.aspect = w / h
      rt.camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', handleResize)

    return () => {
      disposed = true
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(rafRef.current)
      const rt = runtimeRef.current
      if (rt) {
        rt.pane.dispose()
        rt.controls.dispose()
        if (rt.coilMesh) rt.coilMesh.geometry.dispose()
        if (rt.dayCoilLine) rt.dayCoilLine.geometry.dispose()
        if (rt.hourCoilLine) rt.hourCoilLine.geometry.dispose()
        if (rt.minCoilLine) rt.minCoilLine.geometry.dispose()
        if (rt.secCoilLine) rt.secCoilLine.geometry.dispose()
        rt.material.dispose()
        rt.lineMaterial.dispose()
        rt.renderer.dispose()
        runtimeRef.current = null
      }
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
    />
  )
}
