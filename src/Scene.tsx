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

// ── CPU evalCoil (kept for camera) ───────────────────────────────────────────
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

// ── GPU compute: build evalCoil as TSL for a given max level ─────────────────
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

    // Century helix: use baseFrac[0] for precise theta
    // turnsPerVertex_0 = tStep * totalTurns[0] (small, f32-safe)
    const helixFrac = baseFracUniforms[0].add(tStepU.mul(float(totalTurns[0])).mul(idx.toFloat()))
    const theta = fract(helixFrac).mul(float(TAU))
    const cT = cos(theta), sT = sin(theta)

    const px = radiusU.mul(cT).toVar('px')
    const py = t.sub(0.5).mul(lengthU).toVar('py')
    const pz = radiusU.mul(sT).toVar('pz')

    const nx = cT.negate().toVar('nx')
    const ny = float(0).toVar('ny')
    const nz = sT.negate().toVar('nz')

    const bx = lengthU.negate().mul(sT).div(tmagU).toVar('bx')
    const by = radiusU.negate().mul(omegaU).div(tmagU).toVar('by')
    const bz = lengthU.mul(cT).div(tmagU).toVar('bz')

    // Un-normalized tangent (accumulated through levels for Gram-Schmidt)
    const dtx = radiusU.negate().mul(omegaU).mul(sT).toVar('dtx')
    const dty = lengthU.toVar('dty')
    const dtz = radiusU.mul(omegaU).mul(cT).toVar('dtz')

    for (let lvl = 1; lvl <= maxLevel; lvl++) {
      const lvlFrac = baseFracUniforms[lvl].add(tStepU.mul(float(totalTurns[lvl])).mul(idx.toFloat()))
      const alpha = fract(lvlFrac).mul(float(TAU))
      const cA = cos(alpha), sA = sin(alpha)
      const off = offUniforms[lvl]

      // D = offset direction (rotated N), E = perpendicular (rotated B)
      const dx = cA.mul(nx).add(sA.mul(bx)).toVar(`dx${lvl}`)
      const dy = cA.mul(ny).add(sA.mul(by)).toVar(`dy${lvl}`)
      const dz = cA.mul(nz).add(sA.mul(bz)).toVar(`dz${lvl}`)
      const ex = sA.negate().mul(nx).add(cA.mul(bx)).toVar(`ex${lvl}`)
      const ey = sA.negate().mul(ny).add(cA.mul(by)).toVar(`ey${lvl}`)
      const ez = sA.negate().mul(nz).add(cA.mul(bz)).toVar(`ez${lvl}`)

      // Offset position
      px.addAssign(off.mul(dx))
      py.addAssign(off.mul(dy))
      pz.addAssign(off.mul(dz))

      // Accumulate tangent: T += off * angularRate * E
      // angularRate = totals[k] * TAU, precomputed as f64 then baked to f32
      const winding = off.mul(float(totalTurns[lvl] * TAU))
      dtx.addAssign(winding.mul(ex))
      dty.addAssign(winding.mul(ey))
      dtz.addAssign(winding.mul(ez))

      // Gram-Schmidt: project D perpendicular to tangent → N
      const tLen = sqrt(dtx.mul(dtx).add(dty.mul(dty)).add(dtz.mul(dtz)))
      const txn = dtx.div(tLen), tyn = dty.div(tLen), tzn = dtz.div(tLen)
      const dot = dx.mul(txn).add(dy.mul(tyn)).add(dz.mul(tzn))
      const nnx = dx.sub(dot.mul(txn)).toVar(`nnx${lvl}`)
      const nny = dy.sub(dot.mul(tyn)).toVar(`nny${lvl}`)
      const nnz = dz.sub(dot.mul(tzn)).toVar(`nnz${lvl}`)
      const nLen = sqrt(nnx.mul(nnx).add(nny.mul(nny)).add(nnz.mul(nnz)))
      nx.assign(nnx.div(nLen)); ny.assign(nny.div(nLen)); nz.assign(nnz.div(nLen))

      // B = T × N
      bx.assign(tyn.mul(nz).sub(tzn.mul(ny)))
      by.assign(tzn.mul(nx).sub(txn.mul(nz)))
      bz.assign(txn.mul(ny).sub(tyn.mul(nx)))
    }

    posBuf.element(idx).assign(vec3(px, py, pz))
  })().compute(maxPts)
}

// ── Types ────────────────────────────────────────────────────────────────────
interface LevelGPU {
  posBuf: ReturnType<typeof attributeArray>
  compute: ReturnType<typeof createCoilCompute>
  line: THREE.Line
  maxPts: number
}

interface Runtime {
  renderer: WebGPURenderer; scene: THREE.Scene
  camera: THREE.PerspectiveCamera; controls: OrbitControls
  levels: LevelGPU[]
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

    const START_YEAR = -10000 // 10,000 BCE
    const END_YEAR = 2100    // 2100 CE
    const TOTAL_YEARS = END_YEAR - START_YEAR // 12,100
    const CENTURY_TURNS = TOTAL_YEARS / 100   // 121

    const params = {
      coilRadius: 1,
      turnSpacing: 10,
      offsets: [0, 0.1740, 0.0871, 0.0300, 0.0007, 0.0005, 0.0001],
      panSpeed: 10,
      focusT: 0.5,
    }

    // Shared uniforms
    // For precision: tBase is the t value at idx=0, computed in f64 on CPU.
    // tStep is the t increment per vertex (tRange / (maxPts-1)).
    // The shader computes t = tBase + tStep * idx, but we also pass
    // the century-turn fractional part at tBase (f64-precise) to seed the cascade.
    const tBaseU = uniform(0)
    const tStepU = uniform(0)
    // Per-level f64-precise fractional turns, computed on CPU
    const baseFracUniforms = HIERARCHY.map(() => uniform(0))
    const radiusU = uniform(params.coilRadius)
    const lengthU = uniform(CENTURY_TURNS * params.turnSpacing)
    const omegaU = uniform(CENTURY_TURNS * TAU)
    const tmagU = uniform(1)
    const offUniforms = params.offsets.map(v => uniform(v))

    function syncHelixUniforms() {
      const R = params.coilRadius, L = CENTURY_TURNS * params.turnSpacing
      const omega = CENTURY_TURNS * TAU
      radiusU.value = R; lengthU.value = L; omegaU.value = omega
      tmagU.value = Math.sqrt(R * R * omega * omega + L * L)
    }

    function getTotalTurns(): number[] {
      const t = [CENTURY_TURNS]
      for (let i = 1; i < HIERARCHY.length; i++) t.push(t[i - 1] * HIERARCHY[i].turnsPerParent)
      return t
    }

    function getHelixConsts() {
      const R = params.coilRadius, L = CENTURY_TURNS * params.turnSpacing
      const omega = CENTURY_TURNS * TAU
      return { R, L, omega, tMag: Math.sqrt(R * R * omega * omega + L * L) }
    }

    function winBounds(totalTurns: number, spt: number) {
      const halfT = (WINDOW / 2) / totalTurns
      const tS = Math.max(0, params.focusT - halfT)
      const tE = Math.min(1, params.focusT + halfT)
      return { tS, tR: tE - tS }
    }

    // ── Camera (CPU, year level) ──────────────────────────────────────────
    const _pos = { x: 0, y: 0, z: 0 }, _nrm = { x: 0, y: 0, z: 0 }, _bin = { x: 0, y: 0, z: 0 }
    const _rN = new THREE.Vector3(), _rB = new THREE.Vector3(), _tan = new THREE.Vector3(), _off = new THREE.Vector3()
    let prevFocusT = params.focusT

    function cameraFrame(t: number) {
      const totals = getTotalTurns(), hc = getHelixConsts()
      evalCoil(t, 2, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _pos, _nrm, _bin)
      const eps = 1e-7, p0 = { x: 0, y: 0, z: 0 }, p1 = { x: 0, y: 0, z: 0 }
      evalCoil(Math.max(0, t - eps), 2, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, p0)
      evalCoil(Math.min(1, t + eps), 2, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, p1)
      _tan.set(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z).normalize()
      const nv = new THREE.Vector3(_nrm.x, _nrm.y, _nrm.z).addScaledVector(_tan, -new THREE.Vector3(_nrm.x, _nrm.y, _nrm.z).dot(_tan)).normalize()
      const bv = new THREE.Vector3().crossVectors(_tan, nv)
      const da = t * getTotalTurns()[3] * TAU, c = Math.cos(da), s = Math.sin(da)
      _rN.copy(nv).multiplyScalar(c).addScaledVector(bv, s)
      _rB.copy(nv).multiplyScalar(-s).addScaledVector(bv, c)
    }

    // Level hues: evenly spaced around the color wheel
    const LEVEL_HUES = [0, 0.08, 0.16, 0.33, 0.55, 0.72, 0.88]

    function createLevels(scene: THREE.Scene): LevelGPU[] {
      syncHelixUniforms()
      const totals = getTotalTurns()
      return HIERARCHY.map((h, lvl) => {
        const maxPts = WINDOW * h.spt + 1
        const posBuf = attributeArray(new Float32Array(maxPts * 3), 'vec3')
        const compute = createCoilCompute(
          lvl, totals, maxPts, offUniforms,
          tBaseU, tStepU, baseFracUniforms, radiusU, lengthU, omegaU, tmagU,
          posBuf,
        )
        const col = new THREE.Color().setHSL(LEVEL_HUES[lvl], 1, 0.5)
        const mat = new THREE.LineBasicMaterial({ color: col })
        const geom = new THREE.BufferGeometry()
        geom.setAttribute('position', posBuf.value)
        const line = new THREE.Line(geom, mat)
        line.frustumCulled = false
        scene.add(line)
        return { posBuf, compute, line, maxPts }
      })
    }

    function disposeLevels(rt: Runtime) {
      for (const lv of rt.levels) { rt.scene.remove(lv.line); lv.line.geometry.dispose() }
      rt.levels = []
    }

    // ── Dispatch all compute shaders ──────────────────────────────────────
    function dispatchCompute(rt: Runtime) {
      const totals = getTotalTurns()
      for (let i = 0; i < offUniforms.length; i++) offUniforms[i].value = params.offsets[i]
      for (let i = 0; i < rt.levels.length; i++) {
        const w = winBounds(totals[i], HIERARCHY[i].spt)
        const maxPts = rt.levels[i].maxPts
        const tBase = w.tS                          // f64
        const tStep = w.tR / (maxPts - 1)           // f64
        tBaseU.value = tBase
        tStepU.value = tStep
        // Cascade fractional turns in f64 — keeps intermediate values small
        // so we don't lose precision multiplying tBase by huge totals
        let f = tBase * CENTURY_TURNS
        baseFracUniforms[0].value = f % 1
        for (let k = 1; k < totals.length; k++) {
          f = (f % 1) * HIERARCHY[k].turnsPerParent
          baseFracUniforms[k].value = f % 1
        }
        rt.renderer.compute(rt.levels[i].compute)
      }
    }

    function updateFocus(rt: Runtime) {
      const oldT = prevFocusT, newT = params.focusT
      prevFocusT = newT
      cameraFrame(oldT)
      const p = new THREE.Vector3(_pos.x, _pos.y, _pos.z)
      _off.copy(rt.camera.position).sub(p)
      const lx = _off.dot(_rN), ly = _off.dot(_rB), lz = _off.dot(_tan)

      dispatchCompute(rt)

      cameraFrame(newT)
      const p2 = new THREE.Vector3(_pos.x, _pos.y, _pos.z)
      rt.camera.position.copy(p2).addScaledVector(_rN, lx).addScaledVector(_rB, ly).addScaledVector(_tan, lz)
      rt.controls.target.copy(p2)
    }

    function fullRebuild(rt: Runtime) {
      disposeLevels(rt)
      rt.levels = createLevels(rt.scene)
      dispatchCompute(rt)
    }

    // ── Init ───────────────────────────────────────────────────────────────
    const init = async () => {
      if (!navigator.gpu) {
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU is not supported in this browser.</div>'
        return
      }
      const w = window.innerWidth, h = window.innerHeight
      const renderer = new WebGPURenderer({ canvas: canvas!, antialias: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      renderer.setSize(w, h, false)
      await renderer.init()
      if (!(renderer as any).backend?.isWebGPUBackend) {
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU backend failed to initialize.</div>'
        renderer.dispose(); return
      }
      if (disposed) { renderer.dispose(); return }

      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x111111)
      const camera = new THREE.PerspectiveCamera(60, w / h, 0.0001, 500)
      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true; controls.dampingFactor = 0.08; controls.enablePan = false

      scene.add(new THREE.DirectionalLight(0xffffff, 2).translateX(3).translateY(5).translateZ(4))
      scene.add(new THREE.AmbientLight(0xffffff, 0.3))


      const pane = new Pane({ title: 'Coilendar' })

      const rt: Runtime = { renderer, scene, camera, controls, levels: [], pane }
      runtimeRef.current = rt

      rt.levels = createLevels(scene)
      dispatchCompute(rt)

      cameraFrame(params.focusT)
      const p = new THREE.Vector3(_pos.x, _pos.y, _pos.z)
      controls.target.copy(p)
      camera.position.copy(p).addScaledVector(_rN, -2).addScaledVector(_rB, 0.5)

      // Right-click drag
      let dragging = false, dragStartX = 0, dragStartT = 0, focusDragging = false
      const focusBinding = pane.addBinding(params, 'focusT', { min: 0, max: 1, step: 0.0001, label: 'focus' })
      focusBinding.on('change', () => { if (!focusDragging) updateFocus(rt) })
      canvas.addEventListener('pointerdown', (e) => { if (e.button === 2) { dragging = true; focusDragging = true; dragStartX = e.clientX; dragStartT = params.focusT; e.preventDefault() } })
      canvas.addEventListener('pointermove', (e) => {
        if (!dragging) return
        const totals = getTotalTurns()
        params.focusT = THREE.MathUtils.clamp(dragStartT - (e.clientX - dragStartX) * params.panSpeed / totals[totals.length - 1], 0, 1)
        updateFocus(rt)
      })
      const stopDrag = () => { dragging = false; focusDragging = false }
      canvas.addEventListener('pointerup', stopDrag)
      canvas.addEventListener('pointercancel', stopDrag)
      canvas.addEventListener('contextmenu', (e) => e.preventDefault())

      // Structural params → full rebuild (recompiles shaders with new totalTurns)
      // Century turns fixed by START_YEAR/END_YEAR (121 turns for 10000 BCE–2100 CE)
      pane.addBinding(params, 'coilRadius', { min: 0.1, max: 5, step: 0.1, label: 'coil radius' })
        .on('change', () => { syncHelixUniforms(); dispatchCompute(rt) })
      pane.addBinding(params, 'turnSpacing', { min: 0.1, max: 20, step: 0.1, label: 'turn spacing' })
        .on('change', () => { syncHelixUniforms(); dispatchCompute(rt) })

      const names = ['century', 'decade', 'year', 'day', 'hour', 'min', 'sec']
      for (let i = 1; i < params.offsets.length; i++) {
        const idx = i
        const isLast = i === 6
        pane.addBinding(params.offsets, idx as unknown as keyof typeof params.offsets, {
          min: isLast ? 0.000001 : 0.0001,
          max: i < 3 ? 2 : 0.5,
          step: isLast ? 0.000001 : 0.0001,
          label: names[idx] + ' off',
        })
          .on('change', () => { offUniforms[idx].value = params.offsets[idx]; dispatchCompute(rt) })
      }
      pane.addBinding(params, 'panSpeed', { min: 1, max: 3600, step: 1, label: 'sec/pixel' })

      const animate = () => {
        if (disposed) return
        rafRef.current = requestAnimationFrame(animate)
        rt.controls.update()
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
      disposed = true; window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(rafRef.current)
      const rt = runtimeRef.current
      if (rt) { rt.pane.dispose(); rt.controls.dispose(); disposeLevels(rt); rt.renderer.dispose(); runtimeRef.current = null }
    }
  }, [])

  return <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
}
