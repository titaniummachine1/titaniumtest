//! D1554 (user: 'the absolute best adaptive learning-rate algorithm'): Schedule-Free AdamW on `--backend cuda`
//! (`--opt sfadamw`). Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky, 'The Road Less Scheduled' (NeurIPS 2024),
//! winner of the AlgoPerf 2024 self-tuning track: no decay schedule and no horizon; the averaged iterate x is the model
//! to evaluate at any step (LAWS D5). Per tensor (Algorithm 1, AdamW form, bias-corrected second moment):
//!   y = (1 - b1) z + b1 x          the point where the gradient is taken (= the graph's weights buffer)
//!   v = b2 v + (1 - b2) g^2,  vhat = v / (1 - b2^t)
//!   z = clip(z - lr_t (g / (sqrt(vhat) + eps) + decay y))
//!   c = lr_t^2 / sum_{i<=t} lr_i^2,  x = (1 - c) x + c z
//!   y = (1 - b1) z + b1 x
//! z and x start at the initial weights (step 1 reads them from y). Validation / export read x through `SfReg`
//! (bullet keeps the optimiser states private; states are created in the sorted tensor-name order of the graph).
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use bullet_compiler::tensor::{DType, IRTrace, TType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseBuilder,
    runtime::{Device, DeviceProps, Gpu, Stream},
};
use bullet_trainer::optimiser::{OptimiserState, OptimiserUpdateSync};

/// the averaged iterates x of every tensor, in creation order (= the graph's sorted tensor names)
pub struct SfReg<G: Gpu>(pub Arc<Mutex<Vec<Arc<Buffer<G>>>>>);

impl<G: Gpu> Clone for SfReg<G> {
    fn clone(&self) -> Self {
        SfReg(self.0.clone())
    }
}
impl<G: Gpu> Default for SfReg<G> {
    fn default() -> Self {
        SfReg(Arc::new(Mutex::new(Vec::new())))
    }
}
impl<G: Gpu> std::fmt::Debug for SfReg<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SfReg({} tensors)", self.0.lock().map(|v| v.len()).unwrap_or(0))
    }
}

pub struct SfParams<G: Gpu> {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub min_weight: f32,
    pub max_weight: f32,
    pub reg: SfReg<G>,
}

// manual impls: a derive would demand G: Clone + Debug (the GPU marker type is neither)
impl<G: Gpu> Clone for SfParams<G> {
    fn clone(&self) -> Self {
        SfParams {
            decay: self.decay,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            min_weight: self.min_weight,
            max_weight: self.max_weight,
            reg: self.reg.clone(),
        }
    }
}
impl<G: Gpu> std::fmt::Debug for SfParams<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SfParams(decay {} b1 {} b2 {} eps {} clip {}..{} {:?})", self.decay, self.beta1, self.beta2, self.eps, self.min_weight, self.max_weight, self.reg)
    }
}

impl<G: Gpu> Default for SfParams<G> {
    fn default() -> Self {
        SfParams { decay: 0.0, beta1: 0.9, beta2: 0.999, eps: 1e-8, min_weight: -1.98, max_weight: 1.98, reg: SfReg::default() }
    }
}

fn build(p: &SfParams<impl Gpu>, size: usize, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
    let (builder, p2size) = PointwiseBuilder::vectorised(size);
    let scalar = TType::new(1, DType::F32);
    let ty = TType::new(size, DType::F32);
    let adj_buf = builder.new_buffer(scalar);
    let rate_buf = builder.new_buffer(scalar);
    let c_buf = builder.new_buffer(scalar);
    let bc_buf = builder.new_buffer(scalar);
    let first_buf = builder.new_buffer(scalar);
    let grad_buf = builder.new_buffer(ty);
    let y_buf = builder.new_buffer(ty);
    let z_buf = builder.new_buffer(ty);
    let x_buf = builder.new_buffer(ty);
    let v_buf = builder.new_buffer(ty);

    let adj = adj_buf.read(0, 0).splat(p2size);
    let rate = rate_buf.read(0, 0).splat(p2size);
    let c = c_buf.read(0, 0).splat(p2size);
    let bc = bc_buf.read(0, 0).splat(p2size);
    let first = first_buf.read(0, 0).splat(p2size);
    let tid = builder.tid();
    let g = adj * grad_buf.read(tid, p2size);
    let y = y_buf.read(tid, p2size);
    // step 1: z = x = y = the initial weights
    let z = first * y + (1.0 - first) * z_buf.read(tid, p2size);
    let x = first * y + (1.0 - first) * x_buf.read(tid, p2size);
    let v = p.beta2 * v_buf.read(tid, p2size) + (1.0 - p.beta2) * g * g;
    let vhat = v * bc.recip();
    let step = g * (vhat.sqrt() + p.eps).recip() + p.decay * y;
    let z = (z - rate * step).max(p.min_weight).min(p.max_weight);
    let x = (1.0 - c) * x + c * z;
    let y = (1.0 - p.beta1) * z + p.beta1 * x;

    y_buf.write(tid, y);
    z_buf.write(tid, z);
    x_buf.write(tid, x);
    v_buf.write(tid, v);
    builder.ir().eliminate_common_subexprs()?;
    unsafe { Ok(builder.inner().lower("sfadamw".to_string(), props)?) }
}

pub struct SfAdamW<G: Gpu> {
    z: Arc<Buffer<G>>,
    x: Arc<Buffer<G>>,
    v: Arc<Buffer<G>>,
    c_dev: Arc<Buffer<G>>,
    bc_dev: Arc<Buffer<G>>,
    first_dev: Arc<Buffer<G>>,
    host: [TValue; 3],
    t: u64,
    sum_lr2: f64,
    p: SfParams<G>,
    op: CompiledKernel<G>,
}

fn f32s(v: TValue) -> Vec<f32> {
    match v {
        TValue::F32(x) => x,
        _ => panic!("sfadamw: expected an f32 buffer"),
    }
}

impl<G: Gpu> OptimiserState<G> for SfAdamW<G> {
    type Params = SfParams<G>;

    fn new(device: &Arc<Device<G>>, size: usize, p: Self::Params) -> Result<Self, G::Error> {
        let op = build(&p, size, device.props()).unwrap().compile(device.clone())?;
        let zeros = TValue::zeros(DType::F32, size);
        let x = Buffer::from_host(device, &zeros)?;
        p.reg.0.lock().unwrap().push(x.clone());
        let one = TValue::F32(vec![1.0]);
        Ok(SfAdamW {
            z: Buffer::from_host(device, &zeros)?,
            x,
            v: Buffer::from_host(device, &zeros)?,
            c_dev: Buffer::from_host(device, &one)?,
            bc_dev: Buffer::from_host(device, &one)?,
            first_dev: Buffer::from_host(device, &one)?,
            host: [one.clone(), one.clone(), one],
            t: 0,
            sum_lr2: 0.0,
            p,
            op,
        })
    }

    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> Result<OptimiserUpdateSync<'a, G>, G::Error> {
        let lr = f32s(learning_rate.to_host_async(stream)?.value()?)[0] as f64;
        self.t += 1;
        self.sum_lr2 += lr * lr;
        let c = if self.sum_lr2 > 0.0 { lr * lr / self.sum_lr2 } else { 1.0 };
        let bc = 1.0 - (self.p.beta2 as f64).powi(self.t.min(i32::MAX as u64) as i32);
        let first = if self.t == 1 { 1.0 } else { 0.0 };
        self.host = [TValue::F32(vec![c as f32]), TValue::F32(vec![bc.max(1e-12) as f32]), TValue::F32(vec![first])];
        let me: &'a Self = self;
        let mut sync = OptimiserUpdateSync::default();
        sync.push_copy(me.c_dev.copy_from_host_async(stream, &me.host[0])?);
        sync.push_copy(me.bc_dev.copy_from_host_async(stream, &me.host[1])?);
        sync.push_copy(me.first_dev.copy_from_host_async(stream, &me.host[2])?);
        sync.push_kernel(me.op.execute(
            stream.clone(),
            vec![gradient_factor, learning_rate, me.c_dev.clone(), me.bc_dev.clone(), me.first_dev.clone(), grads],
            vec![weights, me.z.clone(), me.x.clone(), me.v.clone()],
        )?);
        Ok(sync)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        let size = self.v.size();
        self.v.copy_from_host(&TValue::zeros(DType::F32, size))?;
        self.t = 0;
        self.sum_lr2 = 0.0;
        Ok(())
    }

    fn set_params(&mut self, p: Self::Params) -> Result<(), G::Error> {
        let size = self.v.size();
        let device = self.v.device();
        self.op = build(&p, size, device.props()).unwrap().compile(device)?;
        self.p = p;
        Ok(())
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        for (id, s) in map {
            for (tag, b) in [("z", &s.z), ("x", &s.x), ("v", &s.v)] {
                let bytes: Vec<u8> = f32s(b.to_host()?).iter().flat_map(|f| f.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{path}/sf_{tag}_{id}.bin"), bytes);
            }
            let _ = std::fs::write(format!("{path}/sf_t_{id}.txt"), format!("{} {}", s.t, s.sum_lr2));
        }
        Ok(())
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        for (id, s) in map.iter_mut() {
            for (tag, b) in [("z", &s.z), ("x", &s.x), ("v", &s.v)] {
                if let Ok(bytes) = std::fs::read(format!("{path}/sf_{tag}_{id}.bin")) {
                    if bytes.len() == b.size() * 4 {
                        let v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        b.copy_from_host(&TValue::F32(v))?;
                    }
                }
            }
            if let Ok(t) = std::fs::read_to_string(format!("{path}/sf_t_{id}.txt")) {
                let mut it = t.split_whitespace();
                s.t = it.next().and_then(|a| a.parse().ok()).unwrap_or(0);
                s.sum_lr2 = it.next().and_then(|a| a.parse().ok()).unwrap_or(0.0);
            }
        }
        Ok(())
    }
}

/// the weights to validate / export under Schedule-Free: the averaged iterates x, one per tensor in sorted-name order
pub fn averaged_weights<G: Gpu>(reg: &SfReg<G>, live: &bullet_trainer::model::ModelWeights) -> Result<bullet_trainer::model::ModelWeights, String> {
    let xs = reg.0.lock().map_err(|_| "sfadamw: registry poisoned".to_string())?;
    let names: Vec<String> = live.iter().map(|(id, _)| id.clone()).collect();
    if names.len() != xs.len() {
        return Err(format!("sfadamw: {} tensors but {} averaged iterates", names.len(), xs.len()));
    }
    let mut w = live.clone();
    for (id, x) in names.iter().zip(xs.iter()) {
        let v = x.to_host().map_err(|e| format!("sfadamw: read x of {id}: {e:?}"))?;
        if v.size() != live.get(id).values.size() {
            return Err(format!("sfadamw: size mismatch for {id}"));
        }
        w.set(id, v);
    }
    Ok(w)
}
