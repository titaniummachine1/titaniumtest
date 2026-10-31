//! Weight layout conversion between the hub/engine `TrainableNet` (row-major per bucket-row) and the
//! bullet graph (an affine `new_affine(id, in, out)` stores `w` of shape (out, in) COLUMN-major: element
//! (o, i) at `i * out + o`). ft and psqt are already in that order (feature-major), the dense layers are not.
use crate::train::TrainableNet;

/// `src[r * cols + c]` (row-major rows x cols) -> column-major (`dst[c * rows + r]`).
pub fn to_col_major(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(src.len(), rows * cols);
    let mut d = vec![0f32; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            d[c * rows + r] = src[r * cols + c];
        }
    }
    d
}

/// Inverse of [`to_col_major`].
pub fn from_col_major(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(src.len(), rows * cols);
    let mut d = vec![0f32; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            d[r * cols + c] = src[c * rows + r];
        }
    }
    d
}

/// The named tensors of the graph, in graph layout.
pub struct GraphTensors {
    pub ftw: Vec<f32>,
    pub ftb: Vec<f32>,
    pub psqw: Vec<f32>,
    pub l1w: Vec<f32>,
    pub l1b: Vec<f32>,
    pub l2w: Vec<f32>,
    pub l2b: Vec<f32>,
    pub outw: Vec<f32>,
    pub outb: Vec<f32>,
}

pub fn net_to_graph(n: &TrainableNet) -> GraphTensors {
    let (nb, w, l1, l2, oi) = (n.buckets, n.width, n.l1, n.l2, n.out_in());
    GraphTensors {
        ftw: n.ft.clone(),
        ftb: n.ftb.clone(),
        psqw: n.psqt.clone(),
        l1w: to_col_major(&n.l1_w, nb * l1, 2 * w),
        l1b: n.l1_b.clone(),
        l2w: to_col_major(&n.l2_w, nb * l2, l1),
        l2b: n.l2_b.clone(),
        outw: to_col_major(&n.out_w, nb, oi),
        outb: n.out_b.clone(),
    }
}

/// Write graph tensors back into a net (everything the graph trains; dis head / s57 untouched).
pub fn graph_to_net(g: &GraphTensors, n: &mut TrainableNet) {
    let (nb, w, l1, l2, oi) = (n.buckets, n.width, n.l1, n.l2, n.out_in());
    n.ft = g.ftw.clone();
    n.ftb = g.ftb.clone();
    n.psqt = g.psqw.clone();
    n.l1_w = from_col_major(&g.l1w, nb * l1, 2 * w);
    n.l1_b = g.l1b.clone();
    n.l2_w = from_col_major(&g.l2w, nb * l2, l1);
    n.l2_b = g.l2b.clone();
    n.out_w = from_col_major(&g.outw, nb, oi);
    n.out_b = g.outb.clone();
}

#[cfg(test)]
mod tests {
    use super::*;
    use titanium::nnue::BUNDLED_NNUE;
    #[test]
    fn col_major_roundtrip() {
        let v: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let c = to_col_major(&v, 3, 4);
        assert_eq!(c[0..3], [0.0, 4.0, 8.0]); // column 0 of the row-major matrix
        assert_eq!(from_col_major(&c, 3, 4), v);
    }
    #[test]
    fn net_roundtrip_is_identity() {
        let mut n = TrainableNet::from_tnn_bytes(BUNDLED_NNUE).unwrap();
        let before = n.to_tnn_bytes();
        let g = net_to_graph(&n);
        graph_to_net(&g, &mut n);
        assert!(n.to_tnn_bytes() == before);
    }
}
