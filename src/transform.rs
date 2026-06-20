//! 坐标变换、对称操作及自同构缓存，用于剪枝对称等价的状态。

use fxhash::FxHashSet;
use rayon::prelude::*;

/// 坐标类型（有符号 8 位整数，适用于小规模网格）。
pub type Coord = i8;

/// 坐标对 `(x, y)`。
pub type Pos = (Coord, Coord);

/// 8 种正交变换矩阵（旋转 90° 倍数与反射），每个矩阵为 2×2。
/// 用于生成点集的所有对称等价变换。
const TRANSFORMS: [[[Coord; 2]; 2]; 8] = [
    [[1, 0], [0, 1]],   // 恒等
    [[0, -1], [1, 0]],  // 旋转 90°
    [[-1, 0], [0, -1]], // 旋转 180°
    [[0, 1], [-1, 0]],  // 旋转 270°
    [[1, 0], [0, -1]],  // 关于 x 轴反射
    [[-1, 0], [0, 1]],  // 关于 y 轴反射
    [[0, 1], [1, 0]],   // 关于直线 y = x 反射
    [[0, -1], [-1, 0]], // 关于直线 y = -x 反射
];

/// 将正交变换 `transform` 应用于点 `p`。
#[inline]
fn apply_transform(transform: &[[Coord; 2]; 2], p: Pos) -> Pos {
    let (x, y) = p;
    let a = transform[0][0];
    let b = transform[0][1];
    let c = transform[1][0];
    let d = transform[1][1];
    (a * x + b * y, c * x + d * y)
}

/// 预计算点集 `s` 的所有自同构（即保持点集不变的变换 + 平移组合）。
///
/// 返回一个向量，每个元素为一个正交变换和对应的平移量。
fn precompute_automorphisms(s: &[Pos]) -> Vec<([[Coord; 2]; 2], Pos)> {
    if s.is_empty() {
        return Vec::new();
    }
    let set: FxHashSet<Pos> = s.iter().cloned().collect();

    TRANSFORMS
        .par_iter()
        .flat_map(|&transform| {
            let transformed_first = apply_transform(&transform, s[0]);
            let mut local_results = Vec::new();

            for &target in s {
                let translation = (
                    target.0 - transformed_first.0,
                    target.1 - transformed_first.1,
                );
                // 检查该变换和平移是否将整个点集映射到自身
                let valid = s.iter().all(|&pt| {
                    let transformed = apply_transform(&transform, pt);
                    let moved = (transformed.0 + translation.0, transformed.1 + translation.1);
                    set.contains(&moved)
                });
                if valid {
                    local_results.push((transform, translation));
                }
            }
            local_results
        })
        .collect()
}

/// 对称性缓存，用于快速查找与给定位置对称等价的所有其他位置。
///
/// 通过预计算初始点集的自同构，避免了重复的对称性判断。
pub struct SymmetryCache {
    /// 所有自同构，每个自同构由一个正交变换和后续平移组成。
    automorphisms: Vec<([[Coord; 2]; 2], Pos)>,
}

impl SymmetryCache {
    /// 从点集 `s` 构建对称性缓存。
    pub fn new(s: &[Pos]) -> Self {
        let automorphisms = precompute_automorphisms(s);
        SymmetryCache { automorphisms }
    }

    /// 返回与给定位置 `p` 对称等价的所有其他位置（不包括 `p` 自身）。
    ///
    /// 基于缓存的自同构，计算每个自同构下 `p` 的像，并收集其中与 `p` 不同的点。
    pub fn find_symmetric_positions(&self, p: Pos) -> FxHashSet<Pos> {
        let mut results = FxHashSet::default();
        for &(transform, translation) in &self.automorphisms {
            let transformed_p = apply_transform(&transform, p);
            let q = (
                transformed_p.0 + translation.0,
                transformed_p.1 + translation.1,
            );
            if q != p {
                results.insert(q);
            }
        }
        results
    }
}