//! 搜索算法：深度优先搜索 + 自适应并行，利用对称性剪枝和边界可达和。

use fxhash::{FxHashMap, FxHashSet};
use arrayvec::ArrayVec;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{INIT_POSITIONS, PARALLEL_THRESHOLD};
use crate::transform::*;

pub type Num = u16;

/// 八邻域偏移量。
const NEIGHBORS: [(Coord, Coord); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
];

/// 边界空格的“可达和”集合，使用栈上数组存储（最多 255 种）。
#[derive(Clone)]
pub struct BoundaryInfo {
    pub sums: ArrayVec<Num, 256>,
    pub neighbor_count: u8,
}

impl BoundaryInfo {
    pub fn new() -> Self {
        BoundaryInfo {
            sums: ArrayVec::new(),
            neighbor_count: 0,
        }
    }

    /// 向该边界空格添加一个相邻数字 `n`，更新所有可能的子集和。
    pub fn add_number(&mut self, n: Num) {
        let old = self.sums.clone();
        self.sums.clear();
        self.sums.push(n);
        for &v in &old {
            if !self.sums.contains(&v) {
                self.sums.push(v);
            }
            let new_val = v + n;
            if !self.sums.contains(&new_val) {
                self.sums.push(new_val);
            }
        }
        self.sums.sort_unstable();
        let mut deduped = ArrayVec::new();
        let mut last = None;
        for &val in &self.sums {
            if Some(val) != last {
                deduped.push(val);
                last = Some(val);
            }
        }
        self.sums = deduped;
        self.neighbor_count += 1;
    }

    pub fn contains(&self, n: Num) -> bool {
        self.sums.contains(&n)
    }
}

/// 搜索上下文（可变状态：已填数字网格 + 边界信息）。
pub struct Context {
    pub grid: FxHashMap<Pos, Num>,
    pub boundary: FxHashMap<Pos, BoundaryInfo>,
}

impl Context {
    /// 从初始已占位置集合创建上下文（所有位置初始数字为 1）。
    pub fn new_from_positions(positions: &[Pos]) -> Self {
        let mut grid = FxHashMap::default();
        for &p in positions {
            grid.insert(p, 1);
        }
        let boundary = Self::compute_boundary_sums(&grid);
        Context { grid, boundary }
    }

    /// 计算当前网格所有边界空格的可达和。
    fn compute_boundary_sums(grid: &FxHashMap<Pos, Num>) -> FxHashMap<Pos, BoundaryInfo> {
        let mut result = FxHashMap::default();
        let mut candidates = FxHashSet::default();
        for &pos in grid.keys() {
            for &(dx, dy) in &NEIGHBORS {
                let nb = (pos.0 + dx, pos.1 + dy);
                if !grid.contains_key(&nb) {
                    candidates.insert(nb);
                }
            }
        }
        for &pos in &candidates {
            let mut nums = Vec::new();
            for &(dx, dy) in &NEIGHBORS {
                let nb = (pos.0 + dx, pos.1 + dy);
                if let Some(&v) = grid.get(&nb) {
                    nums.push(v);
                }
            }
            if nums.is_empty() {
                continue;
            }
            let k = nums.len();
            let mut info = BoundaryInfo::new();
            for mask in 1..(1 << k) {
                let mut sum = 0;
                for i in 0..k {
                    if mask & (1 << i) != 0 {
                        sum += nums[i];
                    }
                }
                info.sums.push(sum);
            }
            info.sums.sort_unstable();
            let mut deduped = ArrayVec::new();
            let mut last = None;
            for &val in &info.sums {
                if Some(val) != last {
                    deduped.push(val);
                    last = Some(val);
                }
            }
            info.sums = deduped;
            info.neighbor_count = nums.len() as u8;
            result.insert(pos, info);
        }
        result
    }

    /// 下一个要填入的数字（从 2 开始递增）。
    pub fn next_number(&self) -> Num {
        (self.grid.len() + 2 - INIT_POSITIONS.len()) as Num
    }

    /// 返回所有当前可填入的候选位置（边界空格中可达和包含 `next_number`）。
    pub fn candidates(&self) -> Vec<Pos> {
        let next = self.next_number();
        let mut vec = Vec::new();
        for (&pos, info) in &self.boundary {
            if info.contains(next) {
                vec.push(pos);
            }
        }
        vec
    }

    /// 在位置 `pos` 填入数字 `n`，并更新边界信息（原地修改）。
    pub fn place_number_mut(&mut self, pos: Pos, n: Num) {
        self.boundary.remove(&pos);
        for &(dx, dy) in &NEIGHBORS {
            let nb = (pos.0 + dx, pos.1 + dy);
            if self.grid.contains_key(&nb) {
                continue;
            }
            let info = self.boundary.entry(nb).or_insert_with(BoundaryInfo::new);
            info.add_number(n);
        }
        self.grid.insert(pos, n);
    }
}

impl Clone for Context {
    fn clone(&self) -> Self {
        Context {
            grid: self.grid.clone(),
            boundary: self.boundary.clone(),
        }
    }
}

/// 全局最优解记录。
pub struct GlobalBest {
    pub max_number: std::sync::atomic::AtomicU16,
    pub grid: std::sync::Mutex<Option<FxHashMap<Pos, Num>>>,
}

/// ---------- 辅助函数 ----------

/// 处理叶子节点（无候选位置）：更新全局最优解。
#[inline]
fn handle_leaf(ctx: &Context, best: &Arc<GlobalBest>) {
    let current_max = ctx.next_number() - 1;
    let prev = best
        .max_number
        .fetch_max(current_max, std::sync::atomic::Ordering::SeqCst);
    if current_max > prev {
        let mut best_grid = best.grid.lock().unwrap();
        if current_max == best.max_number.load(std::sync::atomic::Ordering::SeqCst) {
            *best_grid = Some(ctx.grid.clone());
            println!("发现新的最优状态: {}", current_max);
            crate::print_grid(&ctx.grid);
        }
    }
}

/// 打印搜索进度（每 50,000 个状态打印一次）。
#[inline]
fn log_progress(explored_count: &Arc<AtomicUsize>) {
    let count = explored_count.fetch_add(1, Ordering::Relaxed);
    if count % 50000 == 0 {
        print!("\x1b[2K\x1b[1G已搜索 {} 个状态", count);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
}

/// ---------- 深度优先搜索入口 ----------

/// 深度优先搜索，自适应并行（候选数 > `PARALLEL_THRESHOLD` 时启用并行）。
pub fn dfs(ctx: &mut Context, best: &Arc<GlobalBest>, explored_count: &Arc<AtomicUsize>) {
    log_progress(explored_count);

    let candidates = ctx.candidates();

    // 叶子节点：无可填候选
    if candidates.is_empty() {
        handle_leaf(ctx, best);
        return;
    }

    // 若候选数较多，采用并行分支（每个候选克隆独立上下文）
    if candidates.len() > PARALLEL_THRESHOLD {
        rayon::scope(|s| {
            for pos in candidates {
                let mut cloned_ctx = ctx.clone();
                s.spawn(move |_| {
                    let next = cloned_ctx.next_number();
                    cloned_ctx.place_number_mut(pos, next);
                    dfs(&mut cloned_ctx, best, explored_count);
                });
            }
        });
        return;
    }

    // 候选数少，串行执行（带撤销，节省内存）
    let next = ctx.next_number();

    for pos in candidates {
        // --- 保存撤销信息 ---
        let old_pos_info = ctx.boundary.remove(&pos);
        let mut neighbor_restores = Vec::new();

        for &(dx, dy) in &NEIGHBORS {
            let nb = (pos.0 + dx, pos.1 + dy);
            if ctx.grid.contains_key(&nb) {
                continue;
            }
            let old_info = ctx.boundary.get(&nb).cloned().unwrap_or_else(BoundaryInfo::new);
            neighbor_restores.push((nb, old_info));
            let info = ctx.boundary.entry(nb).or_insert_with(BoundaryInfo::new);
            info.add_number(next);
        }

        ctx.grid.insert(pos, next);

        // 递归（可能再次进入并行分支）
        dfs(ctx, best, explored_count);

        // --- 撤销修改 ---
        ctx.grid.remove(&pos);

        for (nb, old) in neighbor_restores {
            if old.sums.is_empty() && old.neighbor_count == 0 {
                ctx.boundary.remove(&nb);
            } else {
                ctx.boundary.insert(nb, old);
            }
        }

        if let Some(info) = old_pos_info {
            ctx.boundary.insert(pos, info);
        }
    }
}

/// 生成所有对称不等价的初始状态（已填入 2）。
pub fn generate_initial_states() -> Vec<Context> {
    let initial_ctx = Context::new_from_positions(INIT_POSITIONS);
    let symmetry_cache = SymmetryCache::new(INIT_POSITIONS);
    let next_number = initial_ctx.next_number();
    debug_assert_eq!(next_number, 2);

    let mut seen = FxHashSet::default();
    let mut result = Vec::new();

    for (&pos, info) in &initial_ctx.boundary {
        if seen.contains(&pos) {
            continue;
        }
        if info.contains(next_number) {
            let mut ctx = initial_ctx.clone();
            ctx.place_number_mut(pos, next_number);
            result.push(ctx);
            seen.extend(symmetry_cache.find_symmetric_positions(pos));
        }
    }
    result
}