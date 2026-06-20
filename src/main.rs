//! 主程序：启动并行搜索，输出最终结果。

use rayon::prelude::*;
use fxhash::FxHashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

mod search;
use search::*;

mod transform;
use transform::*;

/// 初始已占位置。可根据需要修改。
pub const INIT_POSITIONS: &[Pos] = &[(0, 0), (1, 0)];

/// 并行搜索阈值（候选数超过此值启用并行）。
pub const PARALLEL_THRESHOLD: usize = 4;

/// 打印网格（彩色坐标轴）。
pub fn print_grid(grid: &FxHashMap<Pos, Num>) {
    if grid.is_empty() {
        println!("Grid is empty.");
        return;
    }

    const COLOR_AXIS: &str = "\x1b[32m";
    const COLOR_RESET: &str = "\x1b[0m";

    let mut min_x = Coord::MAX;
    let mut max_x = Coord::MIN;
    let mut min_y = Coord::MAX;
    let mut max_y = Coord::MIN;

    for &(x, y) in grid.keys() {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let max_digits = grid
        .values()
        .map(|v| v.to_string().len())
        .max()
        .unwrap_or(1);
    let cell_width = max_digits + 1;
    let y_width = (max_y as i16)
        .abs()
        .max((min_y as i16).abs())
        .to_string()
        .len()
        + 1;
    let y_width = y_width.max(3);

    // 打印 x 轴标签
    print!(r"y\x");
    let offset = y_width - 2;
    print!("{:width$}", "", width = offset);
    for x in min_x..=max_x {
        print!("{}{:>width$}{}", COLOR_AXIS, x, COLOR_RESET, width = cell_width);
    }
    println!();

    // 逐行打印
    for y in (min_y..=max_y).rev() {
        print!(
            "{}{:>y_width$} {}",
            COLOR_AXIS,
            y,
            COLOR_RESET,
            y_width = y_width
        );
        for x in min_x..=max_x {
            let s = match grid.get(&(x, y)) {
                Some(num) => num.to_string(),
                None => ".".to_string(),
            };
            print!("{:>width$}", s, width = cell_width);
        }
        println!();
    }
}

fn main() {
    let initial_states = generate_initial_states();
    let best = Arc::new(GlobalBest {
        max_number: std::sync::atomic::AtomicU16::new(0),
        grid: std::sync::Mutex::new(None),
    });

    let explored_count = Arc::new(AtomicUsize::new(0));

    // 每个初始状态拥有独立的上下文，并行搜索
    initial_states.into_par_iter().for_each(|mut ctx| {
        dfs(&mut ctx, &best, &explored_count);
    });

    // 打印最终结果
    if let Some(grid) = best.grid.lock().unwrap().as_ref() {
        println!(
            "最终最优网格 (最大数字: {})",
            best.max_number.load(std::sync::atomic::Ordering::SeqCst)
        );
        print_grid(grid);
    } else {
        println!("未找到任何解。");
    }
}