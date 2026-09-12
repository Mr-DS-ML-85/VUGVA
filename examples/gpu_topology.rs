//! GPU topology and NUMA awareness example.
//!
//! Run:  cargo run --example gpu_topology

use vugva::gpu::GpuCluster;

fn main() -> vugva::Result<()> {
    println!("=== VUGVA GPU Topology ===\n");

    let gpus: Vec<i32> = (0..8).collect();
    let cluster = GpuCluster::discover(&gpus)?;

    println!("Found {} GPU(s):", cluster.infos.len());
    for (i, info) in cluster.infos.iter().enumerate() {
        println!("\n  [{i}] {}", info.name);
        println!("      Device ID: {}", info.device_id);
        println!(
            "      sm_{}{}",
            info.compute_capability.0, info.compute_capability.1
        );
        println!("      VRAM:      {} MB", info.total_vram / 1024 / 1024);
        println!("      Free:      {} MB", info.free_vram / 1024 / 1024);
        println!("      SMs:       {}", info.sm_count);
        println!("      NUMA node: {}", info.numa_node);
    }

    // P2P access
    println!("\nPeer Access:");
    for (i, &ord) in cluster.ordinals.iter().enumerate() {
        for (j, &ord2) in cluster.ordinals.iter().enumerate() {
            if i != j {
                let can = cluster.peer_matrix.can_access(i, j);
                println!("  GPU {ord} → GPU {ord2}: {can}");
            }
        }
    }

    // NUMA bandwidth factors
    println!("\nNUMA Bandwidth Factors:");
    for &ord in &cluster.ordinals {
        let node = info_numa_node(&cluster, ord);
        let factor = cluster.numa.dma_bandwidth_factor(node, node);
        println!("  GPU {ord} (node {node}): {factor}");
    }

    // Optimal DRAM node per GPU
    println!("\nOptimal DRAM Nodes:");
    for &ord in &cluster.ordinals {
        let node = cluster.optimal_dram_node(ord);
        println!("  GPU {ord} → NUMA node {node}");
    }

    println!("\nTopology discovery complete!");
    Ok(())
}

fn info_numa_node(cluster: &GpuCluster, ordinal: i32) -> usize {
    cluster
        .infos
        .iter()
        .find(|i| i.device_id == ordinal)
        .map(|i| i.numa_node)
        .unwrap_or(0)
}
