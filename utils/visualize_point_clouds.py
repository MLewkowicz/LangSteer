"""
Point cloud visualization utilities for CALVIN dataset processing.

This module provides functions to visualize point clouds at various stages
of processing (before/after cropping, before/after downsampling).
"""

import os
import numpy as np
from typing import Optional
from termcolor import cprint

# Check for Open3D (optional, for visualization)
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None


def visualize_point_clouds(
    static_pcd: np.ndarray,
    gripper_pcd: np.ndarray,
    static_rgb: Optional[np.ndarray] = None,
    gripper_rgb: Optional[np.ndarray] = None,
    static_depth: Optional[np.ndarray] = None,
    gripper_depth: Optional[np.ndarray] = None,
    frame_idx: int = 0,
    title: str = "Point Clouds",
    save_path: Optional[str] = None,
    stage: str = ""
):
    """
    Visualize point clouds (can be used at various processing stages).
    
    Args:
        static_pcd: Static camera point cloud (N, 3) or (H, W, 3) in world coordinates
        gripper_pcd: Gripper camera point cloud (M, 3) or (H, W, 3) in world coordinates
        static_rgb: Optional static RGB image (H, W, 3) or colors (N, 3)
        gripper_rgb: Optional gripper RGB image (H, W, 3) or colors (M, 3)
        static_depth: Optional static depth image (H, W) - only used for filtering if pcd is (H, W, 3)
        gripper_depth: Optional gripper depth image (H, W) - only used for filtering if pcd is (H, W, 3)
        frame_idx: Frame index for title
        title: Window title
        save_path: Optional directory to save point cloud files and plots
        stage: Processing stage label (e.g., "after_cropping", "after_downsampling")
    """
    # Reshape if needed (handle both (H, W, 3) and (N, 3) formats)
    if len(static_pcd.shape) == 3:
        static_pcd_flat = static_pcd.reshape(-1, 3)
        if static_rgb is not None and len(static_rgb.shape) == 3:
            static_rgb_flat = static_rgb.reshape(-1, 3)
        else:
            static_rgb_flat = static_rgb
    else:
        static_pcd_flat = static_pcd
        static_rgb_flat = static_rgb
    
    if len(gripper_pcd.shape) == 3:
        gripper_pcd_flat = gripper_pcd.reshape(-1, 3)
        if gripper_rgb is not None and len(gripper_rgb.shape) == 3:
            gripper_rgb_flat = gripper_rgb.reshape(-1, 3)
        else:
            gripper_rgb_flat = gripper_rgb
    else:
        gripper_pcd_flat = gripper_pcd
        gripper_rgb_flat = gripper_rgb
    
    # Filter out invalid points if depth is provided (for full images)
    # Only use depth-based filtering if the point cloud is in image format (H, W, 3)
    # If it's already downsampled (N, 3), skip depth filtering
    if static_depth is not None and len(static_depth.shape) == 2 and len(static_pcd.shape) == 3:
        # Point cloud is in (H, W, 3) format, can use depth for filtering
        static_valid_mask = (
            (static_depth.flatten() > 0) & 
            ~np.isnan(static_pcd_flat).any(axis=1) &
            ~np.isinf(static_pcd_flat).any(axis=1)
        )
        static_pcd_valid = static_pcd_flat[static_valid_mask]
        if static_rgb_flat is not None:
            static_rgb_valid = static_rgb_flat[static_valid_mask]
        else:
            static_rgb_valid = None
    else:
        # Filter NaN/Inf only (for downsampled point clouds or when depth not provided)
        static_valid_mask = ~np.isnan(static_pcd_flat).any(axis=1) & ~np.isinf(static_pcd_flat).any(axis=1)
        static_pcd_valid = static_pcd_flat[static_valid_mask]
        if static_rgb_flat is not None:
            static_rgb_valid = static_rgb_flat[static_valid_mask]
        else:
            static_rgb_valid = None
    
    if gripper_depth is not None and len(gripper_depth.shape) == 2 and len(gripper_pcd.shape) == 3:
        # Point cloud is in (H, W, 3) format, can use depth for filtering
        gripper_valid_mask = (
            (gripper_depth.flatten() > 0) & 
            ~np.isnan(gripper_pcd_flat).any(axis=1) &
            ~np.isinf(gripper_pcd_flat).any(axis=1)
        )
        gripper_pcd_valid = gripper_pcd_flat[gripper_valid_mask]
        if gripper_rgb_flat is not None:
            gripper_rgb_valid = gripper_rgb_flat[gripper_valid_mask]
        else:
            gripper_rgb_valid = None
    else:
        # Filter NaN/Inf only (for downsampled point clouds or when depth not provided)
        gripper_valid_mask = ~np.isnan(gripper_pcd_flat).any(axis=1) & ~np.isinf(gripper_pcd_flat).any(axis=1)
        gripper_pcd_valid = gripper_pcd_flat[gripper_valid_mask]
        if gripper_rgb_flat is not None:
            gripper_rgb_valid = gripper_rgb_flat[gripper_valid_mask]
        else:
            gripper_rgb_valid = None
    
    # Normalize RGB to [0, 1] if needed
    if static_rgb_valid is not None:
        if static_rgb_valid.max() > 1.0:
            static_rgb_valid = static_rgb_valid / 255.0
    if gripper_rgb_valid is not None:
        if gripper_rgb_valid.max() > 1.0:
            gripper_rgb_valid = gripper_rgb_valid / 255.0
    
    # Check if we're in a headless environment
    display_available = os.environ.get('DISPLAY') is not None
    
    if not HAS_OPEN3D:
        cprint("Open3D not available. Using matplotlib fallback.", "yellow")
        display_available = False  # Force matplotlib
    
    if not display_available and save_path is None:
        cprint("No display available. Use save_path to save point clouds to files.", "yellow")
        return
    
    # Save point clouds to files if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save as PLY files (can be opened in MeshLab, CloudCompare, etc.)
        if HAS_OPEN3D:
            if static_rgb_valid is not None:
                static_pcd_o3d = o3d.geometry.PointCloud()
                static_pcd_o3d.points = o3d.utility.Vector3dVector(static_pcd_valid)
                static_pcd_o3d.colors = o3d.utility.Vector3dVector(static_rgb_valid)
            else:
                static_pcd_o3d = o3d.geometry.PointCloud()
                static_pcd_o3d.points = o3d.utility.Vector3dVector(static_pcd_valid)
            
            if gripper_rgb_valid is not None:
                gripper_pcd_o3d = o3d.geometry.PointCloud()
                gripper_pcd_o3d.points = o3d.utility.Vector3dVector(gripper_pcd_valid)
                gripper_pcd_o3d.colors = o3d.utility.Vector3dVector(gripper_rgb_valid)
            else:
                gripper_pcd_o3d = o3d.geometry.PointCloud()
                gripper_pcd_o3d.points = o3d.utility.Vector3dVector(gripper_pcd_valid)
            
            stage_suffix = f"_{stage}" if stage else ""
            static_ply_path = os.path.join(save_path, f'frame_{frame_idx:06d}_static{stage_suffix}.ply')
            gripper_ply_path = os.path.join(save_path, f'frame_{frame_idx:06d}_gripper{stage_suffix}.ply')
            
            o3d.io.write_point_cloud(static_ply_path, static_pcd_o3d)
            o3d.io.write_point_cloud(gripper_ply_path, gripper_pcd_o3d)
            
            cprint(f"Saved point clouds to {static_ply_path} and {gripper_ply_path}", "green")
        else:
            # Save as numpy arrays
            stage_suffix = f"_{stage}" if stage else ""
            static_npy_path = os.path.join(save_path, f'frame_{frame_idx:06d}_static_pcd{stage_suffix}.npy')
            gripper_npy_path = os.path.join(save_path, f'frame_{frame_idx:06d}_gripper_pcd{stage_suffix}.npy')
            
            if static_rgb_valid is not None:
                np.save(static_npy_path, np.hstack([static_pcd_valid, static_rgb_valid]))
            else:
                np.save(static_npy_path, static_pcd_valid)
            
            if gripper_rgb_valid is not None:
                np.save(gripper_npy_path, np.hstack([gripper_pcd_valid, gripper_rgb_valid]))
            else:
                np.save(gripper_npy_path, gripper_pcd_valid)
            
            cprint(f"Saved point clouds to {static_npy_path} and {gripper_npy_path}", "green")
    
    # Try interactive visualization if display is available
    if display_available and HAS_OPEN3D:
        # Create Open3D point clouds
        if static_rgb_valid is not None:
            static_pcd_o3d = o3d.geometry.PointCloud()
            static_pcd_o3d.points = o3d.utility.Vector3dVector(static_pcd_valid)
            static_pcd_o3d.colors = o3d.utility.Vector3dVector(static_rgb_valid)
        else:
            static_pcd_o3d = o3d.geometry.PointCloud()
            static_pcd_o3d.points = o3d.utility.Vector3dVector(static_pcd_valid)
        
        if gripper_rgb_valid is not None:
            gripper_pcd_o3d = o3d.geometry.PointCloud()
            gripper_pcd_o3d.points = o3d.utility.Vector3dVector(gripper_pcd_valid)
            gripper_pcd_o3d.colors = o3d.utility.Vector3dVector(gripper_rgb_valid)
        else:
            gripper_pcd_o3d = o3d.geometry.PointCloud()
            gripper_pcd_o3d.points = o3d.utility.Vector3dVector(gripper_pcd_valid)
        
        # Translate gripper point cloud slightly for visualization
        gripper_pcd_o3d.translate([0.1, 0, 0])  # Shift right by 0.1m
        
        # Create visualization
        vis_list = [static_pcd_o3d, gripper_pcd_o3d]
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis_list.append(coord_frame)
        
        # Try to visualize with Open3D, fallback to matplotlib if it fails
        window_title = f"{title} - Frame {frame_idx}" + (f" ({stage})" if stage else "")
        try:
            o3d.visualization.draw_geometries(
                vis_list,
                window_name=window_title,
                width=1280,
                height=720,
                point_show_normal=False
            )
        except Exception as e:
            cprint(f"Open3D visualization failed: {e}", "yellow")
            cprint("Falling back to matplotlib 3D visualization...", "yellow")
            display_available = False  # Force matplotlib fallback
    
    # Fallback to matplotlib if Open3D failed or not available
    if not display_available or not HAS_OPEN3D:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(16, 8))
            
            # Static camera point cloud
            ax1 = fig.add_subplot(121, projection='3d')
            if static_rgb_valid is not None:
                scatter1 = ax1.scatter(static_pcd_valid[:, 0], static_pcd_valid[:, 1], static_pcd_valid[:, 2],
                                      c=static_rgb_valid, s=1, alpha=0.6)
            else:
                scatter1 = ax1.scatter(static_pcd_valid[:, 0], static_pcd_valid[:, 1], static_pcd_valid[:, 2],
                                      s=1, alpha=0.6)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title('Static Camera Point Cloud')
            
            # Gripper camera point cloud
            ax2 = fig.add_subplot(122, projection='3d')
            if gripper_rgb_valid is not None:
                scatter2 = ax2.scatter(gripper_pcd_valid[:, 0], gripper_pcd_valid[:, 1], gripper_pcd_valid[:, 2],
                                      c=gripper_rgb_valid, s=1, alpha=0.6)
            else:
                scatter2 = ax2.scatter(gripper_pcd_valid[:, 0], gripper_pcd_valid[:, 1], gripper_pcd_valid[:, 2],
                                      s=1, alpha=0.6)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title('Gripper Camera Point Cloud')
            
            window_title = f"{title} - Frame {frame_idx}" + (f" ({stage})" if stage else "")
            plt.suptitle(f'{window_title}', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                stage_suffix = f"_{stage}" if stage else ""
                plot_path = os.path.join(save_path, f'frame_{frame_idx:06d}_3d_plot{stage_suffix}.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                cprint(f"Saved 3D plot to {plot_path}", "green")
                plt.close()
            else:
                plt.show()
            
        except Exception as e2:
            cprint(f"Matplotlib 3D visualization also failed: {e2}", "red")
            cprint("Skipping 3D point cloud visualization.", "yellow")
    
    # Also show RGB images using matplotlib if provided
    if (static_rgb is not None and len(static_rgb.shape) == 3) or \
       (gripper_rgb is not None and len(gripper_rgb.shape) == 3):
        try:
            import matplotlib
            if not display_available:
                matplotlib.use('Agg')  # Use non-interactive backend for headless
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # Static RGB
            if static_rgb is not None and len(static_rgb.shape) == 3:
                axes[0, 0].imshow(static_rgb)
                axes[0, 0].set_title(f'Static RGB ({static_rgb.shape[0]}x{static_rgb.shape[1]})')
            else:
                axes[0, 0].text(0.5, 0.5, 'No RGB available', ha='center', va='center')
                axes[0, 0].set_title('Static RGB')
            axes[0, 0].axis('off')
            
            # Static Depth
            if static_depth is not None and len(static_depth.shape) == 2:
                im1 = axes[0, 1].imshow(static_depth, cmap='viridis')
                axes[0, 1].set_title(f'Static Depth ({static_depth.shape[0]}x{static_depth.shape[1]})')
                plt.colorbar(im1, ax=axes[0, 1])
            else:
                axes[0, 1].text(0.5, 0.5, 'No depth available', ha='center', va='center')
                axes[0, 1].set_title('Static Depth')
            axes[0, 1].axis('off')
            
            # Gripper RGB
            if gripper_rgb is not None and len(gripper_rgb.shape) == 3:
                axes[1, 0].imshow(gripper_rgb)
                axes[1, 0].set_title(f'Gripper RGB ({gripper_rgb.shape[0]}x{gripper_rgb.shape[1]})')
            else:
                axes[1, 0].text(0.5, 0.5, 'No RGB available', ha='center', va='center')
                axes[1, 0].set_title('Gripper RGB')
            axes[1, 0].axis('off')
            
            # Gripper Depth
            if gripper_depth is not None and len(gripper_depth.shape) == 2:
                im2 = axes[1, 1].imshow(gripper_depth, cmap='viridis')
                axes[1, 1].set_title(f'Gripper Depth ({gripper_depth.shape[0]}x{gripper_depth.shape[1]})')
                plt.colorbar(im2, ax=axes[1, 1])
            else:
                axes[1, 1].text(0.5, 0.5, 'No depth available', ha='center', va='center')
                axes[1, 1].set_title('Gripper Depth')
            axes[1, 1].axis('off')
            
            window_title = f'Frame {frame_idx} - RGB and Depth Images' + (f" ({stage})" if stage else "")
            plt.suptitle(window_title, fontsize=14)
            plt.tight_layout()
            
            if save_path:
                stage_suffix = f"_{stage}" if stage else ""
                img_path = os.path.join(save_path, f'frame_{frame_idx:06d}_images{stage_suffix}.png')
                plt.savefig(img_path, dpi=150, bbox_inches='tight')
                cprint(f"Saved RGB/Depth images to {img_path}", "green")
                plt.close()
            else:
                plt.show()
            
        except ImportError:
            cprint("Matplotlib not available. Skipping RGB image visualization.", "yellow")
