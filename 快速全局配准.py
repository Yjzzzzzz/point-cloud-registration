import open3d as o3d
import time
import copy
import numpy as np

# ----------传入点云数据，计算FPFH-----------
def fpfh_compute(pcd):
    radius_normal = 0.01  # kdtree参数，用于估计法线的半径，
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # 估计法线的1个参数，使用混合型的kdtree，半径内取最多30个邻居
    radius_feature = 0.25  # kdtree参数，用于估计FPFH特征的半径
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    # 计算FPFH特征,搜索方法kdtree
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                               o3d.geometry.KDTreeSearchParamHybrid
                                                               (radius=radius_feature, max_nn=50))
    return pcd_fpfh  # 返回FPFH特征


# ------------FastGlobalRegistration配准---------------
def execute_fast_global_registration(source, target, source_fpfh, target_fpfh):  # 传入两个点云和点云的特征
    distance_threshold = 0.5  # 设定距离阈值
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result =  o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


# ----------------可视化配准结果----------------
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)         # 由于函数transformand paint_uniform_color会更改点云，
    target_temp = copy.deepcopy(target)         # 因此调用copy.deepcoy进行复制并保护原始点云。
    source_temp.paint_uniform_color([1, 0, 0])  # 点云着色
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)
    # o3d.io.write_point_cloud("trans_of_source.pcd", source_temp)  # 保存配准后的点云
    o3d.visualization.draw_geometries([source_temp, target_temp], width=600, height=600, mesh_show_back_face=False)


#  --------------------读取点云数据------------------
source = o3d.io.read_point_cloud("./nangua.ply")
target = o3d.io.read_point_cloud("./nangua1.ply")
start = time.time()
#  -----------计算源点云和目标点云的FPFH-------------
source_fpfh = fpfh_compute(source)
target_fpfh = fpfh_compute(target)
# ------------------调用FGR执行粗配准----------------
print("开始fgr粗配准")
result_fast = execute_fast_global_registration(source, target,
                                               source_fpfh, target_fpfh)
print("Fast global registration took %.3f sec.\n" % (time.time() - start))
print(result_fast)
draw_registration_result(source, target, result_fast.transformation)  # 源点云旋转平移到目标点云
# -----------------点到面ICP进行精配准------------
loss = o3d.pipelines.registration.L2Loss() #鲁棒核函数的调用方法(具有如下几种形式：CauchyLoss、GMLoss、HuberLoss、L1Loss、L2Loss、TukeyLoss)
print("开始点到面ICP精配准")
icp_p2plane = o3d.pipelines.registration.registration_icp(
        source, target, 5, result_fast.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),    # 执行点到面的ICP算法
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))  # 设置最大迭代次数
print(icp_p2plane)   # 输出ICP相关信息
print("Transformation is:")
print(icp_p2plane.transformation)  # 输出变换矩阵
draw_registration_result(source, target, icp_p2plane.transformation)
# -----------------点到点ICP进行精配准------------
icp_p2point = o3d.pipelines.registration.registration_icp(
        source, target, 5, result_fast.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),    # 执行点到点的ICP算法
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))  # 设置最大迭代次数
print(icp_p2point)   # 输出ICP相关信息
print("Transformation is:")
print(icp_p2point.transformation)  # 输出变换矩阵
draw_registration_result(source, target, icp_p2point.transformation)
# -----------------GICP进行精配准------------
print("Apply GeneralizedICP")
generalized_icp = o3d.pipelines.registration.registration_generalized_icp(
    source, target, 5, result_fast.transformation,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35))  # 设置最大迭代次数
print(generalized_icp)  # 输出GeneralizedICP相关信息
print("Transformation is:")
print(generalized_icp.transformation)  # 输出变换矩阵
draw_registration_result(source, target, generalized_icp.transformation)#可视化配准结果
# ---------------将两个点云进行拼接------------------
pcd_connect = o3d.geometry.PointCloud()
pcd_connect.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(source.points),
                                                                np.asarray(target.points))))  # 拼接点云