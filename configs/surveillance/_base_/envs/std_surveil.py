from torch import pi as PI

# 感光元件尺寸
# 1/4 inches: 靶面尺寸约为3.2mm x 2.4mm, 常用于小型监控设备, 如家庭安全摄像头.
# 1/3 inches: 靶面尺寸约为4.8mm x 3.6mm, 适用于普通监控摄像头, 常见于室内和室外的一般监控场景.
# 1/2 inches: 靶面尺寸约为6.4mm x 4.8mm, 用于需要较高分辨率的监控设备, 如商场、银行等.
# 2/3 inches: 靶面尺寸约为8.8mm x 6.6mm, 常用于专业级监控和广播级摄像机, 适合远距离监控和低光照环境.
# 1 inch: 靶面尺寸约为12.7mm x 9.6mm, 用于高端监控设备和专业摄影机, 提供高图像质量和细节.

clips = dict(  # meter
    common=[4.8e-3, 3.6e-3],
)

# 普通监控摄像头焦距
# 2.8mm: 提供较宽的视角, 适合覆盖较大的区域, 如会议室或大型开放空间.
# 3.6mm: 提供较宽的视角, 适用于室内环境, 如商店或走廊.
# 4mm: 中等视角, 适合中等大小的房间或室外街道监控.
# 6mm: 较窄的视角, 适用于需要更集中监控的区域, 如门口或特定走廊.
# 8mm: 更窄的视角, 适合远距离监控, 如停车场或街道对面的建筑物.

focals = dict(  # meter
    mid=4e-3,
)

# 普通监控摄像头分辨率
# CIF（Common Intermediate Format）: 352x288像素, 是一种较旧的分辨率标准, 适用于基本的监控需求.
# D1: 720x576像素, 是标清（SD）分辨率, 提供比CIF更清晰的图像.
# HD（High Definition）: 1280x720像素, 即720p, 提供高清图像, 是当前较为常见的分辨率.
# Full HD: 1920x1080像素, 即1080p, 提供全高清图像, 是当前主流的高清监控分辨率.
# 2MP（Megapixel）: 1920x1080像素, 与Full HD相同, 但通常用于描述网络摄像头.
# 3MP: 2048x1536像素, 提供比Full HD更高的分辨率.
# 4MP: 2560x1440像素, 接近于4K分辨率, 提供非常清晰的图像.
# 5MP: 2592x1944像素, 提供比4MP更高的分辨率.
# 4K: 3840x2160像素, 提供超高清晰度的图像, 适用于需要极高图像质量的场合.

resols = {  # pixel
    "LR": [352, 288],
    "SD": [720, 576],
    "HD": [1280, 720],
    "1080p": [1920, 1080],
}

cam_intrinsics = dict(
    # `template`
    # cam_name=dict(
    #   param_name=param_dict[specific_param_type],
    # )
    # `std_cam`
    # std_cam=dict(
    #     clip_shape=clips["common"],
    #     focal_length=focals["mid"],
    #     resolution=resols["SD"],
    # ),
    # `std_cam`
    std_cam=dict(
        aov=[PI / 3, PI / 3],
        dof=[3, 0.5],
    ),
)


point_configs = dict(
    # point_type=dict(
    #   color="",  # this is necessary
    #   extra_params=[
    #       # this is optional
    #   ],
    # )
    occupancy=dict(
        color="grey",
    ),
    install_permitted=dict(
        color="yellow",
    ),
    must_monitor=dict(
        color="red",
        extra_params=[
            "h_res_req_min",  # pixels per meter
            "h_res_req_max",  # pixels per meter
            "v_res_req_min",  # pixels per meter
            "v_res_req_max",  # pixels per meter
        ],
    ),
    camera=dict(
        color="blue",
        extra_params=[
            "pan",  # [-pi, pi)
            "tilt",  # [-pi/2, pi/2]
        ],
    ),
    visible=dict(
        color="green",
    ),
)

voxel_length = 0.1  # meter
