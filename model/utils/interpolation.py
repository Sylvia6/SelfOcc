import torch


def linear_interpolation(feat, grid):
    # feat: B, C, W
    # grid: B, N   normlized to [-1, 1]
    assert len(feat.shape) == 3
    assert feat.shape[0] == grid.shape[0]
    feat = feat.unsqueeze(2) # B, C, 1, W
    grid = torch.stack([grid, torch.zeros_like(grid)], dim=2).unsqueeze(-2) # B, N, 1, 2
    out = torch.nn.functional.grid_sample(feat, grid, align_corners=False) # B, C, N, 1
    return out.squeeze(-1) # B, C, N


def bilinear_interpolation(feat, grid):
    # feat: B, C, H, W
    # grid: B, N, 2  normlized to [-1, 1] in order x, y
    assert len(feat.shape) == 4
    assert feat.shape[0] == grid.shape[0]
    assert grid.shape[-1] == 2
    grid = grid.unsqueeze(-2) # B, N, 1, 2
    out = torch.nn.functional.grid_sample(feat, grid, align_corners=False) # B, C, N, 1
    return out.squeeze(-1) # B, C, N


def trilinear_interpolation(feat, grid):
    # feat: B, C, D, H, W
    # grid: B, N, 3  normlized to [-1, 1] in order x, y, z
    assert len(feat.shape) == 5
    assert feat.shape[0] == grid.shape[0]
    assert grid.shape[-1] == 3
    grid = grid.unsqueeze(-2).unsqueeze(-2)    # B, N, 1, 1, 3
    out = torch.nn.functional.grid_sample(feat, grid, align_corners=False) # B, C, N, 1, 1
    return out.squeeze(-1).squeeze(-1) # B, C, N


def quadlinear_interpolation_3_1(feat, grid):
    # feat: B, C, T, D, H, W
    # grid: B, N, 4  normlized to [-1, 1]  in order x, y, z, t
    assert len(feat.shape) == 6
    assert feat.shape[0] == grid.shape[0]
    assert grid.shape[-1] == 4
    B, C, T, D, H, W = feat.shape
    _, N, _ = grid.shape

    grid3d = grid[:, :, 1:] # B, N, 3
    grid1d = grid[:, :, 0] # B, N
    feat = feat.reshape(B, C*T, D, H, W) # B, C*T, D, H, W
    feat = trilinear_interpolation(feat, grid3d) # B, C*T, N
    feat = feat.reshape(B, C, T, N).permute(0, 3, 1, 2) # B, N, C, T
    feat = feat.reshape(B*N, C, T) # B*N, C, T
    grid1d = grid1d.reshape(B*N, 1) # B*N, 1
    feat = linear_interpolation(feat, grid1d) # B*N, C, 1
    return feat.reshape(B, N, C).permute(0, 2, 1) # B, C, N


def quadlinear_interpolation_2_2(feat, grid):
    # feat: B, C, T, D, H, W
    # grid: B, N, 4  normlized to [-1, 1]
    assert len(feat.shape) == 6
    assert feat.shape[0] == grid.shape[0]
    assert grid.shape[-1] == 4
    B, C, T, D, H, W = feat.shape
    _, N, _ = grid.shape

    grid2d_up = grid[:, :, 2:] # B, N, 2
    grid2d_down = grid[:, :, :2] # B, N, 2
    feat = feat.reshape(B, C*T*D, H, W) # B, C*T*D, H, W
    feat = bilinear_interpolation(feat, grid2d_up) # B, C*T*D, N
    feat = feat.permute(0,2,1).reshape(B, N, C, T, D) # B, N, C, T, D
    feat = feat.reshape(B*N, C, T, D) # B*N, C, T, D
    grid2d_down = grid2d_down.reshape(B*N, 1, 2) # B*N, 1, 2
    feat = bilinear_interpolation(feat, grid2d_down) # B*N, C, 1
    return feat.reshape(B, N, C).permute(0, 2, 1) # B, C, N


def quadlinear_interpolation(feat, grid, memory_safe=True):
    if memory_safe:
        return quadlinear_interpolation_3_1(feat, grid)
    else:
        return quadlinear_interpolation_2_2(feat, grid)



def __test_latency():
    """
    cpu:--------
    3_1 0:00:00.010080
    2_2 0:00:00.879021
    gpu test:--------
    3_1 0:00:00.000096
    2_2 0:00:00.000060
    """
    feat = torch.rand(10, 64, 10, 20, 100, 100)
    grid = torch.zeros(10, 100, 4)
    from datetime import datetime
    def timeit(N):
        tic = datetime.now()
        for i in range(N):
            quadlinear_interpolation_3_1(feat, grid)
        toc = datetime.now()
        print("3_1", (toc - tic)/N)

        tic = datetime.now()
        for i in range(N):
            quadlinear_interpolation_2_2(feat, grid)
        toc = datetime.now()
        print("2_2", (toc - tic)/N)
    print("cpu:--------")
    timeit(10)
    print("gpu warmup:--------")
    feat = feat.cuda()
    grid = grid.cuda()
    timeit(10)

    print("gpu test:--------")
    timeit(100)


def __test_quadlinear_interpolation():
    feat = torch.ones(10, 64, 10, 20, 100, 100) # B, C, T, D, H, W
    grid = torch.zeros(10, 100, 4)
    assert quadlinear_interpolation_3_1(feat, grid)[0,0,0] == 1
    assert quadlinear_interpolation_2_2(feat, grid)[0,0,0] == 1
    grid = torch.ones(10, 100, 4) * -1
    assert quadlinear_interpolation_3_1(feat, grid)[0,0,0] == 1.0/16
    assert quadlinear_interpolation_2_2(feat, grid)[0,0,0] == 1.0/16
    print("quadlinear_interpolation test passed")


if __name__ == "__main__":
    __test_quadlinear_interpolation()
    __test_latency()