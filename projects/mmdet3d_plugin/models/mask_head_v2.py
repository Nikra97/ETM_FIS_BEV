class MaskHeadSmallConvIFC_V3(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, n_future=3, output_dict=None):
        super().__init__()

        # inter_dims = [dim, context_dim // 2, context_dim // 4,
        #               context_dim // 8, context_dim // 16, context_dim // 64, context_dim // 128]
        self.n_future = n_future
        gn = 8
        T = self.n_future
        fpn_dims = [256,256,256,512]
        
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(gn, dim)
        self.lay2 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(gn, dim)
        self.lay3 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(gn, dim)
        self.lay4 = torch.nn.Conv2d(dim, dim*T, 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(gn, dim*T)
        self.lay5 = torch.nn.Conv2d(dim*2, dim*T, 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(gn, dim*T)

        self.a = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=[1, 1, 1], bias=False),
            nn.BatchNorm3d(
                num_features=dim, eps=1e-5, momentum=0.1
            ),
            nn.ReLU(inplace=True),
        )

        # Depthwise (channel-separated) 3x3x3x1 conv
        # Depthwise (channel-separated) 1x3x3x1 spatial conv
        self.b1 = nn.Conv3d(
            dim,
            dim,
            kernel_size=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 1, 1],
            bias=False,
        )
        # Depthwise (channel-separated) 3x1x1x1 temporal conv
        self.b2 = nn.Conv3d(
            dim,
            dim,
            kernel_size=[3, 1, 1],
            stride=[1, 1, 1],
            padding=[1, 0, 0],
            bias=False,
        )

        self.convert_to_weight = MLP(dim, dim, dim*T, 2)

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], dim, 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], dim, 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], dim, 1)
        self.adapter4 = torch.nn.Conv2d(fpn_dims[3], dim*T, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, seg_memory, fpns, hs):
        x = seg_memory
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter1(fpns[-1])
        
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter2(fpns[-2])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter3(fpns[-3])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter4(fpns[-4])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")

        T = self.n_future
        H, W = x.shape[-2:]
        B = x.shape[0]

        x = x.unsqueeze(1).reshape(B, -1, T, H, W)
        # D2+1 Module from Vov3D -> Basically replacing a 3D Convolution 
        x = self.b1(x)
        x = self.b2(x)
        x = F.relu(x)
        x = self.a(x).permute(0, 2, 1, 3, 4)
    
        B, BT, C, H, W = x.shape
        L, B, N, C = hs.shape

        w = self.convert_to_weight(hs).permute(1, 0, 2, 3)
        w = w.unsqueeze(1).reshape(B, T, L, N, -1)

        # Unsure about the fusion across the batch dimension in  W maybe x needs to be x.reshape(1,B*BT*C, H, W),
        mask_logits = F.conv2d(x.reshape(B, BT*C, H, W),
                            w.reshape(B*T*L*N, C, 1, 1), groups=BT*B)

        mask_logits = mask_logits.view(
            B, T, L, N, H, W).permute(2, 0, 3, 1, 4, 5)
        return mask_logits

########## IFC ########################################
class MaskHead(nn.Module):
    def __init__(self, hidden_dim, fpn_dims, num_frames):
        super().__init__()
        self.num_frames = num_frames

        self.lay1 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(32, hidden_dim)
        self.lay2 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(32, hidden_dim)
        self.lay3 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(32, hidden_dim)
        self.out_lay = DepthwiseSeparableConv2d(
            hidden_dim, hidden_dim, 5, padding=2, activation1=F.relu, activation2=F.relu)

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], hidden_dim, 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], hidden_dim, 1)

        self.convert_to_weight = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, fpns: List[Tensor], tq: Tensor):
        x = self.lay1(x)
        x = self.gn1(x)

        cur_fpn = self.adapter1(fpns[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)
        x = self.gn2(x)

        cur_fpn = self.adapter2(fpns[1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        BT, C, H, W = x.shape
        L, B, N, C = tq.shape
        T = BT // B

        x = self.out_lay(x)
        w = self.convert_to_weight(tq).permute(1, 0, 2, 3)
        w = w.unsqueeze(1).repeat(1, T, 1, 1, 1)

        mask_logits = F.conv2d(x.view(1, BT*C, H, W),
                               w.reshape(B*T*L*N, C, 1, 1), groups=BT)
        mask_logits = mask_logits.view(
            B, T, L, N, H, W).permute(2, 0, 3, 1, 4, 5)

        return mask_logits
