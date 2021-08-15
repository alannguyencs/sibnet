from constants import *

MIDDLE_DEPTH = 8
BINARY_DEPTH = 2

def get_shift_operator(shift_point):
    shift_base = [[[0. for _ in range(3)] for _ in range(3)]]
    shift_base[0][shift_point[0]][shift_point[1]] = 1.0
    shift_operator = torch.tensor([shift_base for _ in range(MIDDLE_DEPTH)]).to(device)
    return shift_operator

shift_left_operator = get_shift_operator(shift_point=(1, 2))
shift_top_operator = get_shift_operator(shift_point=(2, 1))
shift_top_left_operator = get_shift_operator(shift_point=(2, 2))
shift_top_right_operator = get_shift_operator(shift_point=(2, 0))

def get_conv_sibling(depth_in, depth_out):
    return nn.Sequential(
                nn.Conv2d(depth_in, depth_out, kernel_size=3, stride=1, padding=1),
            )


def mid_conv_layer(depth_in, depth_out, is_final=False):
    if is_final:
        return nn.Sequential(
                nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
            )
    else:
        return nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU(),
                )

def deconv_layer(depth_in, depth_out, is_final=False, output_size=-1):
    if is_final:
        if output_size == -1:
            return nn.Sequential(
                        nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                )
        else:
            return nn.Sequential(
                        nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                        nn.Upsample(size=output_size, mode='bilinear'),
                )

    else:
        if output_size == -1:
            return nn.Sequential(
                        nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(depth_out),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                )
        else:
            return nn.Sequential(
                        nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(depth_out),
                        nn.ReLU(),
                        nn.Upsample(size=output_size, mode='bilinear'),
                )

class Arch(nn.Module):
    def __init__(self):
        super(Arch, self).__init__()
        self.name = 'SIBNET_v6'
        #resnet hyper params
        RESNET_DEPTH_3 = 512
        RESNET_DEPTH_4 = 1024
        RESNET_DEPTH_5 = 2048

 
        resnet_model = torchvision.models.resnet50(pretrained=True)

        self.conv123 = nn.Sequential(*list(resnet_model.children())[0:6]) #0
        self.conv4 = nn.Sequential(*list(resnet_model.children())[6]) #1

        #counting
        self.conv_cnt = nn.Sequential(*list(resnet_model.children())[7]) #5
        # self.AvgPool2d = nn.AvgPool2d(kernel_size=7, stride=1, padding=0) #6
        self.AvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1)) #6
        self.fc = nn.Linear(in_features=2048, out_features=1, bias=True) #7

        #continue deconv half branch
        self.deconv4_half = deconv_layer(RESNET_DEPTH_4, MIDDLE_DEPTH) #8
        self.mid3_half = mid_conv_layer(RESNET_DEPTH_3, MIDDLE_DEPTH) #9
        self.deconv3_half = deconv_layer(MIDDLE_DEPTH * 2, BINARY_DEPTH, is_final=True, output_size=256) #10

        #full        
        self.deconv4_full = deconv_layer(RESNET_DEPTH_4, MIDDLE_DEPTH)
        self.mid3_full = mid_conv_layer(RESNET_DEPTH_3, MIDDLE_DEPTH)
        self.deconv3_full = deconv_layer(MIDDLE_DEPTH * 2, MIDDLE_DEPTH, output_size=256)

        #sibling branch
        self.conv_left = get_conv_sibling(MIDDLE_DEPTH * 2, BINARY_DEPTH)
        self.conv_top = get_conv_sibling(MIDDLE_DEPTH * 2, BINARY_DEPTH)
        self.conv_top_left = get_conv_sibling(MIDDLE_DEPTH * 2, BINARY_DEPTH)
        self.conv_top_right = get_conv_sibling(MIDDLE_DEPTH * 2, BINARY_DEPTH)

        self.conv_semantic = mid_conv_layer(MIDDLE_DEPTH, BINARY_DEPTH, is_final=True) #19



    def forward(self, x):
        #conv
        conv_out3 = self.conv123(x)
        conv_out4 = self.conv4(conv_out3)

        #counting
        counting = self.conv_cnt(conv_out4)
        counting = self.AvgPool2d(counting)
        counting = torch.squeeze(counting)
        counting = self.fc(counting)

        #continue deconv half branch
        deconv_out4_half = self.deconv4_half(conv_out4)
        mid3_half = self.mid3_half(conv_out3)
        fused3_half = torch.cat([deconv_out4_half, mid3_half], 1)
        final_half = self.deconv3_half(fused3_half)

        #sibling branch
        deconv_out4_full = self.deconv4_full(conv_out4)
        mid3_full = self.mid3_full(conv_out3)
        fused3_full = torch.cat([deconv_out4_full, mid3_full], 1)
        out = self.deconv3_full(fused3_full)

        #segmentation and sibling
        out_left = F.conv2d(out, shift_left_operator, padding=1, groups=MIDDLE_DEPTH)
        out_top = F.conv2d(out, shift_top_operator, padding=1, groups=MIDDLE_DEPTH)
        out_top_left = F.conv2d(out, shift_top_left_operator, padding=1, groups=MIDDLE_DEPTH)
        out_top_right = F.conv2d(out, shift_top_right_operator, padding=1, groups=MIDDLE_DEPTH)

        fused_left = torch.cat([out, out_left], 1)
        fused_top = torch.cat([out, out_top], 1)
        fused_top_left = torch.cat([out, out_top_left], 1)
        fused_top_right = torch.cat([out, out_top_right], 1)

        output_semantic = self.conv_semantic(out)
        output_left = self.conv_left(fused_left)
        output_top = self.conv_top(fused_top)
        output_top_left = self.conv_top_left(fused_top_left)
        output_top_right = self.conv_top_right(fused_top_right)

        return final_half, counting, [output_semantic, output_left, output_top, output_top_left, output_top_right]