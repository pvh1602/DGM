import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type=None, *args, color_conditioning=False, **kwargs):
        assert mask_type == 'A' or mask_type == 'B' , "Mask type {} is not exist".format(mask_type)
        super().__init__(*args, **kwargs)
        self.color_conditioning = color_conditioning
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def forward(self, x):
        return F.conv2d(x, self.weight*self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k//2, :] = 1
        self.mask[:, :, k//2, :k//2] = 1

        if self.color_conditioning:     # if condition on colors of image
            assert self.in_channels % 3 == 0 and self.out_channels % 3 == 0
            one_third_in, one_third_out = self.in_channels // 3, self.out_channels // 3
            if mask_type == 'B':
                self.mask[:one_third_out, :one_third_in, k // 2, k // 2] = 1
                self.mask[one_third_out:2*one_third_out, :2*one_third_in, k // 2, k // 2] = 1
                self.mask[2*one_third_out:, :, k // 2, k // 2] = 1
            else:
                self.mask[one_third_out:2*one_third_out, :one_third_in, k // 2, k // 2] = 1
                self.mask[2*one_third_out:, :2*one_third_in, k // 2, k // 2] = 1      
        else:          
            if mask_type == 'B':
                self.mask[:, : , k//2, k//2] = 1


class ResBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.block = nn.ModuleList([
            nn.ReLU(),
            MaskConv2d('B', in_channels, in_channels // 2, kernel_size=1, **kwargs),
            nn.ReLU(),
            MaskConv2d('B', in_channels // 2, in_channels // 2, kernel_size=7, padding=3, **kwargs),
            nn.ReLU(),
            MaskConv2d('B', in_channels // 2, in_channels, kernel_size=1, **kwargs)
        ])

    def forward(self, x, skipped=True):
        out = x
        for layer in self.block:
            out = layer(out)
        if skipped:
            # print("check")
            return out + x  

            
        return out  # no skip connection

class LayerNorm(nn.LayerNorm):
    def __init__(self, color_conditioning, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_conditioning = color_conditioning

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.color_conditioning:
            x = x.contiguous().view(*(x_shape[:-1] + (3, -1)))
        x = super().forward(x)
        if self.color_conditioning:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()

class PixelCNN(nn.Module):
    def __init__(self, input_size, device, num_resblock, n_colors,
                    n_filters=64, color_conditioning=False,
                    use_layer_norm=False, use_mixture_logistic=False,
                    skipped=True):
        """

        """
        super().__init__()
        self.channels, self.height, self.width = input_size
        print("self.channels ", self.channels)
        self.device = device
        self.num_resblock = num_resblock
        kwargs = dict(color_conditioning = color_conditioning)

        block_init = lambda: ResBlock(n_filters, **kwargs)
        
        model = nn.ModuleList([MaskConv2d('A', in_channels=self.channels, out_channels=n_filters, 
                                kernel_size=7, padding=3, **kwargs)])

        for _ in range(self.num_resblock):
            if use_layer_norm:
                if color_conditioning:
                    model.append(LayerNorm(color_conditioning, n_filters // 3))
                else:
                    model.append(LayerNorm(color_conditioning, n_filters))
            model.extend([
                nn.ReLU(),
                block_init()
            ])

        model.extend([nn.ReLU(), 
                    MaskConv2d('B', n_filters, n_filters, kernel_size=1, **kwargs)
                    ])
        model.extend([nn.ReLU(), 
                    MaskConv2d('B', n_filters, n_colors * self.channels, kernel_size=1, **kwargs)
                    ])

        self.net = model
        self.input_shape = input_size
        self.n_colors = n_colors
        print("self.n_colors ", self.n_colors)
        self.color_conditioning = color_conditioning
        self.skipped = skipped

    def forward(self, x):
        # return F.sigmoid(self.net(x))
        batch_size = x.shape[0]
        out = (x.float() / (self.n_colors - 1) - 0.5) / 0.5
        for layer in self.net:
            if isinstance(layer, ResBlock):
                out = layer(out, self.skipped)
            out = layer(out)
            # print("out's size ", out.size())
        
        if self.color_conditioning:
            return out.view(batch_size, self.channels, self.n_colors, 
                            *self.input_shape[1:]).permute(0, 2, 1, 3, 4)
        else:
            return out.view(batch_size, self.n_colors, *self.input_shape)
    
    def loss(self, x):
        outputs = self(x)
        # print(x.size())
        # print(x[0])
        # print(outputs.size())
        # print(outputs[0])
        return F.cross_entropy(outputs, x.long())


    def sample(self, num_sample):
        """
        num_sample: the number of sample in each generating time
        """
        samples = torch.zeros(num_sample, self.channels, self.height, self.width).to(self.device)
        with torch.no_grad():
            
            for r in range(self.height):
                for c in range(self.width):
                    for k in range(self.channels):
                        logits = self(samples)[:,:,k,r,c]
                        probs = F.softmax(logits, dim=1)
                        samples[:,k,r,c] = torch.multinomial(probs,1).squeeze(-1)
        samples = samples.permute(0,2,3,1).cpu().numpy()
        # print(samples[0])
        return samples

        
        












