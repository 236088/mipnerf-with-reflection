import torch
import torch.nn as nn
from ray_utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map, cast_rays
from pose_utils import to8b
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None]**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret


class ReflectNeRF(nn.Module):
    def __init__(self,
                 randomized=False,
                 ray_shape="cone",
                 bkgd=torch.ones([3]),
                 num_levels=2,
                 num_samples=128,
                 hidden=256,
                 density_noise=1,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 min_deg=0,
                 max_deg=16,
                 viewdirs_min_deg=0,
                 viewdirs_max_deg=4,
                 device=torch.device("cpu")
                 ):
        super(ReflectNeRF, self).__init__()
        self.init_randomized = randomized
        self.randomized = randomized
        self.ray_shape = ray_shape
        self.bkgd = bkgd
        self.num_levels = num_levels
        self.num_samples = num_samples
        self.density_input = (max_deg - min_deg) * 3 * 2
        self.rgb_input = 3 + ((viewdirs_max_deg - viewdirs_min_deg) * 3 * 2)
        self.density_noise = density_noise
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.density_bias = density_bias
        self.hidden = hidden
        self.device = device
        self.density_activation = nn.Softplus()

        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.density_net0 = nn.Sequential(
            nn.Linear(self.density_input, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.density_net1 = nn.Sequential(
            nn.Linear(self.density_input + hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        
        self.final_density = nn.Sequential(
            nn.Linear(hidden, 1),
        )
        self.final_normal = nn.Sequential(
            nn.Linear(hidden, 3),
        )
        self.final_ks = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )


        self.rgb_net0 = nn.Sequential(
            nn.Linear(hidden, hidden)
        )
        self.viewdirs_encoding = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
        self.rgb_net1 = nn.Sequential(
            nn.Linear(hidden + self.rgb_input, hidden),
            nn.ReLU(True),
        )
        self.final_rgb = nn.Sequential(
            nn.Linear(hidden, 3),
            nn.Sigmoid()
        )
        _xavier_init(self)
        self.to(device)

    # def render_ray(self, rays):
    
    # density gradient vector is used for learning mlp normal
        
    def forward(self, rays, enable_second_ray=True):
        comp_rgbs = []
        distances = []
        accs = []
        weights = []
        
        comp_kss = []
        comp_normals = []
        comp_penalties = []
        
        for l in range(self.num_levels):
            # sample
            if l == 0:  # coarse grain sample
                t_vals, (mean, var) = sample_along_rays(rays.origins, rays.directions, rays.radii, self.num_samples,
                                                        rays.near, rays.far, randomized=self.randomized, lindisp=False,
                                                        ray_shape=self.ray_shape)
            else:  # fine grain sample/s
                t_vals, (mean, var) = resample_along_rays(rays.origins, rays.directions, rays.radii,
                                                          t_vals.to(rays.origins.device),
                                                          weight.to(rays.origins.device), randomized=self.randomized,
                                                          stop_grad=True, resample_padding=self.resample_padding,
                                                          ray_shape=self.ray_shape)
            mean.requires_grad = True
            
            # do integrated positional encoding of samples
            samples_enc = self.positional_encoding(mean, var)[0]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])
            
            # predict density
            new_encodings = self.density_net0(samples_enc)
            new_encodings = torch.cat((new_encodings, samples_enc), -1)
            new_encodings = self.density_net1(new_encodings)
            raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1))
            raw_normal = self.final_normal(new_encodings).reshape((-1, self.num_samples, 3))
            raw_ks = self.final_ks(new_encodings).reshape((-1, self.num_samples, 1))

            # normal = nn.functional.normalize(raw_normal, dim=-1, eps=1e-8)
            
            # predict rgb
            #  do positional encoding of viewdirs
            viewdirs = self.viewdirs_encoding(rays.viewdirs.to(self.device))
            viewdirs = torch.cat((viewdirs, rays.viewdirs.to(self.device)), -1)
            viewdirs = torch.tile(viewdirs[:, None, :], (1, self.num_samples, 1))
            viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))
            new_encodings = self.rgb_net0(new_encodings)
            new_encodings = torch.cat((new_encodings, viewdirs), -1)
            new_encodings = self.rgb_net1(new_encodings)
            raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.num_samples, 3))

            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

            # volumetric rendering
            rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            
            density = self.density_activation(raw_density + self.density_bias)
            
            # visibility = torch.clamp(torch.sum(rays.viewdirs.to(self.device)[...,None,:]  * raw_normal, dim=-1), min=0, max=0)
            comp_rgb, comp_normal, comp_ks, distance, acc, weight, alpha = volumetric_rendering(rgb, raw_normal, raw_ks, density, t_vals, rays.directions.to(rgb.device), self.bkgd)
            
            
            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)
            weights.append(weight)
            
            comp_normals.append(comp_normal)
            comp_kss.append(comp_ks)
            if self.training:
                comp_penalties.append(raw_ks*torch.exp(-density))
                   
        # ks is [0,1] weight of high frequancy to low frequancy interpolation, simultaneously, sharpness value [0, 1/2^(max_deg-1)]
        # 
        # refrays = Rays(
        #     origins=origins,
        #     directions=refdirs,
        #     viewdirs=refdirs,
        #     radii=radii + ks/2^4,
        #     lossmult=ones,
        #     near=ones * (distance+eps),
        #     far=ones * self.far)
        #
        # bkgd = env_net(encording)?
        
        if enable_second_ray:
            for l in range(self.num_levels):
                normal = comp_normals[l].detach()
                origins = rays.origins - 2*torch.sum((distance[...,None]*rays.directions)*normal, dim=-1, keepdim=True)*normal
                refdirs = rays.viewdirs - 2*torch.sum(rays.viewdirs*normal, dim=-1, keepdim=True)*normal
                
                ks = comp_ks[l]
                radii = rays.radii + (1-ks)/2**(self.viewdirs_encoding.max_deg-1)
                
                # sample
                if l == 0:  # coarse grain sample
                    t_vals, (mean, var) = sample_along_rays(origins, refdirs, radii, self.num_samples,
                                                            distance[...,None], rays.far, randomized=self.randomized, lindisp=False,
                                                            ray_shape=self.ray_shape)
                else:  # fine grain sample/s
                    t_vals, (mean, var) = resample_along_rays(origins, refdirs, radii, 
                                                            t_vals.to(rays.origins.device),
                                                            weight.to(rays.origins.device), randomized=self.randomized,
                                                            stop_grad=True, resample_padding=self.resample_padding,
                                                            ray_shape=self.ray_shape)
                                
                # do integrated positional encoding of samples
                samples_enc = self.positional_encoding(mean, var)[0]
                samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])

                # predict density
                new_encodings = self.density_net0(samples_enc)
                new_encodings = torch.cat((new_encodings, samples_enc), -1)
                new_encodings = self.density_net1(new_encodings)
                raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1))
                raw_normal = self.final_normal(new_encodings).reshape((-1, self.num_samples, 3))

                # predict rgb
                #  do positional encoding of viewdirs
                viewdirs = self.viewdirs_encoding(refdirs.to(self.device))
                viewdirs = torch.cat((viewdirs, refdirs.to(self.device)), -1)
                viewdirs = torch.tile(viewdirs[:, None, :], (1, self.num_samples, 1))
                viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))
                new_encodings = self.rgb_net0(new_encodings)
                new_encodings = torch.cat((new_encodings, viewdirs), -1)
                new_encodings = self.rgb_net1(new_encodings)
                raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.num_samples, 3))

                # Add noise to regularize the density predictions if needed.
                if self.randomized and self.density_noise:
                    raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

                # volumetric rendering
                rgbs = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
                density = self.density_activation(raw_density + self.density_bias)
                
                # visibility = torch.clamp(torch.sum(refdirs.to(self.device)[...,None,:] * raw_normal, dim=-1), min=0, max=0)
                comp_rgb, distance, acc, weight, alpha = volumetric_rendering(rgbs, None, None, density, t_vals, refdirs.to(rgbs.device), self.bkgd)

                comp_rgbs[l]=(1 - ks)*comp_rgbs[l] + ks*comp_rgb
        
        if self.training:
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), torch.stack(weights), torch.stack(comp_kss), torch.stack(comp_normals), torch.stack(comp_penalties)
        else:
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), torch.stack(weights), torch.stack(comp_kss), torch.stack(comp_normals)
        
    def render_image(self, rays, height, width, chunks=8192):
        """
        Return image, disparity map, accumulated opacity (shaped to height x width) created using rays as input.
        Rays should be all of the rays that correspond to this one single image.
        Batches the rays into chunks to not overload memory of device
        """
        length = rays[0].shape[0]
        rgbs = []
        dists = []
        normals = []
        accs = []
        with torch.no_grad():
            for i in tqdm(range(0, length, chunks), leave=False):
                # put chunk of rays on device
                chunk_rays = namedtuple_map(lambda r: r[i:i+chunks].to(self.device), rays)
                rgb, distance, acc, weight, ks, normal = self(chunk_rays)
                rgbs.append(rgb[-1].cpu())
                dists.append(distance[-1].cpu())
                normals.append(normal[-1].cpu())
                accs.append(acc[-1].cpu())

        rgbs = to8b(torch.cat(rgbs, dim=0).reshape(height, width, 3).numpy())
        dists = torch.cat(dists, dim=0).reshape(height, width).numpy()
        normals = torch.cat(normals, dim=0).reshape(height, width, 3).numpy()
        accs = torch.cat(accs, dim=0).reshape(height, width).numpy()
        return rgbs, dists, normals, accs

    def train(self, mode=True):
        self.randomized = self.init_randomized
        super().train(mode)
        return self

    def eval(self):
        self.randomized = False
        return super().eval()

def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
