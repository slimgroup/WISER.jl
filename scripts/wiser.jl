using DrWatson
@quickactivate "WISER"
import Pkg; Pkg.instantiate()
using JUDI
using ArgParse
using Random
using LineSearches
using InvertibleNetworks, Flux, UNet
using PyPlot,SlimPlotting
using LinearAlgebra, Random, Statistics
using ImageQualityIndexes
using BSON, JLD2
using Statistics, Images
using FFTW
using LinearAlgebra
using Random
using ImageGather
Random.seed!(2023)
include("utils.jl")

# Parse command-line arguments
parsed_args = parse_commandline()
startidx = parsed_args["startidx"]
endidx = parsed_args["endidx"]
n_offsets = parsed_args["n_offsets"]
offset_start = parsed_args["offset_start"]
offset_end = parsed_args["offset_end"]
keep_offset_num = parsed_args["keep_offset_num"]

fwi_only = parsed_args["fwi_only"]
resample_x = parsed_args["resample_x"]
redraw = parsed_args["redraw"]
amplitude = parsed_args["amplitude"]
test_snr = parsed_args["test_snr"]
wiser_batch_size = parsed_args["wiser_batch_size"]
inner_loop = parsed_args["inner_loop"]
lr_pre = parsed_args["lr_wiser"]

fc = parsed_args["fc"]
freq1 = parsed_args["freq1"]
freq2 = parsed_args["freq2"]
freq3 = parsed_args["freq3"]
iter1 = parsed_args["iter1"]
iter2 = parsed_args["iter2"]
iter3 = parsed_args["iter3"]

λ = parsed_args["lambda"] # l2 penalty on latent space
γ = parsed_args["gamma"] # some noise to help noisy gradient

if amplitude != 0f0
	configure_type = "ood"
else
	configure_type = "in-dist"
end

if fwi_only
	mode = "fwi"
else
	mode = "wiser"
end

# Plotting configs
background_type = "1d-average"
op_type = "seismic"
rtm_type = "ext-rtm"
sim_name = "$(mode)-$(configure_type)-freqcont=$(fc)-resample_x=$(resample_x)-redraw=$(redraw)-$(op_type)-$(rtm_type)-$(background_type)"
plot_path = plotsdir(sim_name)

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=100)
PyPlot.rc("font", family="serif");

data_path = datadir("m_train_compass.jld2")
if ~isfile(data_path)
    run(`wget https://www.dropbox.com/scl/fi/zq7p8xofbmfm7a2m0q8u6/'
        'm_train_compass.jld2 -q -O $data_path`)
end
m_train = JLD2.jldopen(data_path, "r")["m_train"];

f0 = 0.015f0
timeD = timeR = TD = 3200f0
dtD = 4f0
dtS = 4f0
nbl = 120

wavelet = ricker_wavelet(TD, dtS, f0)
wavelet = filter_data(wavelet, dtS; fmin=3f0, fmax=Inf)

d = (12.5f0, 12.5f0)
o = (0f0, 0f0)
n = (size(m_train, 1), size(m_train, 2))
# Setup model structure
nsrc = 64	# number of sources
nxrec = n[1]
snr = 12f0

# Training hyperparameters 
device = gpu

lr           = 8f-4
clipnorm_val = 3f0
noise_lev_x  = 0.1f0
noise_lev_init  = deepcopy(noise_lev_x)
noise_lev_y  = 0.0 

batch_size   = 8
n_epochs     = 200
num_post_samples = 64

save_every   = 10
plot_every   = 10
n_condmean   = 20

m_mean = mean(m_train, dims=4)[:,:,1,1]
wb = maximum(find_water_bottom(m_mean.-minimum(m_mean)))
function background_1d_average()
    m_mean = mean(m_train, dims=4)[:,:,1,1]
    wb = maximum(find_water_bottom(m_mean.-minimum(m_mean)))
    m0 = deepcopy(m_mean)
    m0[:,wb+1:end] .= 1f0./Float32.(imfilter(1f0./m_mean[:,wb+1:end], Kernel.gaussian(10)))
    return m0
end

function background_1d_gradient()
    m_1d_gradient = reshape(repeat(range(minimum(m_train), stop=maximum(m_train), length=n[2]), inner=n[1]), n)
    m0 = 1f0./Float32.(imfilter(1f0./m_1d_gradient, Kernel.gaussian(10)))
    return m0
end

m0 = background_1d_average()

n_tot_sample = size(m_train)[end]

cig_path = datadir("cig_train.jld2")
if ~isfile(cig_path)
    run(`wget https://www.dropbox.com/scl/fi/afcd4429f4nzr9ub3qb1y/'
        'cig_train.jld2 -q -O $cig_path`)
end
grad_train = JLD2.jldopen(cig_path, "r")["grad_train"];

for z = 1:n[2]
	grad_train[:,z,:,:] .*= z * d[2]
end

#normalize rtms
max_y = quantile(abs.(vec(grad_train[:,:,:,1:300])),0.9999);
grad_train ./= max_y;

num_train = 800
target_train = m_train[:,:,:,1:num_train];
Y_train      = grad_train[:,:,:,1:num_train];

target_test = m_train[:,:,:,(num_train+50):end];
Y_test      = grad_train[:,:,:,(num_train+50):end];

n_x, n_y, chan_target, n_train = size(target_train)
n_train = size(target_train)[end]
N = n_x*n_y*chan_target
chan_obs   = size(Y_train)[end-1]
chan_cond = 1

X_train  = target_train
X_test   = target_test  

vmax_v = maximum(X_train)
vmin_v = minimum(X_train)

n_batches    = cld(n_train, batch_size)-1
n_train_safe = batch_size*n_batches

# Summary network parametrs
unet_lev = 4
unet = Chain(Unet(chan_obs, chan_cond, unet_lev)|> device);
trainmode!(unet, true); 
unet = FluxBlock(unet);

# Create conditional network
L = 3
K = 9 
n_hidden = 64
low = 0.5f0

Random.seed!(123);
cond_net = NetworkConditionalGlow(chan_target, chan_cond, n_hidden,  L, K;  split_scales=true, activation=SigmoidLayer(low=low,high=1.0f0)) |> device;
G        = SummarizedNet(cond_net, unet)

# Optimizer
opt = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr))

# Training logs 
loss      = []; logdet_train = []; ssim      = []; l2_cm      = [];
loss_test = []; logdet_test  = []; ssim_test = []; l2_cm_test = [];

net_path = datadir("trained-wise-cnf.bson")
if ~isfile(net_path)
    run(`wget https://www.dropbox.com/scl/fi/o6x72s6e1chodnl8l79bd/'
        'trained-wise-cnf.bson -q -O $net_path`)
end

unet_lev = BSON.load(net_path)["unet_lev"];
n_hidden = BSON.load(net_path)["n_hidden"];
L = BSON.load(net_path)["L"];
K = BSON.load(net_path)["K"];

unet = Unet(chan_obs,1,unet_lev);
trainmode!(unet, true);
unet = FluxBlock(Chain(unet)) |> device;

cond_net = NetworkConditionalGlow(1, 1, n_hidden,  L, K;  freeze_conv=true, split_scales=true, activation=SigmoidLayer(low=0.5f0,high=1.0f0)) |> device;
G        = SummarizedNet(cond_net, unet)

Params = BSON.load(net_path)["Params"]; 
noise_lev_x = BSON.load(net_path)["noise_lev_x"]; 
set_params!(G,Params)

# Load in unet summary net
G.sum_net.model = BSON.load(net_path)["unet_model"]; 
G = G |> device;

#make non-amortized NF
n_hidden = BSON.load(net_path)["n_hidden"];
L = BSON.load(net_path)["L"];
K = BSON.load(net_path)["K"];
Random.seed!(123);
G_z = NetworkGlow(1, n_hidden,  L, K; split_scales=true, activation=SigmoidLayer(low=0.5f0,high=1.0f0))|> device;

#make initial posterior sample
batch_size = wiser_batch_size
Random.seed!(123);
Z_fix = randn(Float32, n_x,n_y,1,batch_size)|> device
test_x, test_y, file_str = X_test, Y_test, "test"
num_cols = 7
plots_len = 2
all_sampls = size(test_x)[end]-1
 
ind = (2:div(all_sampls,3):all_sampls)[1]
x = test_x[:,:,:,ind:ind] 

cmin, cmax = extrema(test_x[:,:,:,ind:ind])
function perturbv(x; vmin=cmin, vmax=cmax, period=pi*2, amplitude=amplitude)
    ### perturb velocity
    ### input range 1.48 to 4.5
    ### output range the same
	return x + sin((x-vmin)/(vmax-vmin)*period)*amplitude
end

x_gt  =  Float32.(perturbv.(x)[:,:,1,1]);
vp = x_gt;

wb = 16
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range((wb-1)*d[1], stop=(wb-1)*d[1], length=nsrc))
xsrc = convertToCell(ContJitter((n[1]-1)*d[1], nsrc))
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)
q = judiVector(srcGeometry, wavelet)

nxrec = n[1]
xrec = range(0f0, stop=(n[1]-1)*d[1], length=nxrec)
yrec = 0f0 # WE have to set the y coordiante to zero (or any number) for 2D modeling
zrec = range(d[1], stop=d[1], length=nxrec)

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

model = Model(n, d, o, (1f0./x_gt).^2f0; nb=nbl)
F = judiModeling(model, srcGeometry, recGeometry)
d_obs = F(1f0./x_gt.^2f0) * q
m_back = m0
offsetrange = range(offset_start, stop=offset_end, length=n_offsets)
J = judiExtendedJacobian(F(1f0./m_back.^2f0), q, offsetrange)
d_obs0 = F(1f0./m_back.^2f0) * q
noise_ = deepcopy(d_obs)
for l = 1:nsrc
    noise_.data[l] = randn(Float32, size(d_obs.data[l]))
    noise_.data[l] = real.(ifft(fft(noise_.data[l]).*fft(q.data[1])))
end
snr = test_snr
noise_ = noise_/norm(noise_) * norm(d_obs) * 10f0^(-snr/20f0)
d_obs = d_obs + noise_

@time rtm = J' * (d_obs0 - d_obs)

y = reshape(permutedims(rtm[keep_offset_idx, :, :], [2, 3, 1]), size(m_train, 1), size(m_train, 2), keep_offset_num,1)
for z = 1:n[2]
    y[:,z,:,:] .*= z * d[2]
end
y ./= max_y;
y .+= noise_lev_y*randn(Float32, size(y));

z_base = z_shape_simple(G_z, G_z(Z_fix)[1])

Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
_, Zy_fixed_train, _ = G.forward(Z_fix, Y_train_latent_repeat); #needs to set the proper sizes here
X_gen_init, Y_gen = G.inverse(z_base,Zy_fixed_train) |> cpu;
Zy_fixed_train_full = deepcopy(Zy_fixed_train)

#Training logs 
n_epochs_pre = 400

loss      = []; logdet_train = []; ssim      = []; l2_cm      = []; ssim = [];
loss_inner = []; l2_latent = [];

opt = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr_pre))

factor = 1f-13

Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
_, Zy_fixed_train, lgdet = G.forward(Z_fix, Y_train_latent_repeat); 

Tm = judiTopmute(model.n, wb, 5)  # Mute water column
S = judiDepthScaling(model)

## FWI specific
proj(x::AbstractArray{T}; upper=T(1f0./1.48f0.^2f0), lower=T(1f0./5f0.^2f0)) where T = max.(min.(x,T(upper)),T(lower))
nssample = 1
n_particle_sample = (op_type == "identity") ? batch_size : batch_size

vwater = cmin
v_add_water(v) = hcat(vwater * ones(Float32, n[1], wb-1), v[:,wb:end])

z_base, lgdet = G_z(Z_fix)

X_gen, Y_gen = G.inverse(z_shape_simple(G_z, z_base),Zy_fixed_train);
X_gen_cpu = X_gen |>cpu

function make_m(X_gen_cpu)
for i = 1:size(X_gen_cpu)[end]
    X_gen_cpu[:,:,1,i] = v_add_water(X_gen_cpu[:,:,1,i])
end
m_i = proj(1f0./X_gen_cpu.^2f0)
return m_i
end
m_i = make_m(X_gen_cpu)

ls = BackTracking(order=3, iterations=10)
starting_alpha = 1f-2

fwi_lr = (op_type == "identity") ? 1f0 : 5f-5
freq1 = fc ? freq1 : nothing
freq2 = fc ? freq2 : nothing
freq3 = fc ? freq3 : nothing

iter1 = fc ? iter1 : nothing
iter2 = fc ? iter2 : nothing
iter3 = fc ? iter3 : nothing

if op_type == "identity"
	global noise_ = noise_lev_x * randn(Float32, size(vp))
end

function fg(m_i::Array{Float32, 4}, e::Int64; op_type=op_type)
	gs = zeros(Float32,n_x,n_y,1,n_particle_sample);
	f_all = 0f0
	if op_type == "identity"
		for i in 1:batch_size
            f = 0.5f0 * norm(m_i[:,:,1,i]-1f0./vp.^2f0-noise_)^2f0
            g = m_i[:,:,1,i]-1f0./vp.^2f0-noise_
            gs[:,:,1,i] = g
			f_all += f
		end
	else
		rand_nss = [randperm(nsrc)[1:nssample] for i = 1:n_particle_sample]
		for i in 1:n_particle_sample
			model0 = Model(n, d, o, m_i[:,:,1,i]; nb=nbl)
			rand_ns = rand_nss[i]
			Fnow = F[rand_ns]
			qnow = filter_data(q[rand_ns]; fmin=0f0, fmax=freq_clip(e; fc=fc))
			dobsnow = filter_data(d_obs[rand_ns]; fmin=0f0, fmax=freq_clip(e; fc=fc))
			f, g = fwi_objective(model0, qnow, dobsnow)
			gs[:,:,1,i] = g.data / norm(g.data, Inf)
    		for z = 1:n[2]
        		gs[:,z,1,i] .*= z * d[2]
        		if z < wb
            		gs[:,z,1,i] .= 0
        		end
    		end
			f_all += f
		end
	end
	return f_all, gs
end

function freq_clip(j::Int64; fc=true)::Float32
    if ~fc
        return 100f0
    end
    if j<=iter1
        return freq1
    elseif j<=iter2
        return freq2
    elseif j<=iter3
        return freq3
    else
        return 100f0
    end
end

Z_fix_fix = deepcopy(Z_fix)

for e=1:n_epochs_pre # outer loop

	GC.gc()
	rand_particle = (batch_size == n_particle_sample) ? (1:batch_size) : (randperm(batch_size)[1:n_particle_sample])
    if redraw
        global Z_fix = randn(Float32, n_x,n_y,1,batch_size)|> device
    else
        global Z_fix = Z_fix_fix[:,:,:,rand_particle] |> device
    end
    _, Zy_fixed_train, lgdet = G.forward(Z_fix, repeat(y |>cpu, 1, 1, 1, n_particle_sample)|> device); 

    z_base, lgdet = G_z(Z_fix)
    z_base_init = deepcopy(z_base)

		X_gen, Y_gen = G.inverse(z_shape_simple(G_z, z_base),Zy_fixed_train);
		X_gen_cpu = X_gen |>cpu
		if resample_x
			global m_i[:,:,:,rand_particle] .= make_m(X_gen_cpu)
		end
		global v_i_before_correction = 1f0 ./ sqrt.(m_i[:,:,:,rand_particle])

		f_all, gs = fg(m_i[:,:,:,rand_particle], e; op_type=op_type)
		global m_i[:,:,:,rand_particle] .= proj(m_i[:,:,:,rand_particle] .- fwi_lr .* gs)

        global v_i = 1f0 ./ sqrt.(m_i[:,:,:,rand_particle])
		fig_name = @strdict inner_loop γ λ clipnorm_val amplitude snr freq1 freq2 freq3 iter1 iter2 iter3 fwi_lr factor n_epochs_pre lr_pre e nssample batch_size n_particle_sample

		y_plot = gs[:,:,1,1]
	    a = quantile(abs.(vec(y_plot)), 98/100)
        G.forward(Z_fix_fix, Y_train_latent_repeat); #needs to set the proper sizes here
		X_gen_cpu_before_correction = G.inverse(z_shape_simple(G_z, G_z(Z_fix_fix)[1]),Zy_fixed_train_full)[1] |> cpu
		G.forward(Z_fix, repeat(y |>cpu, 1, 1, 1, n_particle_sample)|> device);  #needs to set the proper sizes here
	
	if ~fwi_only
	for t in 1:inner_loop
		z_base, lgdet = G_z(Z_fix)

		X_gen, Y_gen  = G.inverse(z_shape_simple(G_z, z_base),Zy_fixed_train);
		X_gen_cpu = X_gen |>cpu

		gs_penalty = zeros(Float32,n_x,n_y,1,n_particle_sample);
		f_all_inner = 0
		for i in 1:n_particle_sample
			g = (X_gen_cpu[:,:,1,i] .- v_i[:,:,1,i])  
            ng = randn(Float32, size(g))
			g = g + γ * norm(g)/norm(ng)*ng
            f = norm(g)^2
			gs_penalty[:,:,:,i] =  g
			f_all_inner += f
		end
        println("Outer Iteration = ", e, ", Inner Iteration = ", t, ", weak deep prior term = ", f_all_inner)
		append!(loss_inner, f_all_inner / n_particle_sample)  # normalize by image size and batch size
		append!(logdet_train, -lgdet / N) # logdet is internally normalized by batch size
		append!(l2_latent, norm(z_base)/sqrt(length(z_base)))	# normalized l2 of the corrected latent space

		ΔX, X, ΔY = G.backward_inv(((gs_penalty ./ factor)|>device) / n_particle_sample, X_gen, Y_gen; Y_save=Y_train_latent_repeat)
        nΔX = randn(Float32, size(ΔX)) |> device
		ΔX = ΔX + γ * norm(ΔX)/norm(nΔX)*nΔX
		G_z.backward(vec(de_z_shape_simple(G_z, ΔX)) + λ * vec(z_base), vec(z_base);)

		for p in get_params(G_z) 
				Flux.update!(opt,p.data,p.grad)
		end; clear_grad!(G_z)
	end
	end
	z_base, lgdet = G_z(Z_fix)
	G.forward(Z_fix_fix, Y_train_latent_repeat); #needs to set the proper sizes here
	X_gen_cpu = G.inverse(z_shape_simple(G_z, G_z(Z_fix_fix)[1]),Zy_fixed_train_full)[1] |> cpu
	G.forward(Z_fix, repeat(y |>cpu, 1, 1, 1, n_particle_sample)|> device);  #needs to set the proper sizes here

	G_z(Z_fix);

	if(mod(e,plot_every)==0) 
	fig = figure(figsize=(14,7))
	subplot(2,3,1)
	imshow(X_gen_init[:,:,1,1]', vmin=minimum(x_gt), vmax=maximum(x_gt), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("init");#colorbar(fraction=0.046, pad=0.04);
	
	subplot(2,3,2)
	imshow(X_gen_cpu_before_correction[:,:,1,1]', vmin=minimum(x_gt), vmax=maximum(x_gt), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("generated by NF before iteration");#colorbar(fraction=0.046, pad=0.04);

	subplot(2,3,3)
	imshow(X_gen_cpu[:,:,1,1]', vmin=minimum(x_gt), vmax=maximum(x_gt), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("generated by NF after iteration");#colorbar(fraction=0.046, pad=0.04);
	
	subplot(2,3,4)
	imshow(x_gt[:,:,1,1]', vmin=minimum(x_gt), vmax=maximum(x_gt), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("gt");#colorbar(fraction=0.046, pad=0.04);
	
	subplot(2,3,5)
	imshow(v_i_before_correction[:,:,1,1]', vmin=minimum(x_gt), vmax=maximum(x_gt), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("m before correction");#colorbar(fraction=0.046, pad=0.04);

	subplot(2,3,6)
	imshow(v_i[:,:,1,1]', vmin=minimum(x_gt), vmax=maximum(x_gt), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("m after correction");#colorbar(fraction=0.046, pad=0.04);

	tight_layout()
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_inv_image.png"), fig); close(fig)
			  
	fig = figure(figsize=(14,7))
	subplot(3,3,1)
	imshow(mean(X_gen_init, dims=4)[:,:,1,1]', vmin=minimum(x_gt), vmax=maximum(x_gt), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("initial conditional mean");#colorbar(fraction=0.046, pad=0.04);
	
	subplot(3,3,2)
	imshow(mean(X_gen_cpu_before_correction, dims=4)[:,:,1,1]', vmin=minimum(x_gt), vmax=maximum(x_gt), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("before iteration conditional mean");#colorbar(fraction=0.046, pad=0.04);
	
	subplot(3,3,3)
	imshow(mean(X_gen_cpu, dims=4)[:,:,1,1]', vmin=minimum(x_gt), vmax=maximum(x_gt), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("after iteration conditional mean");#colorbar(fraction=0.046, pad=0.04);
	
	subplot(3,3,4)
	imshow(std(X_gen_init, dims=4)[:,:,1,1]', vmin=0f0, vmax=maximum(std(X_gen_init, dims=4)[:,:,1,1]), interpolation="none", cmap="magma")
	axis("off"); title("initial point-wise std");#colorbar(fraction=0.046, pad=0.04);
	
	subplot(3,3,5)
	imshow(std(X_gen_cpu_before_correction, dims=4)[:,:,1,1]', vmin=0f0, vmax=maximum(std(X_gen_init, dims=4)[:,:,1,1]), interpolation="none", cmap="magma")
	axis("off"); title("before iteration point-wise std");#colorbar(fraction=0.046, pad=0.04);

	subplot(3,3,6)
	imshow(std(X_gen_cpu, dims=4)[:,:,1,1]', vmin=0f0, vmax=maximum(std(X_gen_init, dims=4)[:,:,1,1]), interpolation="none", cmap="magma")
	axis("off"); title("after iteration point-wise std");#colorbar(fraction=0.046, pad=0.04);

	subplot(3,3,7)
	imshow(abs.(x_gt[:,:,1,1]'.- mean(X_gen_init, dims=4)[:,:,1,1]'), vmin=0f0, vmax=maximum(abs.(x_gt[:,:,1,1]'.- mean(X_gen_init, dims=4)[:,:,1,1]')), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("initial conditional mean error");#colorbar(fraction=0.046, pad=0.04);
	
	subplot(3,3,8)
	imshow(abs.(x_gt[:,:,1,1]'.- mean(X_gen_cpu_before_correction, dims=4)[:,:,1,1]'), vmin=0f0, vmax=maximum(abs.(x_gt[:,:,1,1]'.- mean(X_gen_init, dims=4)[:,:,1,1]')), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("before iteration conditional mean error");#colorbar(fraction=0.046, pad=0.04);
	
	subplot(3,3,9)
	imshow(abs.(x_gt[:,:,1,1]'.- mean(X_gen_cpu, dims=4)[:,:,1,1]'), vmin=0f0, vmax=maximum(abs.(x_gt[:,:,1,1]'.- mean(X_gen_init, dims=4)[:,:,1,1]')), interpolation="none", cmap="cet_rainbow4")
	axis("off"); title("after iteration conditional mean error");#colorbar(fraction=0.046, pad=0.04);
	
	tight_layout()
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_UQ.png"), fig); close(fig)
	end
		# Loss function is l2 norm
        ssim_now = assess_ssim(mean(X_gen_cpu_before_correction, dims=4)[:,:,1,1], vp)
		append!(ssim, ssim_now)
		append!(loss, f_all / n_particle_sample)  # normalize by image size and batch size
		append!(logdet_train, -lgdet / N) # logdet is internally normalized by batch size

	if(mod(e,plot_every)==0)
		fig = figure("training logs ", figsize=(10,12))
subplot(5,1,1); title("L2 Term: train="*string(loss[end]))
plot(range(1f0, e, length=length(loss)), loss);
xlabel("Parameter Update"); legend();

if ~fwi_only
subplot(5,1,2); title("Weak Fitting Term: train="*string(loss_inner[end]))
plot(range(1f0, e, length=length(loss_inner)), loss_inner);
xlabel("Parameter Update"); legend();
end

subplot(5,1,3); title("SSIM Term: train="*string(ssim[end]))
plot(range(1f0, e, length=length(ssim)), ssim);
xlabel("Parameter Update"); legend();

if ~fwi_only
subplot(5,1,4); title("Logdet Term: train="*string(logdet_train[end]))
plot(range(1f0, e, length=length(logdet_train)),logdet_train);
xlabel("Parameter Update") ;

subplot(5,1,5); title("l2 of latent space="*string(l2_latent[end]))
plot(range(1f0, e, length=length(l2_latent)),l2_latent);
axhline(y=1,color="red",linestyle="--",label="Normal Noise")
xlabel("Parameter Update") ;
end

tight_layout()
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)

fig = figure(figsize=(14,7))
subplot(1,3,1)
imshow(Z_fix[:,:,1,1]' |> cpu, vmin=-3, vmax=3, interpolation="none", cmap="seismic")
axis("off"); title("input white noise");#colorbar(fraction=0.046, pad=0.04);
subplot(1,3,2)
imshow(z_shape_simple(G_z, z_base_init)[:,:,1,1]' |> cpu, vmin=-3, vmax=3, interpolation="none", cmap="seismic")
axis("off"); title("initial output noise");#colorbar(fraction=0.046, pad=0.04);
subplot(1,3,3)
imshow(z_shape_simple(G_z, z_base)[:,:,1,1]' |> cpu, vmin=-3, vmax=3, interpolation="none", cmap="seismic")
axis("off"); title("refined output noise");#colorbar(fraction=0.046, pad=0.04);
tight_layout()
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_latent.png"), fig); close(fig)


if op_type != "identity"
    ## butterfly plot
    dobsnow = filter_data(d_obs; fmin=0f0, fmax=freq_clip(e; fc=fc)).data[1]
    dobsclean = filter_data(d_obs-noise_; fmin=0f0, fmax=freq_clip(e; fc=fc)).data[1]
    dobspred = filter_data(F[1](make_m(X_gen_cpu)[:,:,1,1]) * q[1]; fmin=0f0, fmax=freq_clip(e; fc=fc)).data[1]
    dobsdiff = dobsnow - dobspred
    cmin, cmax = [-1, 1] * quantile(abs.(vec(dobsnow)),0.92)
    fig = figure(figsize=(20,12));
    imshow(hcat(dobsclean, reverse(dobsnow, dims=2), dobspred, reverse(dobsdiff, dims=2)), vmin=cmin, vmax=cmax, cmap="Greys")
    axis("off")
    title("FWI at iter $(e): noise-free data, noisy data, predicted data, difference")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_butterfly.png"), fig); close(fig)
    close(fig)
end
end

		print("Iter: epoch=", e, "/", n_epochs_pre, 
		    "; f l2 = ",  loss[end], 
		    "; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "\n")
		Base.flush(Base.stdout)

		if(mod(e,save_every)==0) 
			G_save = deepcopy(G_z);
			Params = get_params(G_z) |> cpu;
			save_dict = @strdict inner_loop γ λ clipnorm_val amplitude snr freq1 freq2 freq3 iter1 iter2 iter3 fwi_lr X_gen_cpu Params Z_fix Z_fix_fix factor n_epochs_pre lr_pre e nssample batch_size n_particle_sample m_i
			@tagsave(
				joinpath(datadir(), sim_name, savename(save_dict, "bson"; digits=6)),
				save_dict;
				safe=true
			);
		end
end
